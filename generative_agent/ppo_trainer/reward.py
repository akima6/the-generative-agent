# generative_agent/ppo_trainer/reward.py
import os
import sys
import numpy as np
import tempfile
import subprocess
import json
import configparser
from pymatgen.io.cif import CifWriter
from pathlib import Path

# --- CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Paths from file
ENV_PATH_FILE = PROJECT_ROOT / "env_paths.txt"
RELAXER_SCRIPT = PROJECT_ROOT / "generative_agent" / "evaluation_engine" / "relax_worker.py"

from generative_agent.evaluation_engine.oracle import Oracle

# Load environment paths
MATGL_ENV_PATH = None
MEGNET_ENV_PATH = None

def load_env_paths():
    global MATGL_ENV_PATH, MEGNET_ENV_PATH
    config = configparser.ConfigParser()
    try:
        with open(ENV_PATH_FILE, 'r') as f:
            data = dict(line.strip().split('=', 1) for line in f)
        MATGL_ENV_PATH = data.get('ENV_B_PATH')
        MEGNET_ENV_PATH = data.get('ENV_C_PATH')
    except Exception as e:
        print(f"[Reward] Warning: Could not load env paths: {e}")

load_env_paths()

class RewardCalculator:
    def __init__(self):
        print("[RewardCalculator] Initializing...")
        self.oracle = Oracle(matgl_env_name=MATGL_ENV_PATH, megnet_env_name=MEGNET_ENV_PATH)
        self.relaxer_script = str(RELAXER_SCRIPT)
        
        # This list will hold dictionaries of details for the most recent batch
        # Keys: 'formula', 'initial_E', 'final_E', 'band_gap', 'formation_E', 'valid'
        self.last_batch_stats = []

    def get_rewards(self, structures: list) -> list:
        """
        Calculates rewards and populates self.last_batch_stats with detailed logs.
        """
        self.last_batch_stats = [] # Reset stats
        
        if not structures:
            return []

        rewards = [-5.0] * len(structures)
        
        # Initialize stats with defaults
        for s in structures:
            stat = {
                "formula": s.composition.reduced_formula if s else "INVALID",
                "valid": False,
                "initial_E": None,
                "final_E": None,
                "band_gap": None,
                "formation_E": None,
                "reward": -5.0
            }
            self.last_batch_stats.append(stat)

        # 1. RELAXATION
        relaxed_results = self._batch_relax(structures)
        
        stable_structures = []
        stable_indices = []

        for i, res in enumerate(relaxed_results):
            if res and res.get("is_converged"):
                # Update stats
                self.last_batch_stats[i]["valid"] = True
                self.last_batch_stats[i]["initial_E"] = res.get("initial_energy")
                self.last_batch_stats[i]["final_E"] = res.get("final_energy")
                
                stable_structures.append(res["structure"])
                stable_indices.append(i)
            elif res:
                # Even if failed, log energies if available
                self.last_batch_stats[i]["initial_E"] = res.get("initial_energy")
                self.last_batch_stats[i]["final_E"] = res.get("final_energy")

        # 2. PROPERTIES (Oracle)
        if stable_structures:
            properties = self.oracle.predict_properties(stable_structures)
            
            for i, props in enumerate(properties):
                idx = stable_indices[i]
                
                ef = props.get("formation_energy")
                eg = props.get("band_gap")
                
                self.last_batch_stats[idx]["formation_E"] = ef
                self.last_batch_stats[idx]["band_gap"] = eg
                
                if ef is None or eg is None:
                    continue

                # --- REWARD LOGIC ---
                # Target: Ef < 0 (Stable) AND 0.5 < Eg < 3.0 (Semiconductor)
                
                # Stability Score
                if ef > 0: 
                    r_stability = -5.0 # Unstable
                else:
                    # Bonus for being very stable (up to 2.0)
                    r_stability = min(2.0, -1.0 * ef)

                # Band Gap Score (Gaussian centered at 1.5 eV)
                # 0.5 to 3.0 is the acceptance window
                if 0.5 < eg < 3.0:
                    # Peak at 1.5 eV
                    r_electronic = 5.0 * np.exp(-((eg - 1.5)**2) / (2 * 0.5**2))
                else:
                    r_electronic = -2.0 # Penalty for being Metal or Insulator

                total_score = r_stability + r_electronic
                
                # Clamp
                final_score = max(-5.0, min(10.0, total_score))
                
                rewards[idx] = final_score
                self.last_batch_stats[idx]["reward"] = final_score

        return rewards

    def _batch_relax(self, structures):
        results = [None] * len(structures)
        valid_indices = [i for i, s in enumerate(structures) if s is not None]
        if not valid_indices: return results

        with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
            cifs = []
            for idx in valid_indices:
                p = os.path.join(tmp_in, f"{idx}.cif")
                try:
                    CifWriter(structures[idx]).write_file(p)
                    cifs.append(p)
                except:
                    pass
            
            if not cifs: return results

            cmd = ["conda", "run", "-p", MATGL_ENV_PATH, "python", self.relaxer_script, tmp_out] + cifs
            
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(proc.stdout)
                
                for item in data:
                    fname = os.path.basename(item.get("input_file", ""))
                    if not fname: continue
                    idx = int(fname.split('.')[0])
                    
                    res_entry = {
                        "is_converged": item.get("is_converged", False),
                        "initial_energy": item.get("initial_energy"),
                        "final_energy": item.get("final_energy"),
                        "error": item.get("error")
                    }
                    
                    if res_entry["is_converged"]:
                        try:
                            s = Structure.from_file(item["output_file"])
                            res_entry["structure"] = s
                        except:
                            res_entry["is_converged"] = False
                            
                    results[idx] = res_entry
            except Exception as e:
                print(f"[Reward] Relaxer failed: {e}")
                
        return results
