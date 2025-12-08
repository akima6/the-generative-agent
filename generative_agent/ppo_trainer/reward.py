# generative_agent/ppo_trainer/reward.py
import os
import sys
import numpy as np
import tempfile
import subprocess
import json
from pymatgen.io.cif import CifWriter
from pathlib import Path

# Fix paths to find the Oracle
CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from generative_agent.evaluation_engine.oracle import Oracle

# --- CONFIGURATION ---
MATGL_ENV = "matgl_env"
MEGNET_ENV = "megnet_legacy_env"
RELAXER_SCRIPT = PROJECT_ROOT / "generative_agent" / "evaluation_engine" / "relax_worker.py"

class RewardCalculator:
    def __init__(self):
        print("[RewardCalculator] Initializing Oracle Bridges...")
        # The Oracle manages the connections to the GPU workers (Environment B & C)
        self.oracle = Oracle(matgl_env_name=MATGL_ENV, megnet_env_name=MEGNET_ENV)

    def get_rewards(self, structures: list) -> list:
        """
        Composite Reward Function for Semiconductor Discovery.
        Formula: Reward = Stability_Score + Electronic_Score
        """
        if not structures:
            return []

        # Default reward for failure (e.g., broken crystal) is -5.0
        rewards = [-5.0] * len(structures)
        
        # 1. BATCH RELAXATION (Physical Reality Check)
        print(f"   > Relaxing {len(structures)} structures...")
        relaxed_results = self._batch_relax(structures)
        
        stable_structures = []
        stable_indices = []

        # 2. FILTER: Only predict properties for valid, relaxed structures
        for i, res in enumerate(relaxed_results):
            if res and res.get("is_converged"):
                stable_structures.append(res["structure"])
                stable_indices.append(i)
            # else: Reward stays -5.0

        if not stable_structures:
            print("   > No stable structures found in this batch.")
            return rewards

        # 3. ORACLE PREDICTION (Band Gap & Energy)
        print(f"   > Predicting properties for {len(stable_structures)} stable crystals...")
        properties = self.oracle.predict_properties(stable_structures)
        
        for i, props in enumerate(properties):
            original_idx = stable_indices[i]
            
            e_form = props.get("formation_energy")
            band_gap = props.get("band_gap")
            
            if e_form is None or band_gap is None:
                continue # Keep default penalty
                
            # --- THE "BRAIN" OF THE SEARCH (Reward Logic) ---
            
            # A. STABILITY GATE
            # If Formation Energy > 0, it's thermodynamically unstable (explodes/decomposes).
            # We punish it hard so the AI stops making it.
            if e_form > 0.05: 
                r_stability = -5.0
            else:
                # If stable, give a small bonus (0 to 1.0)
                r_stability = max(0.0, -0.5 * e_form)
                
            # If it's unstable, don't bother checking band gap.
            if r_stability < 0:
                rewards[original_idx] = r_stability
                continue

            # B. ELECTRONIC TARGET (The Semiconductor Sweet Spot)
            # We want Band Gap approx 1.4 eV (Ideal for Solar/Chips)
            # We use a Bell Curve (Gaussian). 
            # 1.4 eV = Max Reward (+5.0)
            # 0.0 eV (Metal) = Low Reward
            # 6.0 eV (Insulator) = Low Reward
            target_gap = 1.4
            sigma = 0.5 # Width of the "sweet spot"
            r_electronic = 5.0 * np.exp(-((band_gap - target_gap)**2) / (2 * sigma**2))
            
            # C. TOTAL REWARD
            total_score = r_stability + r_electronic
            
            # Clip limits to prevent math explosions
            rewards[original_idx] = max(-5.0, min(10.0, total_score))
            
        return rewards

    def _batch_relax(self, structures):
        """Helper: Sends structures to the subprocess worker for relaxation."""
        results = [None] * len(structures)
        
        # Identify valid inputs (not None)
        valid_indices = [i for i, s in enumerate(structures) if s is not None]
        if not valid_indices: return results

        # Create temp directory for IO
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

            # CALL THE WORKER (Environment B)
            cmd = ["conda", "run", "-n", MATGL_ENV, "python", str(RELAXER_SCRIPT), tmp_out] + cifs
            
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(proc.stdout)
                
                for item in data:
                    # Map back output filename "idx.cif" to list index
                    fname = os.path.basename(item.get("input_file", ""))
                    idx = int(fname.split('.')[0])
                    
                    if item.get("is_converged"):
                        try:
                            s = Structure.from_file(item["output_file"])
                            results[idx] = {"is_converged": True, "structure": s}
                        except:
                            results[idx] = {"is_converged": False}
                    else:
                        results[idx] = {"is_converged": False}
            except Exception as e:
                print(f"[Reward] Relaxer failed: {e}")
                
        return results