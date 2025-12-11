# generative_agent/ppo_trainer/reward.py
import os
import sys
import numpy as np
import tempfile
import subprocess
import json
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pathlib import Path

# Fix paths to find the Oracle
CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from generative_agent.evaluation_engine.oracle import Oracle

# --- CONFIGURATION ---
RELAXER_SCRIPT = PROJECT_ROOT / "generative_agent" / "evaluation_engine" / "relax_worker.py"

class RewardCalculator:
    def __init__(self):
        print("[RewardCalculator] Initializing Oracle Bridges...")
        # We pass None for env names to skip conda checks in Oracle if implemented there
        self.oracle = Oracle(matgl_env_name=None, megnet_env_name=None)

    def get_rewards(self, structures: list) -> list:
        if not structures:
            return []

        rewards = [-5.0] * len(structures)
        
        # 1. BATCH RELAXATION
        print(f"   > Relaxing {len(structures)} structures...")
        relaxed_results = self._batch_relax(structures)
        
        stable_structures = []
        stable_indices = []

        # 2. FILTER
        for i, res in enumerate(relaxed_results):
            if res and res.get("is_converged"):
                stable_structures.append(res["structure"])
                stable_indices.append(i)

        if not stable_structures:
            print("   > No stable structures found in this batch.")
            return rewards

        # 3. ORACLE PREDICTION
        print(f"   > Predicting properties for {len(stable_structures)} stable crystals...")
        try:
            properties = self.oracle.predict_properties(stable_structures)
        except Exception as e:
            print(f"   [Warning] Oracle prediction failed: {e}")
            return rewards
        
        for i, props in enumerate(properties):
            original_idx = stable_indices[i]
            
            e_form = props.get("formation_energy")
            band_gap = props.get("band_gap")
            
            if e_form is None or band_gap is None:
                continue 
                
            # A. STABILITY GATE
            if e_form > 0.05: 
                r_stability = -5.0
            else:
                r_stability = max(0.0, -0.5 * e_form)
                
            if r_stability < 0:
                rewards[original_idx] = r_stability
                continue

            # B. ELECTRONIC TARGET (1.4 eV)
            target_gap = 1.4
            sigma = 0.5 
            r_electronic = 5.0 * np.exp(-((band_gap - target_gap)**2) / (2 * sigma**2))
            
            # C. TOTAL
            total_score = r_stability + r_electronic
            rewards[original_idx] = max(-5.0, min(10.0, total_score))
            
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

            if not RELAXER_SCRIPT.exists():
                print(f"[Error] Relaxer script not found at {RELAXER_SCRIPT}")
                return results

            # --- FIX: Execute directly with current Python ---
            cmd = [sys.executable, str(RELAXER_SCRIPT), tmp_out] + cifs
            
            try:
                # 5 minute timeout for relaxation
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
                
                if not proc.stdout.strip():
                    return results

                try:
                    data = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    print(f"[Reward] JSON Decode Error. Output snippet: {proc.stdout[:100]}")
                    return results
                
                for item in data:
                    fname = os.path.basename(item.get("input_file", ""))
                    try:
                        idx = int(fname.split('.')[0])
                    except:
                        continue
                    
                    if item.get("is_converged"):
                        try:
                            s = Structure.from_file(item["output_file"])
                            results[idx] = {"is_converged": True, "structure": s}
                        except:
                            results[idx] = {"is_converged": False}
                    else:
                        results[idx] = {"is_converged": False}
            
            except subprocess.CalledProcessError as e:
                print(f"[Reward] Relaxer subprocess failed. Code: {e.returncode}")
                # Print stderr to diagnose if libraries are missing
                print(f"Stderr: {e.stderr[:500]}")
            except subprocess.TimeoutExpired:
                print("[Reward] Relaxer timed out.")
            except Exception as e:
                print(f"[Reward] Relaxer unexpected error: {e}")
                
        return results
