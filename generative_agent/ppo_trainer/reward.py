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
        self.oracle = Oracle(matgl_env_name=None, megnet_env_name=None)

    def get_rewards(self, structures: list) -> list:
        if not structures: return []
        rewards = [-5.0] * len(structures)
        
        # 1. BATCH RELAXATION
        print(f"   > Relaxing {len(structures)} structures...")
        relaxed_results = self._batch_relax(structures)
        
        stable_structures = []
        stable_indices = []

        for i, res in enumerate(relaxed_results):
            if res and res.get("is_converged"):
                stable_structures.append(res["structure"])
                stable_indices.append(i)

        if not stable_structures:
            print("   > No stable structures found in this batch.")
            # FIX: Return early, but the crucial diagnostic is in _batch_relax
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
            
            if e_form is None or band_gap is None: continue 
                
            # Reward Logic
            if e_form > 0.05: 
                r_stability = -5.0
            else:
                r_stability = max(0.0, -0.5 * e_form)
                
            if r_stability < 0:
                rewards[original_idx] = r_stability
                continue

            target_gap = 1.4
            sigma = 0.5 
            r_electronic = 5.0 * np.exp(-((band_gap - target_gap)**2) / (2 * sigma**2))
            
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

            cmd = [sys.executable, str(RELAXER_SCRIPT), tmp_out] + cifs
            
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
                
                # --- DIAGNOSTIC FIX ---
                if not proc.stdout.strip():
                    # Check for subprocess error even if check=True didn't catch it
                    if proc.stderr:
                         print(f"[Reward] Relaxer returned EMPTY STDOUT. Full STDERR:\n{proc.stderr}")
                    return results

                try:
                    data = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    print(f"[Reward] Relaxer JSON Decode Error. Output snippet: {proc.stdout[:100]}...")
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
                            results[idx] = {"is_converged": False, "error": "Pymatgen read failed"}
                    else:
                        results[idx] = {"is_converged": False, "error": item.get('error', 'Relaxation did not converge')}
            
            except subprocess.CalledProcessError as e:
                # Catch if the subprocess failed
                print(f"[Reward] Relaxer SUBPROCESS FAILED. Code: {e.returncode}")
                print(f"Stderr:\n{e.stderr[:1000]}") # Print 1000 chars of error
            except subprocess.TimeoutExpired:
                print("[Reward] Relaxer timed out.")
            except Exception as e:
                print(f"[Reward] Relaxer unexpected error in main process: {e}")
                
        return results
