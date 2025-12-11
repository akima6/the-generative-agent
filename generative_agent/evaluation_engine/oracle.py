# generative_agent/evaluation_engine/oracle.py
import subprocess
import json
import os
import sys
import tempfile
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

class Oracle:
    def __init__(self, matgl_env_name=None, megnet_env_name=None):
        # Env names are kept for interface compatibility but ignored
        
        # Locate workers relative to this file
        current_dir = Path(__file__).parent
        self.fe_script = current_dir / "workers" / "predict_formation_energy.py"
        self.bg_script = current_dir / "workers" / "predict_band_gap.py"

    def _run_worker(self, script, cifs):
        # FIX: Execute directly with current Python instead of Conda
        cmd = [sys.executable, str(script)] + cifs
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(res.stdout)
        except Exception as e:
            print(f"Bridge Error ({script.name}): {e}")
            # If failed, return empty to handle gracefully
            if 'res' in locals(): 
                print(f"Stderr: {res.stderr[:200]}") # Print first 200 chars of error
            return []

    def predict_properties(self, structures):
        if not structures: return []
        
        with tempfile.TemporaryDirectory() as tmp:
            cifs = []
            valid_indices = []
            
            # Write CIFs
            for i, s in enumerate(structures):
                p = os.path.join(tmp, f"{i}.cif")
                try:
                    CifWriter(s).write_file(p)
                    cifs.append(p)
                    valid_indices.append(i)
                except:
                    pass
            
            if not cifs: return []

            # Run workers
            # We run them sequentially on the current GPU/CPU
            fe_res = self._run_worker(self.fe_script, cifs)
            bg_res = self._run_worker(self.bg_script, cifs)
            
            # Map results back to file paths
            fe_map = {r.get("file_path"): r for r in fe_res}
            bg_map = {r.get("file_path"): r for r in bg_res}
            
            final = []
            # match input structure order
            # Note: The logic in RewardCalculator iterates over the *returned* list
            # corresponding to the *stable_structures* list passed in.
