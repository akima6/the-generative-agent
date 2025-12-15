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
        # We accept the args to keep compatibility with old calls, 
        # but we ignore them in the Single-Env setup.
        
        # Locate workers relative to this file
        current_dir = Path(__file__).parent
        self.fe_script = current_dir / "workers" / "predict_formation_energy.py"
        self.bg_script = current_dir / "workers" / "predict_band_gap.py"

    def _run_worker(self, script, cifs):
        """
        Runs the worker script using the CURRENT Python environment.
        Replaces 'conda run' with 'sys.executable'.
        """
        # CMD: /path/to/python script.py file1.cif file2.cif ...
        cmd = [sys.executable, str(script)] + cifs
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(res.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[Oracle Error] Worker {script.name} failed.")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return []
        except json.JSONDecodeError:
            print(f"[Oracle Error] Worker {script.name} returned invalid JSON.")
            print(f"STDOUT: {res.stdout}")
            return []
        except Exception as e:
            print(f"[Oracle Error] Unexpected: {e}")
            return []

    def predict_properties(self, structures):
        if not structures: return []
        
        with tempfile.TemporaryDirectory() as tmp:
            cifs = []
            for i, s in enumerate(structures):
                p = os.path.join(tmp, f"{i}.cif")
                try:
                    CifWriter(s).write_file(p)
                    cifs.append(p)
                except:
                    pass
            
            if not cifs: return []

            # 1. Run Formation Energy Worker (MatGL)
            fe_res = self._run_worker(self.fe_script, cifs)
            
            # 2. Run Band Gap Worker (MEGNET)
            bg_res = self._run_worker(self.bg_script, cifs)
            
            # Map results by file path
            fe_map = {r.get("file_path"): r for r in fe_res}
            bg_map = {r.get("file_path"): r for r in bg_res}
            
            final = []
            for p in cifs:
                # Combine results
                final.append({
                    "formation_energy": fe_map.get(p, {}).get("formation_energy"),
                    "band_gap": bg_map.get(p, {}).get("band_gap"),
                    "error": fe_map.get(p, {}).get("error") or bg_map.get(p, {}).get("error")
                })
            return final
