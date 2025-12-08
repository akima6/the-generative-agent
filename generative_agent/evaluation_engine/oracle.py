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
    def __init__(self, matgl_env_name="matgl_env", megnet_env_name="megnet_legacy_env"):
        self.matgl_env = matgl_env_name
        self.megnet_env = megnet_env_name
        
        # Locate workers relative to this file
        current_dir = Path(__file__).parent
        self.fe_script = current_dir / "workers" / "predict_formation_energy.py"
        self.bg_script = current_dir / "workers" / "predict_band_gap.py"

    def _run_worker(self, env, script, cifs):
        # The FIX: "python" instead of sys.executable
        cmd = ["conda", "run", "-n", env, "python", str(script)] + cifs
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(res.stdout)
        except Exception as e:
            print(f"Bridge Error ({script.name}): {e}")
            if 'res' in locals(): print(f"Stderr: {res.stderr}")
            return []

    def predict_properties(self, structures):
        if not structures: return []
        
        with tempfile.TemporaryDirectory() as tmp:
            cifs = []
            for i, s in enumerate(structures):
                p = os.path.join(tmp, f"{i}.cif")
                CifWriter(s).write_file(p)
                cifs.append(p)
            
            fe_res = self._run_worker(self.matgl_env, self.fe_script, cifs)
            bg_res = self._run_worker(self.megnet_env, self.bg_script, cifs)
            
            # Map results
            fe_map = {r.get("file_path"): r for r in fe_res}
            bg_map = {r.get("file_path"): r for r in bg_res}
            
            final = []
            for p in cifs:
                final.append({
                    "formation_energy": fe_map.get(p, {}).get("formation_energy"),
                    "band_gap": bg_map.get(p, {}).get("band_gap"),
                    "error": fe_map.get(p, {}).get("error") or bg_map.get(p, {}).get("error")
                })
            return final