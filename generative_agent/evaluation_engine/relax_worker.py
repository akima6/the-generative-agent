# generative_agent/evaluation_engine/relax_worker.py
# Runs in 'matgl_env'

import sys
import os
import json
from pathlib import Path

def main():
    # FIX: Add parent directory to sys.path to find relaxer.py
    # relax_worker.py is in 'evaluation_engine'. We need 'evaluation_engine' to be in path.
    # Actually, we need the CURRENT_DIR (evaluation_engine) to be in path.
    # The script is run as python path/to/relax_worker.py
    # So we need to add the directory it is in.
    worker_dir = str(Path(os.path.abspath(__file__)).parent)
    if worker_dir not in sys.path:
        sys.path.append(worker_dir)
        
    results = []
    args = sys.argv[1:]
    
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
        import warnings
        # Suppress DGL and Matgl warnings
        warnings.filterwarnings("ignore")
        
        # FIX: Import Relaxer
        # The script name is 'relaxer.py', so import as module 'relaxer'
        from relaxer import Relaxer 
        
        if len(args) < 2:
            print(json.dumps([]))
            return

        output_dir = args[0]
        input_paths = args[1:]
        os.makedirs(output_dir, exist_ok=True)

        # FIX: Also check if parent directory is needed for other imports (e.g. matgl)
        # Assuming required dependencies (matgl, ase) are installed in the main environment.
        
        relaxer = Relaxer()

        for fp in input_paths:
            info = {"input_file": fp, "is_converged": False}
            try:
                struct = Structure.from_file(fp)
                
                # --- SAFETY CHECK ---
                if struct.density < 0.1: 
                    info["error"] = "Structure too sparse (density < 0.1)"
                    results.append(info)
                    continue
                if len(struct) < 2: 
                    info["error"] = "Structure has too few atoms (< 2)"
                    results.append(info)
                    continue
                # --------------------

                res = relaxer.relax(struct)
                
                info["is_converged"] = bool(res.get('is_converged', False))

                if info["is_converged"]:
                    base_name = os.path.basename(fp)
                    out_path = os.path.join(output_dir, f"relaxed_{base_name}")
                    CifWriter(res['final_structure']).write_file(out_path)
                    info["output_file"] = out_path
                
                results.append(info)
            except Exception as e:
                # FIX: Print full error for debugging in the log
                info["error"] = f"Relaxation failed with error: {str(e)}"
                results.append(info)

    except Exception as e:
        # Catastrophic failure catch for the whole script
        results = [{"error": f"Worker crashed (Import/Setup): {str(e)}", "input_file": "N/A"}]

    # The ONLY print to stdout. 
    print(json.dumps(results if results else []))

if __name__ == "__main__":
    main()
