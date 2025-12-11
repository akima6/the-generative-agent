# generative_agent/evaluation_engine/relax_worker.py
# Runs in current python environment (not conda)

import sys
import os
import json
from pathlib import Path

def main():
    # FIX: Ensure all local imports work in the subprocess environment
    worker_dir = str(Path(os.path.abspath(__file__)).parent)
    if worker_dir not in sys.path:
        sys.path.append(worker_dir)
        
    results = []
    args = sys.argv[1:]
    
    # Catch all errors and print them in the JSON output
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
        import warnings
        warnings.filterwarnings("ignore")
        
        # Import Relaxer (this will fail if matgl/ase are not installed)
        from relaxer import Relaxer 
        
        if len(args) < 2:
            print(json.dumps([]))
            return

        output_dir = args[0]
        input_paths = args[1:]
        os.makedirs(output_dir, exist_ok=True)

        relaxer = Relaxer()

        for fp in input_paths:
            info = {"input_file": fp, "is_converged": False}
            try:
                struct = Structure.from_file(fp)
                
                # --- SAFETY CHECK (Keep as is) ---
                if struct.density < 0.1 or len(struct) < 2: 
                    info["error"] = "Structure too sparse or too few atoms"
                    results.append(info)
                    continue

                res = relaxer.relax(struct)
                
                info["is_converged"] = bool(res.get('is_converged', False))

                if info["is_converged"]:
                    base_name = os.path.basename(fp)
                    out_path = os.path.join(output_dir, f"relaxed_{base_name}")
                    CifWriter(res['final_structure']).write_file(out_path)
                    info["output_file"] = out_path
                
                results.append(info)
            except Exception as e:
                # Catch per-structure errors
                info["error"] = f"Relaxation failed with error: {str(e)}"
                results.append(info)

    except Exception as e:
        # Catastrophic failure catch for the whole script (ImportError)
        # This will be printed to STDOUT, which the main script will try to parse
        results = [{"error": f"Worker crashed (Import/Setup): {str(e)}", "input_file": "N/A"}]

    # The ONLY print to stdout. 
    print(json.dumps(results if results else []))

if __name__ == "__main__":
    main()
