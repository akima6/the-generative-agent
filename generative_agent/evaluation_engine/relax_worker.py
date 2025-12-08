# generative_agent/evaluation_engine/relax_worker.py
# Runs in 'matgl_env'

import sys
import os
import json

def main():
    results = []
    args = sys.argv[1:]
    
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
        import warnings
        # Suppress DGL and Matgl warnings
        warnings.filterwarnings("ignore")
        
        # Import Relaxer only after setup
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
                
                # --- SAFETY CHECK ---
                # Check density to prevent "no valid edges" crash in DGL
                if struct.density < 0.1: # Extremely low density
                    info["error"] = "Structure too sparse (density < 0.1)"
                    results.append(info)
                    continue
                if len(struct) < 2: # Single atom (cannot calculate bonds)
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
                # Catch standard Python errors
                info["error"] = str(e)
                results.append(info)

    except Exception as e:
        # Catastrophic failure catch for the whole script
        results = [{"error": f"Worker crashed: {str(e)}"}]

    # The ONLY print to stdout. 
    # If results is empty, print empty list to avoid JSON error.
    print(json.dumps(results if results else []))

if __name__ == "__main__":
    main()