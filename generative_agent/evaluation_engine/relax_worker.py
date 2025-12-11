# generative_agent/evaluation_engine/relax_worker.py
import sys
import os
import json
from pathlib import Path

def main():
    # CRITICAL FIX: Ensure worker directory is in path for local imports (like relaxer.py)
    worker_dir = str(Path(os.path.abspath(__file__)).parent)
    if worker_dir not in sys.path:
        sys.path.append(worker_dir)
        
    results = []
    args = sys.argv[1:]
    
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
        import warnings
        warnings.filterwarnings("ignore")
        
        # This relies on matgl/ase being installed in the main environment
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
                info["error"] = f"Relaxation failed with error: {str(e)}"
                results.append(info)

    except Exception as e:
        # Catastrophic failure catch (ImportError, etc.)
        results = [{"error": f"Worker crashed (Import/Setup): {str(e)}", "input_file": "N/A"}]

    print(json.dumps(results if results else []))

if __name__ == "__main__":
    main()
