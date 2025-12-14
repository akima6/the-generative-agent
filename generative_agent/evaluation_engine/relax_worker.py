# generative_agent/evaluation_engine/relax_worker.py
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
        warnings.filterwarnings("ignore")
        
        # Import local relaxer
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
                
                # Basic sanity checks
                if struct.density < 0.1: 
                    info["error"] = "Density too low"
                    results.append(info)
                    continue
                
                # Perform Relaxation
                res = relaxer.relax(struct)
                
                if "error" in res:
                    info["error"] = res["error"]
                else:
                    info["is_converged"] = bool(res.get('is_converged', False))
                    info["initial_energy"] = res.get('initial_energy_per_atom')
                    info["final_energy"] = res.get('final_energy_per_atom')
                    
                    if info["is_converged"]:
                        base_name = os.path.basename(fp)
                        out_path = os.path.join(output_dir, f"relaxed_{base_name}")
                        CifWriter(res['final_structure']).write_file(out_path)
                        info["output_file"] = out_path
                
                results.append(info)
            except Exception as e:
                info["error"] = str(e)
                results.append(info)

    except Exception as e:
        results = [{"error": f"Worker Crashed: {str(e)}"}]

    print(json.dumps(results if results else []))

if __name__ == "__main__":
    main()
