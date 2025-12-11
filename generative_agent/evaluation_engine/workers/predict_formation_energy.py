# generative_agent/evaluation_engine/workers/predict_formation_energy.py
import sys, json, os

def main():
    results = []
    args = sys.argv[1:]
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    try:
        import warnings
        warnings.filterwarnings("ignore")
        import matgl
        from pymatgen.core import Structure
        
        # Using a newer, highly stable MatGL model for formation energy
        model = matgl.load_model("M3GNet-MP-2021.2.8-Eform")
        
        for fp in args:
            info = {"file_path": fp, "formation_energy": None, "error": None}
            try:
                s = Structure.from_file(fp)
                val = model.predict_structure(s).item()
                info["formation_energy"] = float(val)
                results.append(info)
            except Exception as e:
                info["error"] = f"Prediction failed with error: {str(e)}"
                results.append(info)

    except Exception as e:
        results = [{"error": f"Worker crashed (Import/Setup): {str(e)}", "file_path": "N/A"}]

    print(json.dumps(results if results else []))

if __name__ == "__main__": main()
    
