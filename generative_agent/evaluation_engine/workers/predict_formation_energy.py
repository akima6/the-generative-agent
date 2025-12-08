# generative_agent/evaluation_engine/workers/predict_formation_energy.py
# Runs in 'matgl_env'
import sys, json

def main():
    results = []
    cif_paths = sys.argv[1:]
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import matgl
        from pymatgen.core import Structure
        
        # Load model silently
        model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
        
        for fp in cif_paths:
            try:
                s = Structure.from_file(fp)
                val = model.predict_structure(s).item()
                results.append({"file_path": fp, "formation_energy": float(val), "error": None})
            except Exception as e:
                results.append({"file_path": fp, "formation_energy": None, "error": str(e)})
    except Exception as e:
        results = [{"error": f"Worker crashed: {e}"}]
    print(json.dumps(results))

if __name__ == "__main__": main()