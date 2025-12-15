# generative_agent/evaluation_engine/workers/predict_formation_energy.py
import sys
import os
import json
import warnings

# Suppress warnings for clean JSON output
warnings.filterwarnings("ignore")

def main():
    # Input: List of CIF file paths
    input_cifs = sys.argv[1:]
    results = []

    if not input_cifs:
        print(json.dumps([]))
        return

    try:
        import matgl
        from pymatgen.core import Structure
        import torch

        # Force CPU if needed, or use CUDA if available
        # MatGL handles this, but we suppress DGL warnings
        
        # --- LOAD MODEL (Exactly like your old script) ---
        # We use the Eform model specifically for accurate energy prediction
        model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
        
        for cif_path in input_cifs:
            entry = {"file_path": cif_path}
            try:
                if not os.path.exists(cif_path):
                    entry["error"] = "File not found"
                    results.append(entry)
                    continue

                structure = Structure.from_file(cif_path)
                
                # Predict
                # The model returns a tensor, we need a float
                prediction = model.predict_structure(structure)
                val = prediction.item() if hasattr(prediction, "item") else float(prediction)
                
                entry["formation_energy"] = val
                
            except Exception as e:
                entry["error"] = str(e)
            
            results.append(entry)

    except Exception as e:
        # Catch loading errors (e.g. library missing)
        results = [{"error": f"CRITICAL WORKER FAILURE: {str(e)}"}]

    # Return JSON to the Oracle
    print(json.dumps(results))

if __name__ == "__main__":
    main()
