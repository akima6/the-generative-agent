# generative_agent/evaluation_engine/workers/predict_band_gap.py
import sys
import os
import json
import warnings

# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

def main():
    input_cifs = sys.argv[1:]
    results = []

    if not input_cifs:
        print(json.dumps([]))
        return

    try:
        # Import inside main to catch environment errors cleanly
        from megnet.utils.models import load_model
        from pymatgen.core import Structure
        import numpy as np

        # --- LOAD MODEL (Exactly like your old script) ---
        # The 'Bandgap_MP_2018' model is the standard MEGNET pre-trained model
        model = load_model("Bandgap_MP_2018")

        for cif_path in input_cifs:
            entry = {"file_path": cif_path}
            try:
                if not os.path.exists(cif_path):
                    entry["error"] = "File not found"
                    results.append(entry)
                    continue

                structure = Structure.from_file(cif_path)
                
                # Predict
                # MEGNET returns a numpy scalar/array
                prediction = model.predict_structure(structure)
                
                # Handle numpy types for JSON serialization
                if isinstance(prediction, (np.ndarray, list)):
                    val = float(prediction.ravel()[0])
                else:
                    val = float(prediction)
                
                entry["band_gap"] = val
                
            except Exception as e:
                entry["error"] = str(e)
            
            results.append(entry)

    except Exception as e:
        results = [{"error": f"CRITICAL MEGNET FAILURE: {str(e)}"}]

    # Return JSON to the Oracle
    print(json.dumps(results))

if __name__ == "__main__":
    main()
