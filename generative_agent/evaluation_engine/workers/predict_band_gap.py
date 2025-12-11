# generative_agent/evaluation_engine/workers/predict_band_gap.py
# Runs in current python environment (not conda)

import sys, json, os

def main():
    results = []
    args = sys.argv[1:]
    
    # FIX: Add parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    try:
        # Aggressive silence for TensorFlow/Megnet
        import warnings, logging
        warnings.simplefilter("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.getLogger("megnet").setLevel(logging.CRITICAL)
        
        import tensorflow as tf
        from megnet.utils.models import load_model
        from pymatgen.core import Structure
        
        # FIX: Ensure Megnet model loads
        model = load_model("Bandgap_MP_2018")
        
        for fp in args:
            info = {"file_path": fp, "band_gap": None, "error": None}
            try:
                s = Structure.from_file(fp)
                # Suppress TF's eager execution warning
                val = model.predict_structure(s).ravel()[0]
                info["band_gap"] = float(val)
                results.append(info)
            except Exception as e:
                # Catch per-structure errors
                info["error"] = f"Prediction failed with error: {str(e)}"
                results.append(info)

    except Exception as e:
        # Catastrophic failure catch for the whole script (ImportError)
        results = [{"error": f"Worker crashed (Import/Setup): {str(e)}", "file_path": "N/A"}]

    print(json.dumps(results if results else []))

if __name__ == "__main__": main()
