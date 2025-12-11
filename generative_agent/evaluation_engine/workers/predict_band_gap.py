# generative_agent/evaluation_engine/workers/predict_band_gap.py
import sys, json, os

def main():
    results = []
    args = sys.argv[1:]
    
    # CRITICAL FIX: Add current directory to path for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    try:
        # We rely on a robust MatGL model instead of legacy Megnet/TF if possible, 
        # but the original used Megnet. Let's use the most stable modern Megnet.
        import warnings, logging
        warnings.simplefilter("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.getLogger("megnet").setLevel(logging.CRITICAL)
        
        import tensorflow as tf
        from megnet.utils.models import load_model
        from pymatgen.core import Structure
        
        # Load the latest stable model
        model = load_model("Bandgap_MP_2020") 
        
        for fp in args:
            info = {"file_path": fp, "band_gap": None, "error": None}
            try:
                s = Structure.from_file(fp)
                val = model.predict_structure(s).ravel()[0]
                info["band_gap"] = float(val)
                results.append(info)
            except Exception as e:
                info["error"] = f"Prediction failed with error: {str(e)}"
                results.append(info)

    except Exception as e:
        results = [{"error": f"Worker crashed (Import/Setup): {str(e)}", "file_path": "N/A"}]

    print(json.dumps(results if results else []))

if __name__ == "__main__": main()
