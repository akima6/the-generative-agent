# generative_agent/evaluation_engine/workers/predict_band_gap.py
# Runs in 'megnet_legacy_env'
import sys, json, os

def main():
    results = []
    cif_paths = sys.argv[1:]
    try:
        # Aggressive silence
        import warnings, logging
        warnings.simplefilter("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.getLogger("megnet").setLevel(logging.CRITICAL)
        
        import tensorflow as tf
        from megnet.utils.models import load_model
        from pymatgen.core import Structure
        
        model = load_model("Bandgap_MP_2018")
        
        for fp in cif_paths:
            try:
                s = Structure.from_file(fp)
                val = model.predict_structure(s).ravel()[0]
                results.append({"file_path": fp, "band_gap": float(val), "error": None})
            except Exception as e:
                results.append({"file_path": fp, "band_gap": None, "error": str(e)})
    except Exception as e:
        results = [{"error": f"Worker crashed: {e}"}]
    print(json.dumps(results))

if __name__ == "__main__": main()