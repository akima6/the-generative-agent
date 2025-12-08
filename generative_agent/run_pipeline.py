# generative_agent/run_pipeline.py

import os
import subprocess
import tempfile
import json
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from agent import GenerativeAgent
from evaluation_engine.oracle import Oracle

# Config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_ENV = "generative_agent"
MATGL_ENV = "matgl_env"
MEGNET_ENV = "megnet_legacy_env"
RELAXER_SCRIPT = os.path.join(PROJECT_ROOT, "generative_agent", "evaluation_engine", "relax_worker.py")

# CSV output from the generator
SPG = 225
GENERATED_CSV = os.path.join(PROJECT_ROOT, "pretrained_model", f"output_{SPG}.csv")

def main():
    print("--- Starting Pipeline ---")
    
    # 1. Generate (or reuse existing)
    # Note: We skip re-generating if the file exists to speed up debugging.
    # To force new generation, delete the output CSV or uncomment the propose line.
    agent = GenerativeAgent(PROJECT_ROOT, MAIN_ENV)
    
    # Check if we have structures to work with
    if os.path.exists(GENERATED_CSV):
        print(f"Found existing generated file: {GENERATED_CSV}")
        structures = agent._parse_output(GENERATED_CSV)
    else:
        print("Generating new structures...")
        structures = agent.propose(batch_size=5, space_group=SPG)
    
    if not structures:
        print("No structures available.")
        return

    # 2. Relax (Subprocess)
    print(f"\nRelaxing {len(structures)} structures...")
    valid_structures = []
    
    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        cifs = []
        for i, s in enumerate(structures):
            p = os.path.join(tmp_in, f"{i}.cif")
            CifWriter(s).write_file(p)
            cifs.append(p)
            
        cmd = ["conda", "run", "-n", MATGL_ENV, "python", RELAXER_SCRIPT, tmp_out] + cifs
        
        try:
            # We must capture output to parse JSON
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # --- DEBUGGING BLOCK ---
            if not res.stdout.strip():
                print("ERROR: Relaxer worker returned empty output.")
                print(f"STDERR dump:\n{res.stderr}")
                return
            
            try:
                relax_data = json.loads(res.stdout)
            except json.JSONDecodeError as e:
                print(f"CRITICAL ERROR: Failed to parse JSON from relaxer.")
                print(f"Raw STDOUT: {res.stdout}")
                print(f"Raw STDERR: {res.stderr}")
                return
            # -----------------------
            
            for i, data in enumerate(relax_data):
                # Check for errors in individual file processing
                if "error" in data:
                    print(f"  - Structure {i} failed: {data['error']}")
                    continue
                    
                if data.get("is_converged"):
                    s = Structure.from_file(data["output_file"])
                    valid_structures.append({
                        "structure": s,
                        "formula": structures[i].formula
                    })
                else:
                    print(f"  - Structure {i} did not converge.")

        except subprocess.CalledProcessError as e:
            print(f"Relaxer command failed with exit code {e.returncode}")
            print(f"STDERR: {e.stderr}")
            return

    print(f"\nStable structures found: {len(valid_structures)}")

    # 3. Predict (Bridge)
    if valid_structures:
        print("\nPredicting properties...")
        oracle = Oracle(MATGL_ENV, MEGNET_ENV)
        structs_only = [x["structure"] for x in valid_structures]
        preds = oracle.predict_properties(structs_only)
        
        results = []
        for i, item in enumerate(valid_structures):
            p = preds[i]
            results.append({
                "Formula": item["formula"],
                "Formation Energy": p["formation_energy"],
                "Band Gap": p["band_gap"],
                "Error": p.get("error")
            })
            
        df = pd.DataFrame(results)
        df.to_csv("final_results.csv", index=False)
        print("\n--- RESULTS ---")
        print(df[["Formula", "Formation Energy", "Band Gap"]])
        print("\nFull results saved to final_results.csv")

if __name__ == "__main__":
    main()