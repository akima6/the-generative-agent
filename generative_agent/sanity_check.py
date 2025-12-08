# generative_agent/sanity_check.py
import os
import sys
import json
import tempfile
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
import subprocess

# --- CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent
MATGL_ENV = "matgl_env"
MEGNET_ENV = "megnet_legacy_env"

# Path to the specific worker scripts (Environment B & C)
RELAXER_SCRIPT = CURRENT_DIR / "evaluation_engine" / "relax_worker.py"
FE_SCRIPT = CURRENT_DIR / "evaluation_engine" / "workers" / "predict_formation_energy.py"
BG_SCRIPT = CURRENT_DIR / "evaluation_engine" / "workers" / "predict_band_gap.py"

# --- KNOWN GOOD STRUCTURES ---
# 1. Magnesium Oxide (Rock Salt, SG=225) - Should relax and be stable
MgO_structure = Structure(
    Lattice.from_parameters(a=4.21, b=4.21, c=4.21, alpha=90, beta=90, gamma=90),
    ["Mg", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5]]
)

# 2. Silicon (Diamond, SG=227) - Should relax and have a known Band Gap
Si_structure = Structure(
    Lattice.from_parameters(a=5.43, b=5.43, c=5.43, alpha=90, beta=90, gamma=90),
    ["Si", "Si"],
    [[0, 0, 0], [0.25, 0.25, 0.25]]
)

TEST_STRUCTURES = [MgO_structure, Si_structure]
print(f"--- NOVAGEN WORKER SANITY CHECK ---")
print(f"Testing {len(TEST_STRUCTURES)} known-good structures.")

def run_worker_test(env: str, script_path: Path, input_files: list, temp_out_dir: str = None) -> list:
    """Helper to run a worker script and return JSON data, or print error."""
    
    # If a relaxation-type script, it needs an output directory
    cmd_args = [temp_out_dir] + input_files if temp_out_dir else input_files
    cmd = ["conda", "run", "-n", env, "python", str(script_path)] + cmd_args
    
    print(f"\n[EXEC] Running in '{env}': {script_path.name}")
    print(f"     CMD: {' '.join(cmd)}")
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Success, attempt to decode the JSON result
        try:
            data = json.loads(proc.stdout)
            return data
        except json.JSONDecodeError:
            print("  [FAIL] JSON DECODE ERROR.")
            print(f"    STDOUT (corrupted output): \n{proc.stdout[:500]}...")
            print(f"    STDERR (worker logs): \n{proc.stderr}")
            return []

    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] WORKER CRASHED (Return Code {e.returncode}).")
        print(f"    STDOUT (last output): \n{e.stdout}")
        print(f"    STDERR (CRITICAL ERROR): \n{e.stderr}")
        return []
    except Exception as e:
        print(f"  [FAIL] UNEXPECTED ERROR: {e}")
        return []

# --- MAIN TEST LOGIC ---
temp_cifs = []
relaxed_cifs = []

with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
    
    # 1. Write CIFs to Temp Input Folder
    for i, s in enumerate(TEST_STRUCTURES):
        p = os.path.join(tmp_in, f"{i}.cif")
        CifWriter(s).write_file(p)
        temp_cifs.append(p)

    # 2. TEST RELAXER (Environment B)
    relaxation_results = run_worker_test(
        MATGL_ENV, RELAXER_SCRIPT, temp_cifs, tmp_out
    )
    
    if not relaxation_results:
        print("\n!!! RELAXER FAILED ENTIRELY. CHECK CONDA ENVIRONMENT B (matgl_env) !!!")
        sys.exit(1)

    print("\n[SUCCESS] Relaxation worker returned results.")
    
    # Check if any structure converged and collect paths
    for i, res in enumerate(relaxation_results):
        if res.get("is_converged"):
            print(f"  [INFO] Structure {i+1} converged. Max Force: {res.get('max_force', 'N/A')}")
            relaxed_cifs.append(res["output_file"])
        else:
            print(f"  [WARN] Structure {i+1} failed to converge. Error: {res.get('error', 'N/A')}")
            
    if not relaxed_cifs:
        print("\n!!! All known-good structures FAILED to converge. MatGL model may be corrupt or misconfigured. !!!")
        sys.exit(1)

    # 3. TEST ORACLE PROPERTIES (Environment C for Band Gap & Env B for Formation Energy)
    
    # --- Formation Energy (Env B - matgl_env) ---
    fe_results = run_worker_test(MATGL_ENV, FE_SCRIPT, relaxed_cifs)
    
    # --- Band Gap (Env C - megnet_legacy_env) ---
    bg_results = run_worker_test(MEGNET_ENV, BG_SCRIPT, relaxed_cifs)
    
    # 4. FINAL REPORT
    print("\n--- FINAL WORKER REPORT ---")
    for i, (fe, bg) in enumerate(zip(fe_results, bg_results)):
        print(f"\n[Crystal {i+1}]")
        print(f"  Formation Energy: {fe.get('formation_energy', 'FAIL')} eV/atom")
        print(f"  Band Gap:         {bg.get('band_gap', 'FAIL')} eV")
        print(f"  MatGL Error:      {fe.get('error', 'None')}")
        print(f"  MEGNET Error:     {bg.get('error', 'None')}")
        
    print("\nSANITY CHECK COMPLETE. Proceed to fix the issue identified above.")