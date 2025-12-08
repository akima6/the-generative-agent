# The Generative Agent/generative_agent/test_agent.py
# A script to independently test our new GenerativeAgent class.

import os
from agent import GenerativeAgent
from pymatgen.io.cif import CifWriter

# --- Configuration ---
# The root directory of our entire project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The name of the Conda environment where CrystalFormer is installed
MAIN_ENV_NAME = "generative_agent"

def main():
    print("--- Testing the GenerativeAgent Class ---")

    # 1. Initialize the Agent
    # We pass it the root directory and the environment name.
    agent = GenerativeAgent(project_root_dir=PROJECT_ROOT, main_env_name=MAIN_ENV_NAME)

    # 2. Use the Agent to Propose New Structures
    # This will run the `main.py` script as a subprocess.
    space_group_to_test = 12  # Let's try a different space group
    num_to_generate = 3
    
    unrelaxed_structures = agent.propose(
        batch_size=num_to_generate,
        space_group=space_group_to_test
    )

    # 3. Report the Results
    print("\n--- Agent Test Results ---")
    if not unrelaxed_structures:
        print("FAILURE: Agent did not return any structures.")
        return

    print(f"SUCCESS: Agent proposed {len(unrelaxed_structures)} structures.")
    
    # Create an output directory to save the results for visual confirmation
    output_dir = "agent_test_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving proposed structures to the '{output_dir}' directory...")

    for i, struct in enumerate(unrelaxed_structures):
        print(f"  - Structure #{i+1}: {struct.formula.replace(' ', '')}")
        # Save each structure as a .cif file
        cif_path = os.path.join(output_dir, f"proposed_sg12_{i+1}.cif")
        CifWriter(struct).write_file(cif_path)
        
    print("\nTest complete. Check the CIF files in the output directory.")

if __name__ == "__main__":
    main()