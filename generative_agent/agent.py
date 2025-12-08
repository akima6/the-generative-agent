# The Generative Agent/generative_agent/agent.py
# A clean wrapper for the CrystalFormer generation script.

import os
import subprocess
import pandas as pd
import ast
from pymatgen.core import Structure, Lattice, Element

class GenerativeAgent:
    """
    A wrapper class for the CrystalFormer generative model. It uses a subprocess
    to call the official main.py script for generating new crystal structures.
    """
    def __init__(self, project_root_dir: str, main_env_name: str):
        """
        Initializes the agent with necessary paths and environment names.
        
        Args:
            project_root_dir (str): The absolute path to 'The Generative Agent'.
            main_env_name (str): The name of the conda env where CrystalFormer is installed.
        """
        self.root_dir = project_root_dir
        self.env_name = main_env_name
        self.crystalformer_dir = os.path.join(self.root_dir, "CrystalFormer")
        self.generator_script = os.path.join(self.crystalformer_dir, "main.py")
        self.model_path = os.path.join(self.root_dir, "pretrained_model", "epoch_005500.pkl")
        self.output_dir = os.path.join(self.root_dir, "pretrained_model")

    def propose(self, batch_size: int, space_group: int) -> list:
        """
        Generates a batch of new, unrelaxed crystal structures.
        
        Args:
            batch_size (int): The number of structures to generate.
            space_group (int): The space group to use as a condition.
            
        Returns:
            A list of unrelaxed pymatgen.Structure objects.
        """
        print(f"\n--- Agent: Proposing {batch_size} structures for space group {space_group} ---")
        
        output_csv_path = os.path.join(self.output_dir, f"output_{space_group}.csv")
        
        command = [
            "conda", "run", "-n", self.env_name, "python",
            self.generator_script,
            "--optimizer", "none",
            "--restore_path", self.model_path,
            "--spacegroup", str(space_group),
            "--num_samples", str(batch_size),
            "--batchsize", str(batch_size),
            "--temperature", "1.0"
        ]
        
        try:
            # We add the --quiet flag back in, as it's good practice. If it fails for you,
            # you can remove it. The worker scripts are now robust enough to handle it.
            command.insert(2, "--quiet")
            subprocess.run(command, check=True, cwd=self.crystalformer_dir, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # If the --quiet flag fails, try again without it.
            if "unrecognized arguments: --quiet" in e.stderr:
                print("Warning: Your version of conda does not support '--quiet'. Retrying without it.")
                command.pop(2)
                subprocess.run(command, check=True, cwd=self.crystalformer_dir, capture_output=True, text=True)
            else:
                print(f"ERROR: Generation failed. Stderr:\n{e.stderr}")
                return []
        
        print(f"Agent: Generation complete. Parsing results from {output_csv_path}")
        return self._parse_output(output_csv_path)

    def _parse_output(self, csv_path: str) -> list:
        """Helper function to parse the generator's CSV output."""
        if not os.path.exists(csv_path):
            return []
        
        df = pd.read_csv(csv_path)
        structures = []
        for _, row in df.iterrows():
            try:
                l_params = ast.literal_eval(row['L'])
                lattice = Lattice.from_parameters(*l_params)
                
                atom_numbers = ast.literal_eval(row['A'])
                coords = ast.literal_eval(row['X'])
                
                valid_species = [Element.from_Z(z) for z, c in zip(atom_numbers, coords) if z > 0]
                valid_coords = [c for z, c in zip(atom_numbers, coords) if z > 0]
                
                if valid_species:
                    structures.append(Structure(lattice, valid_species, valid_coords))
            except Exception:
                continue # Skip rows that are malformed
        return structures