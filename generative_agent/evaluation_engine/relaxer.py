# --- Core Libraries ---
import numpy as np
import os

# --- Materials Science Libraries ---
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atoms

# --- MATGL ---
# Ensure backend is set to DGL/PyTorch
os.environ["DGLBACKEND"] = "pytorch"
import matgl
from matgl.ext.ase import Relaxer as MatglRelaxer

class Relaxer:
    """
    Wrapper around MATGL's Relaxer that matches NOVAGEN's interface.
    """

    def __init__(self):
        # 1. Force backend
        matgl.set_backend("DGL")

        # 2. Load the potential
        # This will download the model to ~/.cache/matgl/ on the first run
        print("Loading M3GNet potential...")
        self._potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        # 3. Initialize the MATGL Relaxer
        self._relaxer = MatglRelaxer(
            potential=self._potential,
            optimizer="LBFGS",
            relax_cell=True
        )

    def relax(self, structure: Structure, fmax=0.01, steps=100) -> dict:
        """
        Relax a pymatgen Structure.
        """
        # Convert Pymatgen -> ASE
        atoms = AseAtomsAdaptor.get_atoms(structure)
        num_atoms = len(atoms)

        # Run Relaxation
        try:
            # MatglRelaxer.relax returns a dictionary in newer versions
            result = self._relaxer.relax(atoms, fmax=fmax, steps=steps)
        except Exception as e:
            # If the optimization crashes (e.g. atoms exploded)
            return {
                "final_structure": structure,
                "final_energy_per_atom": None,
                "is_converged": False,
                "error": str(e)
            }

        # --- Parse Results ---
        # The result object can vary slightly depending on matgl version
        relaxed_structure = None
        final_energy = None
        converged = True # Assume true unless error caught above

        # Case A: Result is a Dictionary
        if isinstance(result, dict):
            # Extract Structure
            if "final_structure" in result:
                relaxed_structure = result["final_structure"]
            elif "atoms" in result:
                relaxed_structure = AseAtomsAdaptor.get_structure(result["atoms"])
            
            # Extract Energy
            # Try to find trajectory data
            if "trajectory" in result:
                traj = result["trajectory"]
                # Try standard list
                if hasattr(traj, "energies") and len(traj.energies) > 0:
                    final_energy = traj.energies[-1] / num_atoms
                # Try pandas conversion
                elif hasattr(traj, "as_pandas"):
                    try:
                        df = traj.as_pandas()
                        if not df.empty:
                            final_energy = df.iloc[-1]["energy"] / num_atoms
                    except:
                        pass
        
        # Case B: Result is ASE Atoms object
        elif isinstance(result, Atoms):
            relaxed_structure = AseAtomsAdaptor.get_structure(result)
            try:
                final_energy = result.get_potential_energy() / num_atoms
            except:
                pass

        # Fallback if parsing failed
        if relaxed_structure is None:
            relaxed_structure = structure
            converged = False

        return {
            "final_structure": relaxed_structure,
            "final_energy_per_atom": final_energy,
            "is_converged": converged,
            "num_steps": steps
        }
