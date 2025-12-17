# --- Core Libraries ---
import numpy as np
import os

# --- Materials Science Libraries ---
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atoms

# --- MATGL ---
# Ensure backend is set before importing matgl models
os.environ["DGLBACKEND"] = "pytorch"
import matgl
from matgl.ext.ase import Relaxer as MatglRelaxer

class Relaxer:
    """
    Wrapper around MATGL's Relaxer that matches NOVAGEN's interface.
    """

    def __init__(self):
        # Set backend (redundant but safe)
        matgl.set_backend("DGL")

        # Load the potential
        self._potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        # Initialize the MATGL Relaxer
        self._relaxer = MatglRelaxer(
            potential=self._potential,
            optimizer="LBFGS",
            relax_cell=True
        )

    # [FIX] This method is now correctly indented inside the class
    def relax(self, structure: Structure, fmax=0.01, steps=100) -> dict:
        
        # Convert Pymatgen Structure -> ASE Atoms
        atoms = AseAtomsAdaptor.get_atoms(structure)
        num_atoms = len(atoms)

        try:
            # Run the relaxation
            result = self._relaxer.relax(atoms, fmax=fmax, steps=steps)
        except Exception as e:
            return {
                "final_structure": structure, # Return original on failure
                "final_energy_per_atom": None,
                "is_converged": False,
                "error": str(e)
            }

        # --- Parse Result ---
        # MatglRelaxer usually returns a dictionary: {'final_structure': ..., 'trajectory': ...}
        
        relaxed_structure = None
        final_energy_per_atom = None
        converged = False

        # Handle Dictionary Return (Standard for recent MATGL versions)
        if isinstance(result, dict):
            # 1. Extract Structure
            if "final_structure" in result:
                relaxed_structure = result["final_structure"]
            elif "atoms" in result:
                # Fallback if it returns ASE atoms
                relaxed_structure = AseAtomsAdaptor.get_structure(result["atoms"])
            
            # 2. Extract Energy (from trajectory if available)
            if "trajectory" in result:
                # Try to get energy from the trajectory observer
                traj = result["trajectory"]
                # Different versions store energies differently, try common attributes
                if hasattr(traj, "energies") and len(traj.energies) > 0:
                    final_energy_per_atom = traj.energies[-1] / num_atoms
                elif hasattr(traj, "as_pandas"):
                    # Newer matgl versions often have a dataframe
                    df = traj.as_pandas()
                    if not df.empty:
                         final_energy_per_atom = df.iloc[-1]["energy"] / num_atoms

            # Assume converged if no error was raised by the optimizer
            converged = True

        # Handle ASE Atoms Return (Legacy/Fallback)
        elif isinstance(result, Atoms):
            relaxed_structure = AseAtomsAdaptor.get_structure(result)
            try:
                final_energy_per_atom = result.get_potential_energy() / num_atoms
            except:
                final_energy_per_atom = None
            converged = True
        
        else:
            return {
                "is_converged": False,
                "error": f"Unknown return type from MATGL: {type(result)}"
            }

        # Final safety check
        if relaxed_structure is None:
             relaxed_structure = structure # Fallback to input

        return {
            "final_structure": relaxed_structure,
            "final_energy_per_atom": final_energy_per_atom,
            "is_converged": converged,
            "num_steps": steps # MATGL wrapper doesn't always strictly return step count easily
        }
