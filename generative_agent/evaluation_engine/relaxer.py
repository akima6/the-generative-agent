# --- Core Libraries ---
import numpy as np

# --- Materials Science Libraries ---
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# --- ASE ---
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter

# --- MATGL (NEW API) ---
import matgl
matgl.set_backend("DGL")

from matgl.ext.ase import Relaxer as MatglRelaxer


class Relaxer:
    """
    Wrapper around MATGL's Relaxer that matches NOVAGEN's interface.
    """

    def __init__(self):
        import matgl
        matgl.set_backend("DGL")

        self._potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        self._relaxer = MatglRelaxer(
            potential=self._potential,
            optimizer="LBFGS",
            relax_cell=True
        )

    def relax(self, structure: Structure, fmax=0.01, steps=100) -> dict:
        """
        Relax a pymatgen Structure and return standardized results.
        """
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            return {"error": f"ASE conversion failed: {e}"}

        num_atoms = len(atoms)

        # --- Initial energy ---
        try:
            initial_energy = self._relaxer.calculator.get_potential_energy(atoms)
            initial_energy_per_atom = initial_energy / num_atoms
        except Exception:
            initial_energy_per_atom = None

        # --- Relax ---
        try:
            result = self._relaxer.relax(atoms, fmax=fmax, steps=steps)
        except Exception as e:
            return {"error": f"Relaxation failed: {e}"}

        final_atoms = result["final_atoms"]
        final_structure = AseAtomsAdaptor.get_structure(final_atoms)

        final_energy = result["trajectory"].energies[-1]
        final_energy_per_atom = final_energy / num_atoms

        is_converged = result.get("converged", False)

        return {
            "final_structure": final_structure,
            "initial_energy_per_atom": initial_energy_per_atom,
            "final_energy_per_atom": final_energy_per_atom,
            "is_converged": is_converged,
            "num_steps": len(result["trajectory"].energies),
        }
