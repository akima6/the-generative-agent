# --- Core Libraries ---
import numpy as np

# --- Materials Science Libraries ---
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# --- ASE ---
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter

# --- MATGL (NEW API) ---
import os
os.environ["DGLBACKEND"] = "pytorch"

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

def relax(self, structure, fmax=0.01, steps=100):
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.atoms import Atoms

    atoms = AseAtomsAdaptor.get_atoms(structure)
    num_atoms = len(atoms)

    try:
        result = self._relaxer.relax(atoms, fmax=fmax, steps=steps)
    except Exception as e:
        return {
            "is_converged": False,
            "error": str(e)
        }

    # ======================================================
    # CASE 1: MATGL returns ase.Atoms directly
    # ======================================================
    if isinstance(result, Atoms):
        relaxed_atoms = result
        converged = True
        final_energy_per_atom = None
        num_steps = None

    # ======================================================
    # CASE 2: MATGL returns a dictionary
    # ======================================================
    elif isinstance(result, dict):
        converged = result.get("converged", False)

        if "atoms" in result:
            relaxed_atoms = result["atoms"]
        elif "final_atoms" in result:
            relaxed_atoms = result["final_atoms"]
        else:
            return {
                "is_converged": False,
                "error": f"Unknown MATGL result keys: {result.keys()}"
            }

        # Energy
        final_energy_per_atom = None
        num_steps = None
        if "trajectory" in result and hasattr(result["trajectory"], "energies"):
            if len(result["trajectory"].energies) > 0:
                final_energy_per_atom = result["trajectory"].energies[-1] / num_atoms
                num_steps = len(result["trajectory"].energies)

    # ======================================================
    # CASE 3: Unknown return type
    # ======================================================
    else:
        return {
            "is_converged": False,
            "error": f"Unknown MATGL return type: {type(result)}"
        }

    final_structure = AseAtomsAdaptor.get_structure(relaxed_atoms)

    return {
        "final_structure": final_structure,
        "final_energy_per_atom": final_energy_per_atom,
        "is_converged": converged,
        "num_steps": num_steps,
    }
