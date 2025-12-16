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

    atoms = AseAtomsAdaptor.get_atoms(structure)
    num_atoms = len(atoms)

    try:
        result = self._relaxer.relax(atoms, fmax=fmax, steps=steps)
    except Exception as e:
        return {
            "is_converged": False,
            "error": str(e)
        }

    # 🔍 DEBUG (print keys once)
    if not hasattr(self, "_printed_keys"):
        print("MATGL relax() returned keys:", result.keys())
        self._printed_keys = True

    # --- Find relaxed atoms safely ---
    if "atoms" in result:
        relaxed_atoms = result["atoms"]
    elif "final_atoms" in result:
        relaxed_atoms = result["final_atoms"]
    else:
        return {
            "is_converged": False,
            "error": f"No atoms found in relaxer result keys: {result.keys()}"
        }

    final_structure = AseAtomsAdaptor.get_structure(relaxed_atoms)

    # --- Energy ---
    energy = None
    steps_done = None
    if "trajectory" in result and hasattr(result["trajectory"], "energies"):
        if len(result["trajectory"].energies) > 0:
            energy = result["trajectory"].energies[-1] / num_atoms
            steps_done = len(result["trajectory"].energies)

    converged = result.get("converged", False)

    return {
        "final_structure": final_structure,
        "final_energy_per_atom": energy,
        "is_converged": converged,
        "num_steps": steps_done,
    }
