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

def relax(self, structure: Structure, fmax=0.01, steps=100) -> dict:
    from pymatgen.io.ase import AseAtomsAdaptor

    try:
        atoms = AseAtomsAdaptor.get_atoms(structure)
    except Exception as e:
        return {"error": f"ASE conversion failed: {e}", "is_converged": False}

    num_atoms = len(atoms)

    # --- Relax ---
    try:
        result = self._relaxer.relax(atoms, fmax=fmax, steps=steps)
    except Exception as e:
        return {"error": f"Relaxation failed: {e}", "is_converged": False}

    # ✅ Correct keys for MATGL >= 1.0
    relaxed_atoms = result["atoms"]
    converged = result.get("converged", False)

    final_structure = AseAtomsAdaptor.get_structure(relaxed_atoms)

    # Energy per atom (last step)
    if "trajectory" in result and len(result["trajectory"].energies) > 0:
        final_energy = result["trajectory"].energies[-1]
        final_energy_per_atom = final_energy / num_atoms
        num_steps = len(result["trajectory"].energies)
    else:
        final_energy_per_atom = None
        num_steps = None

    return {
        "final_structure": final_structure,
        "final_energy_per_atom": final_energy_per_atom,
        "is_converged": converged,
        "num_steps": num_steps,
    }
