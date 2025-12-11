# generative_agent/evaluation_engine/relaxer.py

# --- Core Libraries ---
import torch
import numpy as np
import sys

# --- Materials Science Libraries ---
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

# --- Atomic Simulation Environment (ASE) for Optimization ---
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter

# --- Materials Graph Library (MATGL) for the ML Potential ---
import matgl
# CRITICAL FIX: Use the universal "MatglCalculator" which replaced M3GNetCalculator
# and explicitly import the potential.
try:
    from matgl.ext.ase import MatglCalculator as M3GNetCalculator
except ImportError:
    # Fallback if an older matgl is somehow used
    sys.stderr.write("MatglCalculator not found. This will likely crash.\n")
    from matgl.ext.ase import M3GNetCalculator
# ---------------------------------------------

class Relaxer:
    """
    Final, robust relaxer using the universal MatglCalculator class.
    """

    def __init__(self):
        # We assume the model name is correct, as this is MatGL's standard.
        self._potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        # Use the general MatglCalculator class
        self._calculator = M3GNetCalculator(potential=self._potential)

    def relax(self, structure: Structure, fmax=0.1, steps=250, log_file='relaxation.log') -> dict:
        """
        Relaxes a pymatgen Structure with relaxed convergence criteria.
        """
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            sys.stderr.write(f"Relaxer Error: Input structure conversion failed. {e}\n")
            return {'error': 'Input structure conversion failed.'}

        atoms.calc = self._calculator

        ucf = UnitCellFilter(atoms)
        
        # Use a temporary log file to avoid conflicts
        temp_log_file = os.path.join(os.path.dirname(log_file), f"temp_relax_{os.getpid()}.log")
        optimizer = LBFGS(ucf, logfile=temp_log_file)

        optimizer.run(fmax=fmax, steps=steps)

        relaxed_pymatgen_structure = AseAtomsAdaptor.get_structure(atoms)

        final_energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = np.sqrt((forces**2).sum(axis=1).max()) if forces is not None else 0.0
        
        is_converged = max_force <= fmax

        if os.path.exists(temp_log_file):
            os.remove(temp_log_file)

        return {
            'final_structure': relaxed_pymatgen_structure,
            'final_energy': final_energy,
            'is_converged': is_converged,
            'max_force': max_force,
            'num_steps': optimizer.get_number_of_steps(),
        }

# --- Self-Contained Test Block ---
if __name__ == '__main__':
    # This block is not executed by the worker, but left for completeness
    pass
