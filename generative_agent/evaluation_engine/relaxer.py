# generative_agent/evaluation_engine/relaxer.py
# Runs in current python environment (not conda)

# --- Core Libraries ---
import torch
import numpy as np

# --- Materials Science Libraries ---
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

# --- Atomic Simulation Environment (ASE) for Optimization ---
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter

# --- Materials Graph Library (MATGL) for the ML Potential ---
import matgl
from matgl.ext.ase import M3GNetCalculator

class Relaxer:
    """
    A class for performing fast, GNN-based structural relaxation of crystal
    structures using an M3GNet potential, designed to integrate into a
    high-throughput materials discovery workflow like BOWSR.
    """

    def __init__(self):
        self._potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        self._calculator = M3GNetCalculator(potential=self._potential)

    def relax(self, structure: Structure, fmax=0.1, steps=250, log_file='relaxation.log') -> dict:
        """
        Relaxes a pymatgen Structure with relaxed convergence criteria.

        Args:
            structure (pymatgen.core.structure.Structure): Input crystal structure.
            fmax (float): The maximum force tolerance (eV/Å). RELAXED to 0.1.
            steps (int): The maximum number of optimization steps. INCREASED to 250.
            log_file (str): Path to the log file for the optimization progress.

        Returns:
            dict: Results including the final relaxed structure and metadata.
        """
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            # We must print to stderr for the main script to capture it
            sys.stderr.write(f"Relaxer Error: Input structure conversion failed. {e}\n")
            return {'error': 'Input structure conversion failed.'}

        atoms.calc = self._calculator

        # UnitCellFilter for full cell relaxation.
        ucf = UnitCellFilter(atoms)

        # Use temporary log file to avoid conflicts
        temp_log_file = os.path.join(os.path.dirname(log_file), f"temp_relax_{os.getpid()}.log")
        optimizer = LBFGS(ucf, logfile=temp_log_file)

        # Run the optimization.
        optimizer.run(fmax=fmax, steps=steps)

        # Convert back to a pymatgen Structure
        relaxed_pymatgen_structure = AseAtomsAdaptor.get_structure(atoms)

        # Compile the output dictionary
        final_energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = np.sqrt((forces**2).sum(axis=1).max()) if forces is not None else 0.0
        
        # Check if the final max force is below our target tolerance.
        is_converged = max_force <= fmax

        # Clean up temp log
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
