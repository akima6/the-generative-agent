# generative_agent/evaluation_engine/relaxer.py
# Runs in current python environment (not conda)

# --- Core Libraries ---
import torch
import numpy as np
import sys
import os

# --- Materials Science Libraries ---
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

# --- Atomic Simulation Environment (ASE) for Optimization ---
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter

# --- Materials Graph Library (MATGL) for the ML Potential ---
import matgl
from matgl.ext.ase import MatglCalculator

class Relaxer:
    def __init__(self):
        self._potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        self._calculator = MatglCalculator(potential=self._potential)

    def relax(self, structure: Structure, fmax=0.1, steps=500, log_file='relaxation.log') -> dict:
        """
        Relaxes a pymatgen Structure with relaxed convergence criteria (fmax=0.1, steps=500).
        """
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            # We must print to stderr for the main script to capture it
            sys.stderr.write(f"Relaxer Error: Input structure conversion failed. {e}\n")
            return {'error': 'Input structure conversion failed.'}

        atoms.calc = self._calculator
        ucf = UnitCellFilter(atoms)

        # Using a temporary log file to avoid conflicts
        temp_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"temp_relax_{os.getpid()}.log")
        
        # Ensure log file exists for LBFGS (if used)
        open(temp_log_file, 'a').close()
        
        optimizer = LBFGS(ucf, logfile=temp_log_file)

        # Run the optimization.
        optimizer.run(fmax=fmax, steps=steps)

        # Convert back to a pymatgen Structure
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
