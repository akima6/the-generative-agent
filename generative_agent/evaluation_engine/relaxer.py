# generative_agent/evaluation_engine/relaxer.py

# --- Core Libraries ---
import torch
import numpy as np

# --- Materials Science Libraries ---
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

# --- Atomic Simulation Environment (ASE) ---
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter

# --- Materials Graph Library (MATGL) ---
import matgl
from matgl.ext.ase import M3GNetCalculator

class Relaxer:
    """
    A class for performing GNN-based structural relaxation using M3GNet.
    Now tracks Initial and Final energy.
    """

    def __init__(self):
        # Load M3GNet potential (PES)
        self._potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        self._calculator = M3GNetCalculator(potential=self._potential)

    def relax(self, structure: Structure, fmax=0.01, steps=100, log_file='relaxation.log') -> dict:
        """
        Relaxes a structure and returns initial/final energies.
        """
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            return {'error': f'ASE Conversion failed: {e}'}

        # Attach ML Potential
        atoms.calc = self._calculator

        # 1. Calculate Initial Energy (Before Relaxation)
        try:
            initial_energy = atoms.get_potential_energy()
            num_atoms = len(atoms)
            initial_energy_per_atom = initial_energy / num_atoms
        except Exception as e:
            return {'error': f'Initial energy calculation failed: {e}'}

        # 2. Setup Relaxation
        ucf = UnitCellFilter(atoms)
        optimizer = LBFGS(ucf, logfile=log_file)

        # 3. Run Optimization
        try:
            optimizer.run(fmax=fmax, steps=steps)
        except Exception as e:
            return {'error': f'Optimization failed: {e}'}

        # 4. Get Final Metrics
        final_structure = AseAtomsAdaptor.get_structure(atoms)
        final_energy = atoms.get_potential_energy()
        final_energy_per_atom = final_energy / num_atoms
        
        forces = atoms.get_forces()
        max_force = np.sqrt((forces**2).sum(axis=1).max()) if forces is not None else 0.0
        is_converged = max_force <= fmax

        return {
            'final_structure': final_structure,
            'initial_energy_per_atom': initial_energy_per_atom,
            'final_energy_per_atom': final_energy_per_atom,
            'energy_delta': initial_energy_per_atom - final_energy_per_atom,
            'is_converged': is_converged,
            'max_force': max_force,
            'num_steps': optimizer.get_number_of_steps(),
        }
