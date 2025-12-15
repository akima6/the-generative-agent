# generative_agent/evaluation_engine/relaxer.py (FINAL VERSION)

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
from ase.geometry import get_distances

# --- Materials Graph Library (MATGL) for the ML Potential ---
import matgl
from matgl.ext.ase import M3GNetCalculator

class Relaxer:
    """
    A class for performing GNN-based structural relaxation using M3GNet.
    Tracks Initial and Final energy per atom.
    """

    def __init__(self):
        """
        Initializes the Relaxer by loading the M3GNet potential.
        """
        # --- CRITICAL FIX: Load the MatGL Model ---
        # This is the essential step that was missing.
        # print("Initializing Relaxer: Loading M3GNet potential...")
        self._potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        self._calculator = M3GNetCalculator(potential=self._potential)
        # print("M3GNet potential loaded successfully.")

    def relax(self, structure: Structure, fmax=0.01, steps=100, log_file='relaxation.log') -> dict:
        """
        Relaxes a structure and returns initial/final energies per atom.
        """
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            return {'error': f'ASE Conversion failed: {e}'}

        # --- FINAL STABILITY GUARD (Prevents NaN/Inf Explosions) ---
        initial_coords = atoms.get_positions()
        
        # 1. Rattle Check: Break perfect symmetry
        if np.allclose(initial_coords, np.round(initial_coords), atol=1e-5):
             atoms.set_positions(initial_coords + (np.random.rand(*initial_coords.shape) - 0.5) * 0.02)
             # print("[Relaxer] RATTLED: Applied perturbation.")

        # 2. Distance Check: Prevent immediate explosion
        distances = get_distances(atoms.get_positions(), cell=atoms.get_cell(), pbc=atoms.get_pbc())[1]
        min_dist = distances.min() if distances.size > 0 else 1.0
        
        if min_dist < 0.5:
             return {'error': f'Atoms too close (Min Dist: {min_dist:.2f} A).'}
        # ----------------------------------------
        
        # Attach ML Potential
        atoms.calc = self._calculator
        num_atoms = len(atoms)

        # 1. Calculate Initial Energy (Before Relaxation)
        try:
            initial_energy = atoms.get_potential_energy()
            initial_energy_per_atom = initial_energy / num_atoms
        except Exception as e:
            return {'error': f'Initial energy calculation failed on potential. {e}'}

        # 2. Setup and Run Optimization
        ucf = UnitCellFilter(atoms)
        optimizer = LBFGS(ucf, log_file=log_file)

        try:
            optimizer.run(fmax=fmax, steps=steps)
        except Exception as e:
            return {'error': f'Optimization failed: {e}'}

        # 3. Get Final Metrics
        final_structure = AseAtomsAdaptor.get_structure(atoms)
        
        try:
            final_energy = atoms.get_potential_energy()
            final_energy_per_atom = final_energy / num_atoms
        except Exception as e:
            return {'error': f'Final energy calculation failed: {e}'}
            
        forces = atoms.get_forces()
        max_force = np.sqrt((forces**2).sum(axis=1).max()) if forces is not None else 0.0
        is_converged = max_force <= fmax

        return {
            'final_structure': final_structure,
            'initial_energy_per_atom': final_energy_per_atom, # Energy is always final/relaxed
            'final_energy_per_atom': final_energy_per_atom,
            'energy_delta': initial_energy_per_atom - final_energy_per_atom,
            'is_converged': is_converged,
            'max_force': max_force,
            'num_steps': optimizer.get_number_of_steps(),
        }
