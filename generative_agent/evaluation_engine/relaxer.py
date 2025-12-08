# relaxer.py (Corrected Version 2.2)

# --- Core Libraries ---
import torch
import numpy as np

# --- Materials Science Libraries ---
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

# --- Atomic Simulation Environment (ASE) for Optimization ---
from ase.optimize import LBFGS
# [# FIX 1]: Import UnitCellFilter from its new, correct location
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
        """
        Initializes the Relaxer by loading the pre-trained M3GNet model
        and setting up the ASE-compatible calculator.
        """
        #print("Initializing Relaxer: Loading M3GNet potential...")
        self._potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        self._calculator = M3GNetCalculator(potential=self._potential)
        #print("M3GNet potential loaded successfully.")

    def relax(self, structure: Structure, fmax=0.01, steps=100, log_file='relaxation.log') -> dict:
        """
        Relaxes a pymatgen Structure, optimizing both atomic positions and the
        unit cell, mimicking the high-throughput approach of the BOWSR paper.

        Args:
            structure (pymatgen.core.structure.Structure): The input crystal structure to relax.
            fmax (float): The maximum force tolerance (in eV/Å). This is the target for
                          a fully converged relaxation. Defaults to 0.01 eV/Å.
            steps (int): The maximum number of optimization steps. The relaxation will
                         stop here even if fmax is not reached. Defaults to 100.
            log_file (str): Path to the log file for the optimization progress.

        Returns:
            dict: A dictionary containing the final relaxed structure and metadata.
        """
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            print(f"Error: Could not convert input structure to ASE Atoms object. {e}")
            return {'error': 'Input structure conversion failed.'}

        # [# FIX 2]: Use the modern syntax for setting the calculator
        atoms.calc = self._calculator

        # Wrap the atoms object with a UnitCellFilter for full cell relaxation.
        ucf = UnitCellFilter(atoms)

        optimizer = LBFGS(ucf, logfile=log_file)

        # Run the optimization.
        optimizer.run(fmax=fmax, steps=steps)

        # Convert back to a pymatgen Structure
        relaxed_pymatgen_structure = AseAtomsAdaptor.get_structure(atoms)

        # Compile the output dictionary
        final_energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = np.sqrt((forces**2).sum(axis=1).max()) if forces is not None else 0.0
        
        # [# FIX 3]: The correct way to determine convergence.
        # We check if the final max force is below our target tolerance.
        is_converged = max_force <= fmax

        return {
            'final_structure': relaxed_pymatgen_structure,
            'final_energy': final_energy,
            'is_converged': is_converged,
            'max_force': max_force,
            'num_steps': optimizer.get_number_of_steps(),
        }

# --- Self-Contained Test Block ---
if __name__ == '__main__':
    relaxer = Relaxer()

    lattice = np.array([[5.8, 0, 0], [0, 5.5, 0], [0, 0, 5.6]])
    species = ["Na", "Cl"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    nacl_structure = Structure(lattice, species, coords)

    rattled_nacl = nacl_structure.copy()
    rattled_nacl.perturb(distance=0.1)
    print("\n--- Initial Rattled & Distorted Structure ---")
    print(rattled_nacl)
    CifWriter(rattled_nacl).write_file('initial_distorted.cif')

    print("\n--- Starting Relaxation ---")
    results = relaxer.relax(rattled_nacl, fmax=0.01, steps=100)
    print("--- Relaxation Finished ---")

    final_structure = results.get('final_structure')
    if final_structure:
        print("\n--- Relaxation Results ---")
        print(f"Converged within force tolerance: {results['is_converged']}")
        print(f"Number of optimization steps: {results['num_steps']}")
        print(f"Final Energy (eV): {results['final_energy']:.4f}")
        print(f"Final Max Force (eV/Å): {results['max_force']:.4f}")
        print("\n--- Final Relaxed Structure ---")
        print(final_structure)
        CifWriter(final_structure).write_file('final_relaxed.cif')
        print("\nInitial and final structures saved to .cif files for visual comparison.")
    else:
        print("\n--- Relaxation Failed ---")
        print(results)