# generative_agent/evaluation_engine/relaxer.py
# ... (All imports remain the same) ...

# ... (Class Relaxer __init__ remains the same) ...

class Relaxer:
    """
    A class for performing GNN-based structural relaxation using M3GNet.
    Now tracks Initial and Final energy.
    """
    # ... (init remains the same) ...

    def relax(self, structure: Structure, fmax=0.01, steps=100, log_file='relaxation.log') -> dict:
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            return {'error': f'ASE Conversion failed: {e}'}

        # --- FINAL STABILITY GUARD (CRITICAL) ---
        # 1. Rattle Check: If all atoms are exactly at a symmetry point, force numerical noise.
        # This prevents internal ASE/MatGL symmetry assertion failures.
        initial_coords = atoms.get_positions()
        coords_are_symmetric = np.allclose(initial_coords, np.round(initial_coords), atol=1e-5)
        
        if coords_are_symmetric:
             # Add a tiny random perturbation (0.01 Angstroms)
             atoms.set_positions(initial_coords + (np.random.rand(*initial_coords.shape) - 0.5) * 0.02)
             print("[Relaxer] RATTLED: Applied tiny perturbation to break perfect symmetry.")

        # 2. Distance Check: If atoms are too close, it causes massive force spikes (NaN/Inf).
        # We perform a simple check.
        from ase.geometry import get_distances
        distances = get_distances(atoms.get_positions(), cell=atoms.get_cell(), pbc=atoms.get_pbc())[1]
        min_dist = distances.min() if distances.size > 0 else 1.0 # Ensure non-empty check
        
        if min_dist < 0.5: # 0.5 Angstrom is chemically impossible
             return {'error': f'Atoms too close (Min Dist: {min_dist:.2f} A).'}
        # ----------------------------------------
        
        # Attach ML Potential
        atoms.calc = self._calculator

        # 1. Calculate Initial Energy (Before Relaxation)
        try:
            initial_energy = atoms.get_potential_energy()
            num_atoms = len(atoms)
            initial_energy_per_atom = initial_energy / num_atoms
        except Exception as e:
            return {'error': f'Initial energy calculation failed on potential. {e}'} # MatGL crashed here!

        # 2. Setup Relaxation
        # ... (rest of the relaxer code is standard) ...
        # ... (final return is standard) ...
        
        # 2. Setup Relaxation
        ucf = UnitCellFilter(atoms)
        optimizer = LBFGS(ucf, log_file=log_file)

        # 3. Run Optimization
        try:
            optimizer.run(fmax=fmax, steps=steps)
        except Exception as e:
            return {'error': f'Optimization failed: {e}'}

        # 4. Get Final Metrics
        final_structure = AseAtomsAdaptor.get_structure(atoms)
        
        # Final energy calculation can sometimes crash again. Add protection.
        try:
            final_energy = atoms.get_potential_energy()
        except Exception as e:
            return {'error': f'Final energy calculation failed: {e}'}
            
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
