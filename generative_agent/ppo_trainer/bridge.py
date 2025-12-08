# generative_agent/ppo_trainer/bridge.py
import numpy as np
import jax.numpy as jnp
from pymatgen.core import Structure, Lattice, Element
import warnings

# Suppress pymatgen warnings about occupancy (common in generative models)
warnings.filterwarnings("ignore")

class TensorBridge:
    """
    Handles the conversion between JAX Tensors (Model Output) 
    and Pymatgen Structures (Physics Object) without using disk I/O.
    """
    
    @staticmethod
    def batch_to_structures(G, L, XYZ, A, M):
        """
        Convert batch of JAX tensors to a list of Pymatgen Structures.
        
        Args:
            G: Space group numbers (batch,)
            L: Lattice parameters [a, b, c, alpha, beta, gamma] (batch, 6)
            XYZ: Fractional coordinates (batch, n_max, 3)
            A: Atom types (batch, n_max) - Integer Z numbers
            M: Multiplicity (batch, n_max) - Used to filter padded atoms
            
        Returns:
            List of valid pymatgen.Structure objects. None for failed conversions.
        """
        # Convert JAX/Torch tensors to standard Numpy arrays for CPU processing
        G_np = np.array(G)
        L_np = np.array(L)
        XYZ_np = np.array(XYZ)
        A_np = np.array(A)
        M_np = np.array(M)
        
        structures = []
        
        for i in range(len(G_np)):
            try:
                struct = TensorBridge._single_to_structure(
                    G_np[i], L_np[i], XYZ_np[i], A_np[i], M_np[i]
                )
                structures.append(struct)
            except Exception as e:
                # This happens if the model generates physically impossible geometry
                # We return None, and the RewardCalculator will give it a penalty later.
                structures.append(None)
                
        return structures

    @staticmethod
    def _single_to_structure(g, l_params, xyz, a, m):
        # 1. Reconstruct Lattice
        # l_params is [a, b, c, alpha, beta, gamma]
        lattice = Lattice.from_parameters(*l_params)
        
        # 2. Filter Valid Atoms
        # The model outputs a fixed size array (e.g. 24 atoms). 
        # We only keep atoms where Species(A) > 0 and Multiplicity(M) > 0
        valid_indices = np.where((m > 0) & (a > 0))[0]
        
        species = []
        coords = []
        
        for idx in valid_indices:
            z = int(a[idx])
            # Element 0 is padding, 119+ is invalid.
            if 0 < z < 119: 
                species.append(Element.from_Z(z))
                coords.append(xyz[idx])
                
        if len(species) == 0:
            raise ValueError("Empty structure generated")
            
        # 3. Create Structure
        # We use the raw lattice and coords because the model has already 
        # performed the symmetry projection in the XYZ step.
        return Structure(lattice, species, coords)