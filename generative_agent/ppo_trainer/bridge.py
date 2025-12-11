# generative_agent/ppo_trainer/bridge.py
import numpy as np
import torch
from pymatgen.core import Structure, Lattice, Element
import warnings

# Suppress pymatgen warnings about occupancy (common in generative models)
warnings.filterwarnings("ignore")

class TensorBridge:
    """
    Handles the conversion between PyTorch Tensors (Model Output) 
    and Pymatgen Structures (Physics Object) without using disk I/O.
    """
    
    @staticmethod
    def _to_numpy(x):
        """Helper to convert Tensor/List to Numpy array safely."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    @staticmethod
    def batch_to_structures(G, L, XYZ, A, M):
        """
        Convert batch of PyTorch tensors to a list of Pymatgen Structures.
        
        Args:
            G: Space group numbers (batch,)
            L: Lattice parameters [a, b, c, alpha, beta, gamma] (batch, 6)
            XYZ: Fractional coordinates (batch, n_max, 3)
            A: Atom types (batch, n_max) - Integer Z numbers
            M: Multiplicity (batch, n_max) - Used to filter padded atoms
            
        Returns:
            List of valid pymatgen.Structure objects. None for failed conversions.
        """
        # Convert Torch tensors to standard Numpy arrays for CPU processing
        G_np = TensorBridge._to_numpy(G)
        L_np = TensorBridge._to_numpy(L)
        XYZ_np = TensorBridge._to_numpy(XYZ)
        A_np = TensorBridge._to_numpy(A)
        M_np = TensorBridge._to_numpy(M)
        
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
                # print(f"Structure generation failed: {e}")
                structures.append(None)
                
        return structures

    @staticmethod
    def _single_to_structure(g, l_params, xyz, a, m):
        # 1. Reconstruct Lattice
        # l_params is [a, b, c, alpha, beta, gamma]
        # Ensure positive lattice constants to avoid Pymatgen errors
        if np.any(l_params[:3] <= 0.1):
            raise ValueError("Lattice constants too small")
            
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
        # We use the raw lattice and co