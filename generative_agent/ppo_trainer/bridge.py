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
                # DEBUG PRINT: Print exactly why it failed
                print(f"[Bridge Error Sample {i}] {e}")
                # Print Lattice params to check for NaNs or Negatives
                print(f"  Lattice: {L_np[i]}")
                structures.append(None)
                
        return structures

    @staticmethod
    def _single_to_structure(g, l_params, xyz, a, m):
        # 1. Reconstruct Lattice
        # l_params is [a, b, c, alpha, beta, gamma]
        
        # Check for NaNs
        if np.isnan(l_params).any():
            raise ValueError("Lattice parameters contain NaNs")
            
        # Check for non-positive lengths
        if np.any(l_params[:3] <= 1e-3):
            raise ValueError(f"Lattice lengths too small or negative: {l_params[:3]}")
            
        # Pymatgen validation for angles
        if np.any(l_params[3:] <= 0) or np.any(l_params[3:] >= 180):
             raise ValueError(f"Invalid Lattice Angles: {l_params[3:]}")

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
            raise ValueError("Empty structure generated (No valid atoms)")
            
        # 3. Create Structure
        return Structure(lattice, species, coords)
