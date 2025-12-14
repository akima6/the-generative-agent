# generative_agent/ppo_trainer/bridge.py
import numpy as np
import torch
from pymatgen.core import Structure, Lattice, Element
import warnings

# Suppress pymatgen warnings
warnings.filterwarnings("ignore")

class TensorBridge:
    """
    Handles the conversion between PyTorch Tensors and Pymatgen Structures.
    """
    
    # --- PHYSICS PATCH: THE COMPRESSOR ---
    # The model currently outputs lattices that are ~1.7x too large (Gas Density).
    # We apply a global correction to shrink them into the solid phase.
    # 0.6 * Length = ~4.6x Density Increase.
    LATTICE_SCALING_FACTOR = 0.55 

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    @staticmethod
    def batch_to_structures(G, L, XYZ, A, M):
        # Convert Torch tensors to standard Numpy arrays
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
                # print(f"[Bridge Error] {e}") # Optional logging
                structures.append(None)
                
        return structures

    @staticmethod
    def _single_to_structure(g, l_params, xyz, a, m):
        # 1. Apply Physics Patch (Compress Lattice)
        # l_params is [a, b, c, alpha, beta, gamma]
        # We only scale a, b, c (indices 0, 1, 2)
        l_params = l_params.copy()
        l_params[:3] = l_params[:3] * TensorBridge.LATTICE_SCALING_FACTOR
        
        # 2. Validation
        if np.isnan(l_params).any(): raise ValueError("NaN Lattice")
        if np.any(l_params[:3] <= 0.1): raise ValueError("Lattice too small")
        
        # 3. Reconstruct Lattice
        # Pymatgen expects degrees for angles (which sample.py provides)
        lattice = Lattice.from_parameters(*l_params)
        
        # 4. Filter Valid Atoms
        valid_indices = np.where((m > 0) & (a > 0))[0]
        
        species = []
        coords = []
        
        for idx in valid_indices:
            z = int(a[idx])
            if 0 < z < 119: 
                species.append(Element.from_Z(z))
                coords.append(xyz[idx])
                
        if len(species) == 0:
            raise ValueError("Empty structure")
            
        return Structure(lattice, species, coords)
