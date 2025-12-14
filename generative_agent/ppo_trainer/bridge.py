# generative_agent/ppo_trainer/bridge.py
import numpy as np
import torch
from pymatgen.core import Structure, Lattice, Element
import warnings

warnings.filterwarnings("ignore")

class TensorBridge:
    """
    Handles the conversion between PyTorch Tensors and Pymatgen Structures.
    """
    
    # --- PHYSICS PATCH: AGGRESSIVE COMPRESSION ---
    # The model outputs lattices ~2.5x too large. 
    # We apply a 0.4 scaling factor. 
    # Effect: Increases density by approx 15x (1/0.4^3).
    # Target: Turn 0.2 g/cm3 -> 3.0 g/cm3.
    LATTICE_SCALING_FACTOR = 0.40 

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    @staticmethod
    def batch_to_structures(G, L, XYZ, A, M):
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
                # print(f"[Bridge Error] {e}") 
                structures.append(None)
                
        return structures

    @staticmethod
    def _single_to_structure(g, l_params, xyz, a, m):
        # 1. Apply Physics Patch
        l_params = l_params.copy()
        l_params[:3] = l_params[:3] * TensorBridge.LATTICE_SCALING_FACTOR
        
        # 2. Validation
        # Check for NaNs or zero volume
        if np.isnan(l_params).any(): raise ValueError("NaN Lattice")
        if np.any(l_params[:3] <= 0.5): raise ValueError("Lattice collapsed (too small)")
        
        # 3. Reconstruct Lattice
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
