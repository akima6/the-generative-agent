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
    
    # --- PHYSICS PATCH: EXPANSION MODE ---
    # The crystals are currently too dense (~5.3 g/cm3 vs 2.2 g/cm3).
    # MatGL fails because atoms are overlapping.
    # We expand the lattice by 35% to reduce density.
    LATTICE_SCALING_FACTOR = 1.35

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
                structures.append(None)
                
        return structures

    @staticmethod
    def _single_to_structure(g, l_params, xyz, a, m):
        l_params = l_params.copy()
        
        # Apply Scaling Patch
        l_params[:3] = l_params[:3] * TensorBridge.LATTICE_SCALING_FACTOR
        
        if np.isnan(l_params).any(): raise ValueError("NaN Lattice")
        if np.any(l_params[:3] <= 0.5): raise ValueError("Lattice collapsed")
        
        lattice = Lattice.from_parameters(*l_params)
        
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
