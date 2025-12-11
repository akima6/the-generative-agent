import torch
import numpy as np
from crystalformer.src.wyckoff import mult_table

def make_lattice_mask():
    '''
    return mask for independent lattice params 
    '''
    # 1-2
    # 3-15 
    # 16-74
    # 75-142
    # 143-194
    # 195-230    
    mask = [1, 1, 1, 1, 1, 1] * 2 +\
           [1, 1, 1, 0, 1, 0] * 13+\
           [1, 1, 1, 0, 0, 0] * 59+\
           [1, 0, 1, 0, 0, 0] * 68+\
           [1, 0, 1, 0, 0, 0] * 52+\
           [1, 0, 0, 0, 0, 0] * 36

    # Convert to tensor. Note: Device will be CPU by default, moved when needed.
    return torch.tensor(mask, dtype=torch.int32).reshape(230, 6)

def symmetrize_lattice(spacegroup, lattice):
    '''
    Place lattice params into lattice according to the space group constraints.
    Args:
        spacegroup: (Batch,) or scalar Int Tensor
        lattice: (Batch, 6) Float Tensor [a, b, c, alpha, beta, gamma]
    Returns:
        L: (Batch, 6) Symmetrized lattice
    '''
    # Ensure inputs are tensors
    if not isinstance(spacegroup, torch.Tensor):
        spacegroup = torch.tensor(spacegroup)
    if not isinstance(lattice, torch.Tensor):
        lattice = torch.tensor(lattice)
        
    device = lattice.device
    spacegroup = spacegroup.to(device)

    # Handle single sample case by unsqueezing
    is_batch = lattice.ndim == 2
    if not is_batch:
        lattice = lattice.unsqueeze(0)
        spacegroup = spacegroup.reshape(1)

    a, b, c, alpha, beta, gamma = lattice.unbind(-1)
    
    # Constants
    zero = torch.tensor(0.0, device=device, dtype=lattice.dtype)
    ninety = torch.tensor(90.0, device=device, dtype=lattice.dtype)
    onetwenty = torch.tensor(120.0, device=device, dtype=lattice.dtype)

    # Construct candidate lattices
    # Triclinic (1-2): [a, b, c, alpha, beta, gamma] (Input)
    
    # Monoclinic (3-15): [a, b, c, 90, beta, 90]
    L_mono = torch.stack([a, b, c, ninety, beta, ninety], dim=-1)
    
    # Orthorhombic (16-74): [a, b, c, 90, 90, 90]
    L_ortho = torch.stack([a, b, c, ninety, ninety, ninety], dim=-1)
    
    # Tetragonal (75-142): [a, a, c, 90, 90, 90]
    L_tetra = torch.stack([a, a, c, ninety, ninety, ninety], dim=-1)
    
    # Hexagonal/Trigonal (143-194): [a, a, c, 90, 90, 120]
    L_hex = torch.stack([a, a, c, ninety, ninety, onetwenty], dim=-1)
    
    # Cubic (195-230): [a, a, a, 90, 90, 90]
    L_cubic = torch.stack([a, a, a, ninety, ninety, ninety], dim=-1)

    # Apply constraints hierarchically using torch.where
    # Logic:
    # If SG <= 2: