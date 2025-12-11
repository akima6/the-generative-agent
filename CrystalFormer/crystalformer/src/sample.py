import torch
import torch.nn.functional as F
import numpy as np

# We assume these tables are available from the rewritten wyckoff.py
from crystalformer.src.wyckoff import mult_table, symops
# We assume lattice.py will be rewritten. 
# If not, the import will fail. 
# For safety, I will keep the import but you must rewrite lattice.py next.
from crystalformer.src.lattice import symmetrize_lattice

def sample_von_mises(loc, kappa, shape):
    """
    Samples from Von Mises distribution using PyTorch.
    Approximation using rejection sampling or similar method suitable for GPU.
    Reference: Best & Fisher (1979).
    """
    # Simple rejection sampling for Von Mises
    # Since kappa can be large, we use a standard implementation approach.
    # However, for ML inference, often a wrapped normal approximation or 
    # specific torch distributions are used. 
    # Torch distributions has VonMises!
    
    # torch.distributions.von_mises.VonMises is available in newer torch versions.
    # loc: (Batch, Kx)
    # kappa: (Batch, Kx)
    
    # Ensure kappa > 0
    kappa = torch.clamp(kappa, min=1e-6)
    
    # We need to sample (Batch, Kx). 
    # But output shape requested is just (Batch,). Wait.
    # The original passed (batchsize,) as shape but loc/kappa were (batchsize, Kx)?
    # No, in sample_x, loc/kappa are sliced to (batchsize,) using the sampled k.
    # So loc and kappa are 1D tensors (Batch,).
    
    vm = torch.distributions.von_mises.VonMises(loc, kappa)
    sample = vm.sample()
    return sample

def project_xyz(g, w, x, idx):
    """
    Apply Wyckoff symmetry op to sampled fc.
    g: scalar (space group)
    w: scalar (wyckoff index)
    x: (3,) tensor
    idx: scalar (0)
    """
    # symops is (230, 28, 576, 3, 4)
    # Ensure inputs are correct types
    g_idx = g - 1
    
    # PyTorch indexing
    # We need to handle if symops is on CPU or GPU.
    device = x.device
    op = symops[g_idx, w, idx].to(device).reshape(3, 4)
    
    affine_point = torch.cat([x, torch.tensor([1.0], device=device)])
    x_new = torch.matmul(op, affine_point)
    x_new -= torch.floor(x_new)
    return x_new

def project_xyz_batch(G, W, X, idx=0):
    """
    Batched version of project_xyz.
    G: (Batch,)
    W: (Batch,)
    X: (Batch, 3)
    """
    batch_size = G.shape[0]
    device = X.device
    
    # G, W are indices. 
    # symops: (230, 28, 576, 3, 4)
    # We need symops[G-1, W, idx] -> (Batch, 3, 4)
    
    # Gather is tricky for multi-dim. 
    # Move table to device if small enough, or index on CPU.
    # symops is ~300MB. Moving to GPU once is fine.
    symops_gpu = symops.to(device)
    
    # Indexing
    # G-1: (Batch,)
    # W: (Batch,)
    # idx: scalar
    
    ops = symops_gpu[G-1, W, idx] # (Batch, 3, 4)
    
    ones = torch.ones((batch_size, 1), device=device)
    affine_points = torch.cat([X, ones], dim=1).unsqueeze(2) # (Batch, 4, 1)
    
    # (Batch, 3, 4) @ (Batch, 4, 1) -> (Batch, 3, 1)
    x_new = torch.bmm(ops, affine_points).squeeze(2)
    x_new -= torch.floor(x_new)
    return x_new

def inference(model, g, W, A, X, Y, Z, is_train=False):
    """
    Runs model inference for a batch.
    g: (Batch,) or Scalar
    W: (Batch, n_max)
    ...
    """
    # Reconstruct XYZ
    XYZ = torch.stack([X, Y, Z], dim=-1) # (Batch, n_max, 3)
    
    # Mult table lookup
    # mult_table: (230, 28)
    # G: (Batch,)
    # W: (Batch, n_max)
    # We need M: (Batch, n_max)
    
    device = W.device
    mult_table_gpu = mult_table.to(device)
    
    # Broadcast G to match W
    G_expanded = (g - 1).unsqueeze(1).expand(-1, W.shape[1])
    M = mult_table_gpu[G_expanded, W]
    
    # Forward
    return model(g, XYZ, A, W, M, is_train)

def sample_top_p(logits, p, temperature):
    """
    Top-p sampling.
    logits: (Batch, C)
    """
    if p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=1), dim=1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        
        logits = logits.masked_fill(indices_to_remove, -1e10)
    
    # Sample
    probs = F.softmax(logits / temperature, dim=1)
    samples = torch.multinomial(probs, 1).squeeze(1) # (Batch,)
    return samples

def sample_x_fn(h_x, Kx, top_p, temperature, batchsize):
    """
    Sample coordinate from Mixture of Von Mises.
    h_x: (Batch, 3*Kx)
    """
    coord_types = 3 * Kx
    relevant = h_x[:, :coord_types]
    x_logit, loc, kappa = torch.split(relevant, [Kx, Kx, Kx], dim=-1)
    
    # Sample k (mixture component)
    k = sample_top_p(x_logit, top_p, temperature) # (Batch,)
    
    # Select loc and kappa corresponding to k
    # loc: (Batch, Kx)
    # k: (Batch,) -> need (Batch, 1) to gather
    k_uns = k.unsqueeze(1)
    
    selected_loc = torch.gather(loc, 1, k_uns).squeeze(1)
    selected_kappa = torch.gather(kappa, 1, k_uns).squeeze(1)
    
    # Softplus on kappa was done in model?
    # No, in transformer.py: "x_kappa = F.softplus(x_kappa)"
    # BUT, we are taking raw h_x here? 
    # In transformer.py, renormalize(h_x) is called at the end of forward.
    # The output of model() IS renormalized.
    # So loc is raw, kappa is softplussed, logit is log-softmaxed.
    # Wait, in transformer.py:
    # x_logit -= logsumexp
    # x_kappa = softplus
    # So inputs here are already processed.
    
    # Sample from VM
    x = sample_von_mises(selected_loc, selected_kappa / temperature, (batchsize,))
    
    # Wrap to [0, 1]
    x = (x + np.pi) / (2.0 * np.pi)
    return x

def sample_crystal(key, transformer, params, n_max, batchsize, atom_types, wyck_types, Kx, Kl, g, w_mask, atom_mask, top_p, temperature, T1, constraints):
    """
    Autoregressive sampling of a crystal batch.
    key: Ignored (PyTorch uses global RNG)
    transformer: PyTorch model
    """
    device = next(transformer.parameters()).device
    
    # Initialize state
    # G can be scalar or tensor. 
    # g passed here is likely an integer (SPACE_GROUP).
    if isinstance(g, int):
        G = torch.full((batchsize,), g, device=device).long()
    else:
        G = g.to(device) # Assuming it's already a tensor of size batch
        
    W = torch.zeros((batchsize, n_max), dtype=torch.long, device=device)
    A = torch.zeros((batchsize, n_max), dtype=torch.long, device=device)
    X = torch.zeros((batchsize, n_max), device=device)
    Y = torch.zeros((batchsize, n_max), device=device)
    Z = torch.zeros((batchsize, n_max), device=device)
    L = torch.zeros((batchsize, n_max, Kl + 12 * Kl), device=device) # 13*Kl
    
    # Autoregressive Loop
    for i in range(n_max):
        # Run inference
        output = inference(transformer, G, W, A, X, Y, Z) # (Batch, 5*n+1, Output)
        
        # 1. Sample W
        # Index: 5*i + 0? No.
        # h0 is index 0.
        # Atom 0 starts at index 1.
        # h_al, h_x, h_y, h_z, w_logit
        # Sequence: h0, [Atom0 Block], [Atom1 Block]...
        # Atom i block indices: 1 + 5*i ... 1 + 5*i + 4
        
        # Original JAX code: 
        # w_logit = inference(...)[:, 5*i]
        # This implies:
        # i=0 -> index 0 (h0)
        # i=1 -> index 5 (last of Atom0 block? No.)
        # Let's trace transformer.py output.
        # h = concatenate([h0, h_final])
        # h0: (1, output)
        # h_final: (5*n, output) -> (h_al, h_x, h_y, h_z, w_logit) interleaved.
        # So:
        # Index 0: h0
        # Index 1: h_al (Atom 0)
        # Index 2: h_x (Atom 0)
        # Index 3: h_y (Atom 0)
        # Index 4: h_z (Atom 0)
        # Index 5: w_logit (Atom 0)
        
        # Original JAX Code in sample_crystal says:
        # w_logit = inference(...)[:, 5*i]
        # For i=0, index 0. Correct (h0).
        # For i=1, index 5. Correct (w_logit of Atom 0).
        # This means w_logit at index 5 predicts W for Atom 1?
        # Yes, standard autoregressive. w_logit_prev predicts w_curr.
        
        w_logit = output[:, 5 * i, :wyck_types]
        w = sample_top_p(w_logit, top_p, temperature)
        
        # Masking override
        if w_mask is not None:
            # w_mask is typically None or shape (n_max,). 
            # If provided, force specific Wyckoff?
            # JAX: w = w.at[:].set(w_mask[i])
            # This sets ALL batch elements to w_mask[i].
            if w_mask[i] is not None: # Assuming w_mask is a list/array where None means sample
                w[:] = w_mask[i]
                
        W[:, i] = w

        # 2. Sample A
        # Index: 5*i + 1 (h_al for Atom i)
        # We need to re-run inference? 
        # JAX code runs inference inside the loop 5 times!
        # "h_al = inference(...)[:, 5*i+1]"
        # This implies the model is run 5 times per atom step.
        # Optimization: We can run it once if we trust the state doesn't change until we write to it?
        # BUT: In JAX code, W is updated: "W = W.at[:, i].set(w)".
        # Then "h_al = inference(...)".
        # So yes, we must run inference again because W changed.
        # This is very expensive (5 * N * Batch * Transformer passes).
        # Standard Transformers use KV-caching. Here we just re-run.
        
        # Optim: We only updated W[:, i]. 
        # We run inference again.
        output = inference(transformer, G, W, A, X, Y, Z)
        h_al = output[:, 5 * i + 1]
        a_logit = h_al[:, :atom_types]
        
        # Masking
        # atom_mask[i]: (atom_types,) boolean?
        # JAX: a_logit + where(atom_mask[i], 0, -1e10).
        # PyTorch:
        if atom_mask is not None:
            mask_i = atom_mask[i] # (atom_types,)
            # Broadcast to batch
            mask_i = mask_i.to(device).unsqueeze(0)
            a_logit = a_logit + torch.where(mask_i > 0, torch.tensor(0.0, device=device), torch.tensor(-1e10, device=device))
            
        # Temperature scheduling
        curr_temp = T1 if i == 0 else temperature
        
        a = sample_top_p(a_logit, top_p, curr_temp)
        
        # Constraints (copy from previous if constraint met)
        # constraints[i] < i implies we copy?
        # JAX: cond(constraints[i] < i, lambda: A[:, constraints[i]], lambda: a)
        if constraints is not None and constraints[i] < i:
            # Copy from previous atom
            # We need to copy per-batch or global?
            # constraints is usually fixed per step.
            ref_idx = constraints[i]
            a = A[:, ref_idx]
            
        A[:, i] = a
        
        # Save Lattice Params (from h_al)
        # Lattice types indices
        # start: atom_types
        # end: atom_types + Kl + 12*Kl
        l_params = h_al[:, atom_types : atom_types + Kl + 12 * Kl]
        L[:, i] = l_params
        
        # 3. Sample X
        # Index: 5*i + 2
        output = inference(transformer, G, W, A, X, Y, Z)
        h_x = output[:, 5 * i + 2]
        x = sample_x_fn(h_x, Kx, top_p, temperature, batchsize)
        
        # Project to first WP
        # project_xyz expects 3D point.
        # Construct (x, 0, 0)
        xyz_temp = torch.stack([x, torch.zeros(batchsize, device=device), torch.zeros(batchsize, device=device)], dim=1)
        xyz_proj = project_xyz_batch(G, w, xyz_temp, idx=0)
        x = xyz_proj[:, 0]
        X[:, i] = x
        
        # 4. Sample Y
        # Index: 5*i + 3
        output = inference(transformer, G, W, A, X, Y, Z)
        h_y = output[:, 5 * i + 3]
        y = sample_x_fn(h_y, Kx, top_p, temperature, batchsize)
        
        # Project (X_fixed, y, 0)
        xyz_temp = torch.stack([X[:, i], y, torch.zeros(batchsize, device=device)], dim=1)
        xyz_proj = project_xyz_batch(G, w, xyz_temp, idx=0)
        y = xyz_proj[:, 1]
        Y[:, i] = y
        
        # 5. Sample Z
        # Index: 5*i + 4
        output = inference(transformer, G, W, A, X, Y, Z)
        h_z = output[:, 5 * i + 4]
        z = sample_x_fn(h_z, Kx, top_p, temperature, batchsize)
        
        # Project (X_fixed, Y_fixed, z)
        xyz_temp = torch.stack([X[:, i], Y[:, i], z], dim=1)
        xyz_proj = project_xyz_batch(G, w, xyz_temp, idx=0)
        z = xyz_proj[:, 2]
        Z[:, i] = z
        
    # End Loop
    
    # Process Lattice
    # We take L from the step corresponding to the last 'real' atom?
    # JAX: 
    # num_sites = sum(A!=0)
    # L = L[arange, num_sites, :] -> This means we take the prediction made by the last valid atom?
    # Note: L[:, i] stored h_al prediction.
    # We probably want L from the last atom that isn't padding.
    # If A=0 (pad), we take L from previous?
    # Actually, JAX code uses 'num_sites' as index.
    # If num_sites=N, we take L[N].
    # But L is size n_max.
    # If num_sites < n_max, we take L at index num_sites.
    
    num_sites = torch.sum(A != 0, dim=1) # (Batch,)
    # Gather L
    # L: (Batch, n_max, Params)
    # index: (Batch, 1, Params)
    num_sites_uns = num_sites.unsqueeze(1).unsqueeze(2).expand(-1, -1, L.shape[2])
    # Clamp to n_max-1 just in case
    num_sites_uns = torch.clamp(num_sites_uns, max=n_max-1)
    
    L_selected = torch.gather(L, 1, num_sites_uns).squeeze(1) # (Batch, Params)
    
    # Split L
    # l_logit (Kl), mu (6*Kl), sigma (6*Kl)
    l_logit, mu, sigma = torch.split(L_selected, [Kl, 6*Kl, 6*Kl], dim=-1)
    
    # Sample k
    k = sample_top_p(l_logit, top_p, temperature) # (Batch,)
    
    # Gather mu/sigma
    mu = mu.reshape(batchsize, Kl, 6)
    sigma = sigma.reshape(batchsize, Kl, 6)
    
    k_uns = k.unsqueeze(1).unsqueeze(2).expand(-1, -1, 6)
    mu_sel = torch.gather(mu, 1, k_uns).squeeze(1)
    sigma_sel = torch.gather(sigma, 1, k_uns).squeeze(1)
    
    # Sample Lattice
    L_final = torch.normal(mean=mu_sel, std=sigma_sel * np.sqrt(temperature))
    
    # Scale length by num_atoms^(1/3)
    # M = mult_table[G, W]
    # num_atoms = sum(M)
    mult_table_gpu = mult_table.to(device)
    G_idx = (G - 1).unsqueeze(1).expand(-1, n_max)
    M = mult_table_gpu[G_idx, W]
    num_atoms = torch.sum(M, dim=1).float()
    
    length, angle = torch.split(L_final, 3, dim=-1)
    length = length * torch.pow(num_atoms.unsqueeze(1), 1.0/3.0)
    angle = angle * (180.0 / np.pi)
    
    L_final = torch.cat([length, angle], dim=-1)
    
    # Symmetrize Lattice
    # We call external function
    L_final = symmetrize_lattice(G, L_final)
    
    XYZ_final = torch.stack([X, Y, Z], dim=-1)
    
    return XYZ_final, A, W, M, L_final

def make_update_lattice(transformer, params, atom_types, Kl, top_p, temperature):
    """
    Returns a function to update lattice only.
    """
    def update_lattice(G, XYZ, A, W):
        device = next(transformer.parameters()).device
        batchsize = XYZ.shape[0]
        
        # Inference once
        output = inference(transformer, G, W, A, XYZ[:,:,0], XYZ[:,:,1], XYZ[:,:,2])
        
        num_sites = torch.sum(A != 0, dim=1)
        
        # Get L params
        # Similar logic to sample_crystal
        h_al = output[:, :, 1::5] # Wait, h_al is every 5th element starting at 1?
        # In sample_crystal logic:
        # h_al = output[:, 5*i+1]
        # We need h_al for all i? No, just the one at num_sites.
        
        # Actually, let's just grab the specific step output
        # We need to construct indices (Batch, 1, Output)
        idx = (5 * num_sites + 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, output.shape[2])
        idx = torch.clamp(idx, max=output.shape[1]-1)
        
        h_selected = torch.gather(output, 1, idx).squeeze(1)
        
        l_params = h_selected[:, atom_types : atom_types + Kl + 12 * Kl]
        
        l_logit, mu, sigma = torch.split(l_params, [Kl, 6*Kl, 6*Kl], dim=-1)
        
        k = sample_top_p(l_logit, top_p, temperature)
        
        mu = mu.reshape(batchsize, Kl, 6)
        sigma = sigma.reshape(batchsize, Kl, 6)
        
        k_uns = k.unsqueeze(1).unsqueeze(2).expand(-1, -1, 6)
        mu_sel = torch.gather(mu, 1, k_uns).squeeze(1)
        sigma_sel = torch.gather(sigma, 1, k_uns).squeeze(1)
        
        L_final = torch.normal(mean=mu_sel, std=sigma_sel * np.sqrt(temperature))
        
        mult_table_gpu = mult_table.to(device)
        G_idx = (G - 1).unsqueeze(1).expand(-1, W.shape[1])
        M = mult_table_gpu[G_idx, W]
        num_atoms = torch.sum(M, dim=1).float()
        
        length, angle = torch.split(L_final, 3, dim=-1)
        length = length * torch.pow(num_atoms.unsqueeze(1), 1.0/3.0)
        angle = angle * (180.0 / np.pi)
        L_final = torch.cat([length, angle], dim=-1)
        
        L_final = symmetrize_lattice(G, L_final)
        
        return L_final
        
    return update_lattice