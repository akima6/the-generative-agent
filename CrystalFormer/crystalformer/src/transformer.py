import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# We assume these tables are available from the rewritten wyckoff.py
from crystalformer.src.wyckoff import wmax_table, dof0_table

class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Block.
    Norm -> Attention -> Add -> Norm -> MLP -> Add
    """
    def __init__(self, num_heads, key_size, model_size, dropout_rate=0.1, widening_factor=4):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
        
        self.ln1 = nn.LayerNorm(model_size)
        self.ln2 = nn.LayerNorm(model_size)

        self.attn = nn.MultiheadAttention(embed_dim=model_size, 
                                          num_heads=num_heads, 
                                          dropout=dropout_rate, 
                                          batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(model_size, widening_factor * model_size),
            nn.GELU(),
            nn.Linear(widening_factor * model_size, model_size)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask=None, is_train=True):
        # x: (Batch, SeqLen, ModelSize)
        h_norm = self.ln1(x)
        
        # attn_mask should be (SeqLen, SeqLen) or (Batch*NumHeads, SeqLen, SeqLen)
        # We assume standard causal mask passed from main model
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, attn_mask=mask, need_weights=False)
        
        if is_train:
            attn_out = self.dropout(attn_out)
        
        x = x + attn_out

        h_norm = self.ln2(x)
        mlp_out = self.mlp(h_norm)
        
        if is_train:
            mlp_out = self.dropout(mlp_out)
            
        x = x + mlp_out
        
        return x

class CrystalTransformer(nn.Module):
    def __init__(self, Nf, Kx, Kl, n_max, h0_size, num_layers, num_heads, key_size, 
                 model_size, embed_size, atom_types, wyck_types, dropout_rate, 
                 attn_dropout=0.1, widening_factor=4, sigmamin=1e-3):
        super().__init__()
        
        self.Nf = Nf
        self.Kx = Kx
        self.Kl = Kl
        self.n_max = n_max
        self.h0_size = h0_size
        self.model_size = model_size
        self.embed_size = embed_size
        self.atom_types = atom_types
        self.wyck_types = wyck_types
        self.sigmamin = sigmamin
        
        self.coord_types = 3 * Kx
        self.lattice_types = Kl + 2 * 6 * Kl
        self.output_size = max(atom_types + self.lattice_types, self.coord_types, wyck_types)

        # Embeddings
        self.g_embeddings = nn.Embedding(230, embed_size)
        self.w_embeddings = nn.Embedding(wyck_types, embed_size)
        self.a_embeddings = nn.Embedding(atom_types, embed_size)
        
        if h0_size > 0:
            self.h0_mlp = nn.Sequential(
                nn.Linear(embed_size, h0_size),
                nn.GELU(),
                nn.Linear(h0_size, wyck_types)
            )
        else:
            self.w_params = nn.Parameter(torch.randn(230, wyck_types) * 0.01)

        # Projections
        # hW input: G(emb) + W(emb) + M(1)
        self.fc_hW = nn.Linear(2 * embed_size + 1, model_size)
        # hA input: G(emb) + A(emb)
        self.fc_hA = nn.Linear(2 * embed_size, model_size)
        # hXYZ input: G(emb) + Fourier(2*Nf)
        self.fc_hXYZ = nn.Linear(embed_size + 2 * Nf, model_size)

        self.layers = nn.ModuleList([
            TransformerBlock(num_heads, key_size, model_size, attn_dropout, widening_factor)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(model_size)
        self.output_proj = nn.Linear(model_size, self.output_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                torch.nn.init.trunc_normal_(m.weight, std=0.01)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def renormalize_coord(self, h_x):
        # h_x: (Batch, n, output_size)
        # Only process first coord_types dims
        relevant = h_x[..., :self.coord_types]
        x_logit, x_loc, x_kappa = torch.split(relevant, [self.Kx, self.Kx, self.Kx], dim=-1)
        
        x_logit = torch.log_softmax(x_logit, dim=-1)
        x_kappa = F.softplus(x_kappa)
        
        padding = torch.zeros(h_x.shape[:-1] + (self.output_size - self.coord_types,), device=h_x.device)
        return torch.cat([x_logit, x_loc, x_kappa, padding], dim=-1)

    def forward(self, G, XYZ, A, W, M, is_train=True):
        """
        Supports Batched Input.
        G: (Batch,)
        XYZ: (Batch, n, 3)
        A: (Batch, n)
        W: (Batch, n)
        M: (Batch, n)
        """
        device = self.g_embeddings.weight.device
        
        # Ensure inputs are tensors
        if not isinstance(G, torch.Tensor): G = torch.tensor(G, device=device)
        XYZ = XYZ.to(device)
        A = A.to(device)
        W = W.to(device)
        M = M.to(device)

        # Handle Batch Dimension logic
        is_batched = XYZ.ndim == 3
        if not is_batched:
            # Add batch dim for internal processing
            G = G.reshape(1)
            XYZ = XYZ.unsqueeze(0)
            A = A.unsqueeze(0)
            W = W.unsqueeze(0)
            M = M.unsqueeze(0)

        batch_size = XYZ.shape[0]
        n_atoms = XYZ.shape[1]
        
        G_idx = G.long() - 1
        g_emb = self.g_embeddings(G_idx) # (Batch, emb)
        
        # --- 1. Compute h0 (First prediction) ---
        wmax_tensor = torch.tensor(wmax_table, device=device) # (230,)
        w_max_val = wmax_tensor[G_idx] # (Batch,)

        if self.h0_size > 0:
            w_logit_0 = self.h0_mlp(g_emb) # (Batch, wyck_types)
        else:
            w_logit_0 = self.w_params[G_idx] 

        # Masking h0
        w_range = torch.arange(self.wyck_types, device=device).unsqueeze(0) # (1, wyck)
        # Broadcast w_max_val to (Batch, 1)
        w_mask_0 = (w_range > 0) & (w_range <= w_max_val.unsqueeze(1))
        
        w_logit_0 = torch.where(w_mask_0, w_logit_0, w_logit_0 - 1e10)
        w_logit_0 = torch.log_softmax(w_logit_0, dim=1)
        
        # h0 vector
        h0 = torch.cat([
            w_logit_0.unsqueeze(1), # (Batch, 1, wyck)
            torch.zeros((batch_size, 1, self.output_size - self.wyck_types), device=device)
        ], dim=-1) # (Batch, 1, output)
        
        if n_atoms == 0:
            return h0 if is_batched else h0.squeeze(0)

        # --- 2. Sequence Embedding ---
        g_emb_exp = g_emb.unsqueeze(1).expand(-1, n_atoms, -1)
        
        # hW
        w_emb = self.w_embeddings(W.long()) # (Batch, n, emb)
        hW_in = torch.cat([g_emb_exp, w_emb, M.float().unsqueeze(-1)], dim=-1)
        hW = self.fc_hW(hW_in) # (Batch, n, model)

        # hA
        a_emb = self.a_embeddings(A.long())
        hA_in = torch.cat([g_emb_exp, a_emb], dim=-1)
        hA = self.fc_hA(hA_in)

        # hXYZ (Fourier)
        X, Y, Z = XYZ[:,:,0], XYZ[:,:,1], XYZ[:,:,2] # (Batch, n)
        
        def fourier_encode(coords, Nf):
            # coords: (Batch, n)
            freqs = torch.arange(1, Nf+1, device=device).float()
            # args: (Batch, n, 1) * (1, 1, Nf) -> (Batch, n, Nf)
            args = 2 * math.pi * coords.unsqueeze(-1) * freqs.view(1, 1, -1)
            return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        hX_f = fourier_encode(X, self.Nf)
        hY_f = fourier_encode(Y, self.Nf)
        hZ_f = fourier_encode(Z, self.Nf)
        
        hX = self.fc_hXYZ(torch.cat([g_emb_exp, hX_f], dim=-1))
        hY = self.fc_hXYZ(torch.cat([g_emb_exp, hY_f], dim=-1))
        hZ = self.fc_hXYZ(torch.cat([g_emb_exp, hZ_f], dim=-1))
        
        # --- 3. Interleave ---
        # Stack: (Batch, n, 5, model)
        h = torch.stack([hW, hA, hX, hY, hZ], dim=2)
        h = h.reshape(batch_size, 5 * n_atoms, self.model_size)
        
        # --- 4. Transformer Attention ---
        seq_len = 5 * n_atoms
        # Causal Mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        float_mask = torch.zeros(seq_len, seq_len, device=device)
        float_mask = float_mask.masked_fill(mask, float('-inf'))
        
        for layer in self.layers:
            h = layer(h, mask=float_mask, is_train=is_train)
            
        h = self.final_norm(h)
        h = self.output_proj(h) # (Batch, 5*n, output)
        
        # --- 5. Unpack and Process Outputs ---
        h = h.reshape(batch_size, n_atoms, 5, -1)
        
        h_al = h[:, :, 0, :]
        h_x = h[:, :, 1, :]
        h_y = h[:, :, 2, :]
        h_z = h[:, :, 3, :]
        w_logit_seq = h[:, :, 4, :]
        
        # Renormalize Coords
        h_x = self.renormalize_coord(h_x)
        h_y = self.renormalize_coord(h_y)
        h_z = self.renormalize_coord(h_z)
        
        # Logits
        a_logit = h_al[..., :self.atom_types]
        w_logit = w_logit_seq[..., :self.wyck_types]
        
        # --- Masking Logic (Batched) ---
        
        # (1) W constraints: W_0 <= W_1 ...
        # W: (Batch, n)
        # w_range: (1, 1, wyck-1)
        w_range = torch.arange(1, self.wyck_types, device=device).view(1, 1, -1)
        W_uns = W.unsqueeze(-1) # (Batch, n, 1)
        
        mask_less_equal = w_range < W_uns
        mask_less = w_range <= W_uns
        
        # dof0 lookup
        dof0_gpu = torch.tensor(dof0_table, device=device) # (230, wyck)
        G_exp = G_idx.unsqueeze(1).expand(-1, n_atoms)
        is_dof0 = dof0_gpu[G_exp, W.long()] # (Batch, n)
        
        w_mask_1 = torch.where(is_dof0.unsqueeze(-1), mask_less, mask_less_equal)
        # Pad column 0
        pad_col = torch.zeros(batch_size, n_atoms, 1, device=device, dtype=torch.bool)
        w_mask_1 = torch.cat([pad_col, w_mask_1], dim=-1)
        
        w_logit = w_logit - torch.where(w_mask_1, torch.tensor(1e10, device=device), torch.tensor(0.0, device=device))
        w_logit = torch.log_softmax(w_logit, dim=-1)
        
        # (2) Pad Atom Logic - FIXED (Out-of-place update)
        # If W==0, heavily prefer pad (idx 0) in w_logit
        # Logic: w_logit[..., 0] += 1e10 if W==0
        idx_0_mask = torch.zeros_like(w_logit, dtype=torch.bool)
        idx_0_mask[..., 0] = True
        w_is_0_mask = (W == 0).unsqueeze(-1).expand_as(w_logit)
        
        w_logit = torch.where(idx_0_mask & w_is_0_mask, w_logit + 1e10, w_logit)
        w_logit = torch.log_softmax(w_logit, dim=-1)
        
        # (3) w_max constraint
        w_indices = torch.arange(self.wyck_types, device=device).view(1, 1, -1)
        w_mask_3 = w_indices <= w_max_val.view(-1, 1, 1)
        w_logit = torch.where(w_mask_3, w_logit, w_logit - 1e10)
        w_logit = torch.log_softmax(w_logit, dim=-1)
        
        # (4) Atom Masking
        w_gt_0 = (W > 0).unsqueeze(-1)
        w_eq_0 = (W == 0).unsqueeze(-1)
        
        a_mask = torch.cat([
            w_gt_0,
            w_eq_0.repeat(1, 1, self.atom_types - 1)
        ], dim=-1)
        
        a_logit = a_logit - torch.where(a_mask, torch.tensor(1e10, device=device), torch.tensor(0.0, device=device))
        a_logit = torch.log_softmax(a_logit, dim=-1)
        
        # Re-assemble w_logit full
        w_logit_final = torch.cat([
            w_logit,
            torch.zeros(batch_size, n_atoms, self.output_size - self.wyck_types, device=device)
        ], dim=-1)
        
        # Lattice Parts (Batch, n, output)
        l_part = h_al[..., self.atom_types : self.atom_types + self.lattice_types]
        l_logit, mu, sigma = torch.split(l_part, [self.Kl, 6*self.Kl, 6*self.Kl], dim=-1)
        
        l_logit = torch.log_softmax(l_logit, dim=-1)
        sigma = F.softplus(sigma) + self.sigmamin
        
        h_al_final = torch.cat([
            a_logit,
            l_logit,
            mu,
            sigma,
            torch.zeros(batch_size, n_atoms, self.output_size - self.atom_types - self.lattice_types, device=device)
        ], dim=-1)
        
        # Final Stack
        h_final = torch.stack([
            h_al_final,
            h_x,
            h_y,
            h_z,
            w_logit_final
        ], dim=2) # (Batch, n, 5, output)
        
        h_final = h_final.reshape(batch_size, 5*n_atoms, self.output_size)
        
        # Concatenate h0 at seq dim
        result = torch.cat([h0, h_final], dim=1) # (Batch, 5*n+1, output)
        
        if not is_batched:
            return result.squeeze(0)
            
        return result

def make_transformer(key, Nf, Kx, Kl, n_max, h0_size, num_layers, num_heads, key_size, 
                     model_size, embed_size, atom_types, wyck_types, dropout_rate, 
                     attn_dropout=0.1, widening_factor=4, sigmamin=1e-3):
    return CrystalTransformer(
        Nf=Nf, Kx=Kx, Kl=Kl, n_max=n_max, h0_size=h0_size, 
        num_layers=num_layers, num_heads=num_heads, key_size=key_size, 
        model_size=model_size, embed_size=embed_size, 
        atom_types=atom_types, wyck_types=wyck_types, 
        dropout_rate=dropout_rate, attn_dropout=attn_dropout, 
        widening_factor=widening_factor, sigmamin=sigmamin
    )
