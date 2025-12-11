import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# We assume these tables will be available from the rewritten wyckoff.py
# If wyckoff.py is not yet rewritten, these imports might fail until Phase 1 is complete.
from crystalformer.src.wyckoff import wmax_table, dof0_table

class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Block to match Haiku's architecture:
    Norm -> Attention -> Add -> Norm -> MLP -> Add
    """
    def __init__(self, num_heads, key_size, model_size, dropout_rate=0.1, widening_factor=4):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
        
        # Layer Norms
        self.ln1 = nn.LayerNorm(model_size)
        self.ln2 = nn.LayerNorm(model_size)

        # Multi-Head Attention
        # Note: Haiku's KeySize usually implies per-head dimension. 
        # PyTorch MHA usually takes embed_dim and num_heads.
        # We assume model_size is the embedding dimension.
        self.attn = nn.MultiheadAttention(embed_dim=model_size, 
                                          num_heads=num_heads, 
                                          dropout=dropout_rate, 
                                          batch_first=True)
        
        # MLP Block
        self.mlp = nn.Sequential(
            nn.Linear(model_size, widening_factor * model_size),
            nn.GELU(),
            nn.Linear(widening_factor * model_size, model_size)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialization (Truncated Normal 0.01 to match original)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask=None, is_train=True):
        # x: (Batch, SeqLen, ModelSize)
        
        # Sub-layer 1: Attention
        h_norm = self.ln1(x)
        
        # PyTorch MHA expects mask: (N*num_heads, L, S) or (L, S).
        # The original passed a causal mask (1, 5*n, 5*n).
        # We need to ensure the mask is boolean or -inf for MHA.
        # If mask is 0/1 (1 for keep, 0 for mask), we need to invert for PyTorch usually
        # but PyTorch `attn_mask` accepts float mask directly (0 for keep, -inf for mask)
        # or boolean (True for mask, False for keep).
        # Let's handle the mask passed from the main model.
        
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, attn_mask=mask, need_weights=False)
        
        if is_train:
            attn_out = self.dropout(attn_out)
        
        x = x + attn_out

        # Sub-layer 2: MLP
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
        
        # h0 Logic
        if h0_size > 0:
            self.h0_mlp = nn.Sequential(
                nn.Linear(embed_size, h0_size),
                nn.GELU(),
                nn.Linear(h0_size, wyck_types)
            )
        else:
            self.w_params = nn.Parameter(torch.randn(230, wyck_types) * 0.01)

        # Projections to model_size
        # Input to these are concatenations of embeddings and features
        # hW input: G_emb (emb) + W_emb (emb) + M (1) = 2*emb + 1
        self.fc_hW = nn.Linear(2 * embed_size + 1, model_size)
        
        # hA input: G_emb (emb) + A_emb (emb) = 2*emb
        self.fc_hA = nn.Linear(2 * embed_size, model_size)
        
        # hX, hY, hZ input: G_emb (emb) + Fourier (2*Nf)
        self.fc_hXYZ = nn.Linear(embed_size + 2 * Nf, model_size)

        # Position embeddings (from commented out code in original, but unused? 
        # Original: "h = h + positional_embeddings[:5*n, :]". 
        # The line was commented out in the source provided. I will skip it to preserve "EXACT model behavior".)

        # Transformer Layers
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
        # h_x: (n, output_size)
        # We only care about the first coord_types dimensions
        n = h_x.shape[0]
        
        # Slice
        relevant = h_x[:, :self.coord_types]
        x_logit, x_loc, x_kappa = torch.split(relevant, [self.Kx, self.Kx, self.Kx], dim=-1)
        
        # LogSoftmax for logits (equivalent to x - logsumexp(x))
        x_logit = torch.log_softmax(x_logit, dim=1)
        
        # Softplus for kappa
        x_kappa = F.softplus(x_kappa)
        
        # Concatenate back
        # Padding zeros for the rest of output_size
        padding = torch.zeros((n, self.output_size - self.coord_types), device=h_x.device)
        
        h_x_new = torch.cat([x_logit, x_loc, x_kappa, padding], dim=-1)
        return h_x_new

    def forward(self, G, XYZ, A, W, M, is_train=True):
        """
        Args:
            G: (Scalar) Space group ID (1-230). In Torch, usually passed as tensor (1,) or scalar.
            XYZ: (n, 3) fractional coordinates
            A: (n, )  element type 
            W: (n, )  wyckoff position index
            M: (n, )  multiplicities
            is_train: bool 
        """
        
        # Ensure inputs are tensors and on correct device
        device = self.g_embeddings.weight.device
        if not isinstance(G, torch.Tensor): G = torch.tensor(G, device=device)
        XYZ = XYZ.to(device)
        A = A.to(device)
        W = W.to(device)
        M = M.to(device)

        assert XYZ.ndim == 2
        assert XYZ.shape[0] == A.shape[0]
        assert XYZ.shape[1] == 3
        
        n = XYZ.shape[0]
        
        # Adjust G index (0-229)
        G_idx = G.long() - 1
        
        # 1. Initial h0 (Wyckoff logits for first atom)
        g_emb = self.g_embeddings(G_idx) # (embed_size,)
        
        # Convert wmax_table to tensor
        # wmax_table is imported. Assuming it's numpy or list.
        w_max_val = torch.tensor(wmax_table[G_idx.item()], device=device)
        
        if self.h0_size > 0:
            w_logit = self.h0_mlp(g_emb) # (wyck_types,)
        else:
            w_logit = self.w_params[G_idx] # (wyck_types,)

        # Masking h0
        # (1) first atom not pad atom (index 0)
        # (2) mask unavailable positions
        w_range = torch.arange(self.wyck_types, device=device)
        w_mask = (w_range > 0) & (w_range <= w_max_val)
        
        # Apply mask: keep where true, else -1e10
        w_logit = torch.where(w_mask, w_logit, w_logit - 1e10)
        
        # Normalization
        w_logit = torch.log_softmax(w_logit, dim=0)
        
        h0 = torch.cat([w_logit.unsqueeze(0), 
                        torch.zeros((1, self.output_size - self.wyck_types), device=device)], dim=-1) # (1, output_size)
        
        if n == 0:
            return h0

        # 2. Sequence Construction
        # Embeddings broadcasted
        g_emb_expanded = g_emb.unsqueeze(0).expand(n, -1) # (n, embed_size)
        
        # hW
        w_emb = self.w_embeddings(W.long()) # (n, embed_size)
        hW_in = torch.cat([g_emb_expanded, w_emb, M.float().unsqueeze(1)], dim=1) # (n, 2*emb + 1)
        hW = self.fc_hW(hW_in) # (n, model_size)

        # hA
        a_emb = self.a_embeddings(A.long()) # (n, embed_size)
        hA_in = torch.cat([g_emb_expanded, a_emb], dim=1) # (n, 2*emb)
        hA = self.fc_hA(hA_in) # (n, model_size)
        
        # hX, hY, hZ (Fourier features)
        X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2] # (n,)
        
        def fourier_encode(coords, Nf):
            # coords: (n,)
            # output: (n, 2*Nf)
            freqs = torch.arange(1, Nf+1, device=device).float() # (Nf,)
            args = 2 * math.pi * coords.unsqueeze(1) * freqs.unsqueeze(0) # (n, Nf)
            return torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        
        hX_f = fourier_encode(X, self.Nf)
        hY_f = fourier_encode(Y, self.Nf)
        hZ_f = fourier_encode(Z, self.Nf)
        
        hX = self.fc_hXYZ(torch.cat([g_emb_expanded, hX_f], dim=1))
        hY = self.fc_hXYZ(torch.cat([g_emb_expanded, hY_f], dim=1))
        hZ = self.fc_hXYZ(torch.cat([g_emb_expanded, hZ_f], dim=1))
        
        # Interleave: [hW, hA, hX, hY, hZ] per atom
        # Stack dim 1 -> (n, 5, model_size)
        h = torch.stack([hW, hA, hX, hY, hZ], dim=1)
        h = h.reshape(5 * n, self.model_size) # (5*n, model_size)
        
        # Prepare for Transformer
        # Batch dimension = 1
        h = h.unsqueeze(0) # (1, 5*n, model_size)
        
        # Mask for attention (Causal / Tril)
        # Original: jnp.tril(jnp.ones((1, 5*n, 5*n)))
        # PyTorch attn_mask: (L, L) or (Batch*Heads, L, L). 
        # Using float mask: 0.0 for include, -inf for exclude.
        # Or boolean: True to key_padding_mask (ignore), but attn_mask meaning depends on version.
        # We use standard causal mask logic:
        # We want position i to attend to 0..i.
        seq_len = 5 * n
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        # Invert for PyTorch floating point mask: 0 for keep, -inf for mask
        attn_mask = torch.zeros((seq_len, seq_len), device=device)
        attn_mask = attn_mask.masked_fill(causal_mask == 0, float('-inf'))
        
        # 3. Transformer Loop
        for layer in self.layers:
            h = layer(h, mask=attn_mask, is_train=is_train)
            
        h = self.final_norm(h)
        h = self.output_proj(h) # (1, 5*n, output_size)
        
        h = h.squeeze(0).reshape(n, 5, -1) # (n, 5, output_size)
        
        # Unpack
        h_al = h[:, 0, :]
        h_x = h[:, 1, :]
        h_y = h[:, 2, :]
        h_z = h[:, 3, :]
        w_logit_seq = h[:, 4, :] # Rename to avoid confusion with h0 w_logit
        
        # 4. Post-processing & Renormalization
        h_x = self.renormalize_coord(h_x)
        h_y = self.renormalize_coord(h_y)
        h_z = self.renormalize_coord(h_z)
        
        # Extract logits
        a_logit = h_al[:, :self.atom_types]
        w_logit_step = w_logit_seq[:, :self.wyck_types]
        
        # --- Masking Logic ---
        # (1) W_0 <= W_1 <= W_2 constraints
        # W is (n,). 
        # w_mask_less_equal: (n, wyck_types-1)
        w_idx = torch.arange(1, self.wyck_types, device=device).unsqueeze(0) # (1, wyck-1)
        W_uns = W.unsqueeze(1) # (n, 1)
        
        mask_less_equal = w_idx < W_uns
        mask_less = w_idx <= W_uns
        
        # dof0_table: (230, wyck_types) -> boolean
        # Select for current G and W
        # dof0_table is (230, max_wyck).
        # We need dof0_table[G-1, W]. W is (n,).
        # Need to cast table to tensor first if not already
        dof0_tensor = torch.tensor(dof0_table, device=device) # (230, wyck_types)
        is_dof0 = dof0_tensor[G_idx, W.long()] # (n,)
        
        w_mask_1 = torch.where(is_dof0.unsqueeze(1), mask_less, mask_less_equal) # (n, wyck-1)
        
        # Prepend zeros column (pad for index 0)
        w_mask_1 = torch.cat([torch.zeros((n, 1), device=device, dtype=torch.bool), w_mask_1], dim=1) # (n, wyck)
        
        w_logit_step = w_logit_step - torch.where(w_mask_1, torch.tensor(1e10, device=device), torch.tensor(0.0, device=device))
        w_logit_step = torch.log_softmax(w_logit_step, dim=1)
        
        # (2) Pad atom logic (if W==0, enhance prob of pad)
        # mask = 1 where we want to place pad atoms (if prev W==0)
        pad_cond = (W == 0).unsqueeze(1) # (n, 1)
        w_mask_2 = torch.cat([
            torch.where(pad_cond, torch.ones((n, 1), device=device), torch.zeros((n, 1), device=device)),
            torch.zeros((n, self.wyck_types - 1), device=device)
        ], dim=1).bool()
        
        # If mask is True, we set logit to 1e10 ?? 
        # Original: jnp.where(w_mask, 1e10, w_logit). 
        # Wait, original says: "enhance the probability of pad atoms".
        # If w_mask is 1 (pad atom pos), set to 1e10 (very high prob).
        w_logit_step = torch.where(w_mask_2, torch.tensor(1e10, device=device), w_logit_step)
        w_logit_step = torch.log_softmax(w_logit_step, dim=1)
        
        # (3) Mask out unavailable positions > w_max
        w_indices = torch.arange(self.wyck_types, device=device).unsqueeze(0) # (1, wyck)
        w_mask_3 = w_indices <= w_max_val
        w_logit_step = torch.where(w_mask_3, w_logit_step, w_logit_step - 1e10)
        w_logit_step = torch.log_softmax(w_logit_step, dim=1)
        
        # (4) Atom Masking
        # If W > 0, mask out pad atom (type 0).
        # If W == 0, mask out true atoms (type > 0).
        a_mask = torch.cat([
            (W > 0).unsqueeze(1), # Column 0
            (W == 0).unsqueeze(1).repeat(1, self.atom_types - 1) # Columns 1..end
        ], dim=1)
        
        a_logit = a_logit + torch.where(a_mask, torch.tensor(-1e10, device=device), torch.tensor(0.0, device=device))
        a_logit = torch.log_softmax(a_logit, dim=1)
        
        # Re-assemble w_logit
        w_logit_step = torch.cat([w_logit_step, 
                                  torch.zeros((n, self.output_size - self.wyck_types), device=device)], dim=-1)
        
        # Lattice parts
        lattice_part = h_al[:, self.atom_types : self.atom_types + self.lattice_types]
        l_logit, mu, sigma = torch.split(lattice_part, 
                                         [self.Kl, self.Kl * 6, self.Kl * 6], dim=-1) # Assuming sigma is same size as mu?
        # Original logic: split into [Kl, Kl+Kl*6] ?? 
        # JAX: split(..., [Kl, Kl+Kl*6], axis=-1).
        # Indices: 0..Kl (l_logit), Kl..Kl+6*Kl (mu), Kl+6*Kl..end (sigma).
        # So sizes are: Kl, 6*Kl, remaining. 
        # Total lattice_types = Kl + 2*6*Kl = Kl + 12*Kl = 13*Kl.
        # So split sizes are Kl, 6*Kl, 6*Kl. Correct.
        
        l_logit = torch.log_softmax(l_logit, dim=1)
        sigma = F.softplus(sigma) + self.sigmamin
        
        h_al_final = torch.cat([
            a_logit, 
            l_logit, 
            mu, 
            sigma,
            torch.zeros((n, self.output_size - self.atom_types - self.lattice_types), device=device)
        ], dim=-1)
        
        # 5. Final Assembly
        # Stack: (n, 5, output_size)
        # Components: h_al, h_x, h_y, h_z, w_logit
        h_final = torch.stack([
            h_al_final,
            h_x,
            h_y,
            h_z,
            w_logit_step
        ], dim=1)
        
        h_final = h_final.reshape(5 * n, self.output_size)
        
        # Concatenate h0
        result = torch.cat([h0, h_final], dim=0) # (5*n + 1, output_size)
        
        return result

def make_transformer(key, Nf, Kx, Kl, n_max, h0_size, num_layers, num_heads, key_size, 
                     model_size, embed_size, atom_types, wyck_types, dropout_rate, 
                     attn_dropout=0.1, widening_factor=4, sigmamin=1e-3):
    """
    Factory function to match the original interface partially.
    Ignores 'key' as PyTorch is stateful.
    Returns an instance of CrystalTransformer.
    """
    return CrystalTransformer(
        Nf=Nf, Kx=Kx, Kl=Kl, n_max=n_max, h0_size=h0_size, 
        num_layers=num_layers, num_heads=num_heads, key_size=key_size, 
        model_size=model_size, embed_size=embed_size, 
        atom_types=atom_types, wyck_types=wyck_types, 
        dropout_rate=dropout_rate, attn_dropout=attn_dropout, 
        widening_factor=widening_factor, sigmamin=sigmamin
    )