import os
import sys
import yaml
import pickle
import torch
import numpy as np
import jax
import jax.numpy as jnp

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CRYSTALFORMER_DIR = os.path.join(CURRENT_DIR, "CrystalFormer")
sys.path.append(CRYSTALFORMER_DIR)

from crystalformer.src.transformer import make_transformer

def load_jax_weights(path):
    print(f"Loading JAX weights from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    params = data['params'] if 'params' in data else data
    return params

def jax_to_torch(param):
    return torch.tensor(np.array(param))

def get_param(params, key, subkey):
    """Safely get param from flat dictionary."""
    if key not in params:
        # Try finding key with/without suffix logic if needed, but flat structure usually precise
        raise KeyError(f"Key '{key}' not found in params.")
    
    val = params[key]
    if subkey not in val:
        raise KeyError(f"Subkey '{subkey}' not found in params['{key}']. Available: {list(val.keys())}")
    
    return jax_to_torch(val[subkey])

def convert_weights(jax_params, torch_model):
    print("Converting weights (Flat Structure)...")
    state_dict = torch_model.state_dict()
    
    # --- 1. Embeddings ---
    # Keys: 'g_embeddings', 'w_embeddings', 'a_embeddings'
    state_dict['g_embeddings.weight'] = get_param(jax_params, 'g_embeddings', 'embeddings')
    state_dict['w_embeddings.weight'] = get_param(jax_params, 'w_embeddings', 'embeddings')
    state_dict['a_embeddings.weight'] = get_param(jax_params, 'a_embeddings', 'embeddings')
    
    # h0 / w_params
    if 'w_params' in jax_params:
        print("Found w_params (Table Lookup h0)")
        state_dict['w_params'] = get_param(jax_params, 'w_params', 'w_params')
        linear_offset = 0
    elif 'sequential/linear' in jax_params:
        print("Found h0 MLP")
        # Handle MLP conversion if needed
        # Assuming table lookup for this checkpoint based on logs
        linear_offset = 2 # If MLP uses 2 linears
        pass 
    else:
        # Default assume table lookup if keys missing
        linear_offset = 0

    # --- 2. Projections (Linear Layers) ---
    # Standard: linear (hW), linear_1 (hA), linear_2,3,4 (hXYZ)
    # Note: If h0 MLP exists, these indices shift.
    
    def get_linear(idx, torch_name):
        suffix = f"_{idx}" if idx > 0 else ""
        key = f"linear{suffix}"
        w = get_param(jax_params, key, 'w')
        b = get_param(jax_params, key, 'b')
        state_dict[f'{torch_name}.weight'] = w.t()
        state_dict[f'{torch_name}.bias'] = b

    # fc_hW (linear_0)
    get_linear(linear_offset + 0, 'fc_hW')
    
    # fc_hA (linear_1)
    get_linear(linear_offset + 1, 'fc_hA')
    
    # fc_hXYZ (Average 2, 3, 4)
    def get_raw_linear(idx):
        suffix = f"_{idx}" if idx > 0 else ""
        key = f"linear{suffix}"
        w = get_param(jax_params, key, 'w')
        b = get_param(jax_params, key, 'b')
        return w, b

    w2, b2 = get_raw_linear(linear_offset + 2)
    w3, b3 = get_raw_linear(linear_offset + 3)
    w4, b4 = get_raw_linear(linear_offset + 4)
    
    state_dict['fc_hXYZ.weight'] = ((w2+w3+w4)/3.0).t()
    state_dict['fc_hXYZ.bias'] = (b2+b3+b4)/3.0
    
    # Current Linear Index Counter
    # We used 0, 1, 2, 3, 4. Next is 5.
    curr_linear_idx = linear_offset + 5

    # --- 3. Transformer Layers ---
    num_layers = len(torch_model.layers)
    
    for i in range(num_layers):
        # Layer Norms
        # 2 per layer. Indices: 0,1 for layer 0. 2,3 for layer 1...
        ln1_idx = 2 * i
        ln2_idx = 2 * i + 1
        
        ln1_suffix = f"_{ln1_idx}" if ln1_idx > 0 else ""
        ln2_suffix = f"_{ln2_idx}" 
        
        # LN1
        key = f"layer_norm{ln1_suffix}"
        state_dict[f'layers.{i}.ln1.weight'] = get_param(jax_params, key, 'scale')
        state_dict[f'layers.{i}.ln1.bias'] = get_param(jax_params, key, 'offset')
        
        # LN2
        key = f"layer_norm{ln2_suffix}"
        state_dict[f'layers.{i}.ln2.weight'] = get_param(jax_params, key, 'scale')
        state_dict[f'layers.{i}.ln2.bias'] = get_param(jax_params, key, 'offset')
        
        # MLP
        # 2 Linears per layer.
        l1_idx = curr_linear_idx
        l2_idx = curr_linear_idx + 1
        curr_linear_idx += 2
        
        get_linear(l1_idx, f'layers.{i}.mlp.0')
        get_linear(l2_idx, f'layers.{i}.mlp.2')
        
        # ATTENTION
        # Keys: multi_head_attention_{i}/query, /key, /value, /linear
        attn_suffix = f"_{i}" if i > 0 else ""
        base_attn_key = f"multi_head_attention{attn_suffix}"
        
        # Query
        w_q = get_param(jax_params, f"{base_attn_key}/query", 'w')
        b_q = get_param(jax_params, f"{base_attn_key}/query", 'b')
        
        # Key
        w_k = get_param(jax_params, f"{base_attn_key}/key", 'w')
        b_k = get_param(jax_params, f"{base_attn_key}/key", 'b')
        
        # Value
        w_v = get_param(jax_params, f"{base_attn_key}/value", 'w')
        b_v = get_param(jax_params, f"{base_attn_key}/value", 'b')
        
        # PyTorch MHA packs Q,K,V
        # JAX shape: (Embed, Heads, Dim) -> Flatten -> (Embed, Embed)
        # PyTorch shape: (Embed, Embed) (Transposed)
        
        def process_attn_w(w):
            return w.reshape(w.shape[0], -1).t()
            
        pt_w_q = process_attn_w(w_q.numpy()) # Convert to numpy for reshape if needed or check torch reshape
        pt_w_k = process_attn_w(w_k.numpy())
        pt_w_v = process_attn_w(w_v.numpy())
        
        # Reshape bias: (Heads, Dim) -> (Embed,)
        pt_b_q = b_q.reshape(-1)
        pt_b_k = b_k.reshape(-1)
        pt_b_v = b_v.reshape(-1)
        
        state_dict[f'layers.{i}.attn.in_proj_weight'] = torch.cat([jax_to_torch(pt_w_q), jax_to_torch(pt_w_k), jax_to_torch(pt_w_v)], dim=0)
        state_dict[f'layers.{i}.attn.in_proj_bias'] = torch.cat([pt_b_q, pt_b_k, pt_b_v], dim=0)
        
        # Output Linear
        # Key: .../linear
        w_o = get_param(jax_params, f"{base_attn_key}/linear", 'w')
        b_o = get_param(jax_params, f"{base_attn_key}/linear", 'b')
        
        state_dict[f'layers.{i}.attn.out_proj.weight'] = w_o.t()
        state_dict[f'layers.{i}.attn.out_proj.bias'] = b_o

    # --- 4. Final Layers ---
    # Final LN
    final_ln_idx = 2 * num_layers
    key = f"layer_norm_{final_ln_idx}"
    state_dict['final_norm.weight'] = get_param(jax_params, key, 'scale')
    state_dict['final_norm.bias'] = get_param(jax_params, key, 'offset')
    
    # Final Linear
    final_lin_idx = curr_linear_idx
    get_linear(final_lin_idx, 'output_proj')

    print("Load state dict into model...")
    torch_model.load_state_dict(state_dict)
    print("Success!")
    return torch_model

def main():
    config_path = os.path.join(CURRENT_DIR, "pretrained_model", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model = make_transformer(
        key=None,
        Nf=config['Nf'], Kx=config['Kx'], Kl=config['Kl'], n_max=config['n_max'],
        h0_size=config['h0_size'], num_layers=config['transformer_layers'],
        num_heads=config['num_heads'], key_size=config['key_size'],
        model_size=config['model_size'], embed_size=config['embed_size'],
        atom_types=config['atom_types'], wyck_types=config['wyck_types'],
        dropout_rate=config['dropout_rate']
    )
    
    jax_path = os.path.join(CURRENT_DIR, "pretrained_model", "epoch_005500.pkl")
    try:
        jax_params = load_jax_weights(jax_path)
        model = convert_weights(jax_params, model)
        
        out_path = os.path.join(CURRENT_DIR, "pretrained_model", "epoch_005500.pt")
        torch.save(model.state_dict(), out_path)
        print(f"Saved PyTorch weights to {out_path}")
        
    except Exception as e:
        print(f"\nCONVERSION FAILED: {e}")
        # Debug print keys if failed
        # print(list(jax_params.keys())[:20])

if __name__ == "__main__":
    main()
