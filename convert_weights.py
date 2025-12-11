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
    """Safely get param from dictionary."""
    if key not in params:
        raise KeyError(f"Key '{key}' not found.")
    if subkey not in params[key]:
        raise KeyError(f"Subkey '{subkey}' not found in '{key}'. Available: {list(params[key].keys())}")
    return jax_to_torch(params[key][subkey])

def convert_weights(jax_params, torch_model):
    print("Converting weights (Flat-Slash Structure)...")
    state_dict = torch_model.state_dict()
    
    # --- 1. EMBEDDINGS (Under '~') ---
    print("  - Converting Embeddings...")
    embeddings_scope = jax_params.get('~')
    if embeddings_scope is None:
        print("Error: Scope '~' not found. Cannot load embeddings.")
        sys.exit(1)

    state_dict['g_embeddings.weight'] = jax_to_torch(embeddings_scope['g_embeddings']['embeddings'])
    state_dict['w_embeddings.weight'] = jax_to_torch(embeddings_scope['w_embeddings']['embeddings'])
    state_dict['a_embeddings.weight'] = jax_to_torch(embeddings_scope['a_embeddings']['embeddings'])

    # --- 2. PROJECTIONS (Linear Layers) ---
    print("  - Converting Projections...")
    
    # Check if w_params exists (h0 size 0) or h0_mlp exists
    # Based on previous logs, we check keys inside '~' or top level 'linear' counts
    if 'w_params' in embeddings_scope:
        print("    Found w_params (Table Lookup)")
        state_dict['w_params'] = jax_to_torch(embeddings_scope['w_params']['w_params'])
        linear_offset = 0
    else:
        print("    Found h0 MLP (Linear 0 & 1)")
        linear_offset = 2

    def get_lin_weights(key_name):
        return get_param(jax_params, key_name, 'w'), get_param(jax_params, key_name, 'b')

    def load_linear_to_torch(jax_idx, torch_name):
        suffix = f"_{jax_idx}" if jax_idx > 0 else ""
        key = f"linear{suffix}"
        w, b = get_lin_weights(key)
        state_dict[f'{torch_name}.weight'] = w.t()
        state_dict[f'{torch_name}.bias'] = b

    # Load h0 MLP if offset is 2
    if linear_offset == 2:
        load_linear_to_torch(0, 'h0_mlp.0')
        load_linear_to_torch(1, 'h0_mlp.2')

    # fc_hW
    load_linear_to_torch(linear_offset + 0, 'fc_hW')
    # fc_hA
    load_linear_to_torch(linear_offset + 1, 'fc_hA')
    
    # fc_hXYZ (Average next 3)
    idx_base = linear_offset + 2
    w4, b4 = get_lin_weights(f"linear_{idx_base}")
    w5, b5 = get_lin_weights(f"linear_{idx_base+1}")
    w6, b6 = get_lin_weights(f"linear_{idx_base+2}")
    
    state_dict['fc_hXYZ.weight'] = ((w4+w5+w6)/3.0).t()
    state_dict['fc_hXYZ.bias'] = (b4+b5+b6)/3.0

    # --- 3. TRANSFORMER BLOCKS ---
    print("  - Converting Transformer Blocks...")
    num_layers = len(torch_model.layers)
    mlp_start_idx = linear_offset + 5 
    
    for i in range(num_layers):
        # Layer Norms
        ln1_key = f"layer_norm_{2*i}" if 2*i > 0 else "layer_norm"
        ln2_key = f"layer_norm_{2*i+1}"
        
        state_dict[f'layers.{i}.ln1.weight'] = get_param(jax_params, ln1_key, 'scale')
        state_dict[f'layers.{i}.ln1.bias'] = get_param(jax_params, ln1_key, 'offset')
        
        state_dict[f'layers.{i}.ln2.weight'] = get_param(jax_params, ln2_key, 'scale')
        state_dict[f'layers.{i}.ln2.bias'] = get_param(jax_params, ln2_key, 'offset')
        
        # MLP
        l1_idx = mlp_start_idx + 2*i
        l2_idx = mlp_start_idx + 2*i + 1
        load_linear_to_torch(l1_idx, f'layers.{i}.mlp.0')
        load_linear_to_torch(l2_idx, f'layers.{i}.mlp.2')
        
        # Attention
        # Key format: "multi_head_attention" or "multi_head_attention_1"
        attn_base = f"multi_head_attention_{i}" if i > 0 else "multi_head_attention"
        
        # NOTE: Keys are like "multi_head_attention/query", NOT nested dicts
        def get_attn_param(submodule, param):
            full_key = f"{attn_base}/{submodule}"
            return get_param(jax_params, full_key, param)

        # Q, K, V
        w_q = get_attn_param('query', 'w')
        w_k = get_attn_param('key', 'w')
        w_v = get_attn_param('value', 'w')
        
        b_q = get_attn_param('query', 'b')
        b_k = get_attn_param('key', 'b')
        b_v = get_attn_param('value', 'b')
        
        # Reshape Weights: (Embed, Heads, Dim) -> (Embed, Embed) -> Transpose
        def to_pt(w): 
            return w.reshape(w.shape[0], -1).t()

        # Reshape Bias: (Heads, Dim) -> (Embed)
        def to_pt_b(b):
            return b.reshape(-1)
        
        state_dict[f'layers.{i}.attn.in_proj_weight'] = torch.cat([to_pt(w_q), to_pt(w_k), to_pt(w_v)], dim=0)
        state_dict[f'layers.{i}.attn.in_proj_bias'] = torch.cat([to_pt_b(b_q), to_pt_b(b_k), to_pt_b(b_v)], dim=0)
        
        # Output Linear
        w_o = get_attn_param('linear', 'w')
        b_o = get_attn_param('linear', 'b')
        
        state_dict[f'layers.{i}.attn.out_proj.weight'] = w_o.t()
        state_dict[f'layers.{i}.attn.out_proj.bias'] = b_o

    # --- 4. FINAL LAYERS ---
    print("  - Converting Final Layers...")
    final_ln_key = f"layer_norm_{2*num_layers}"
    state_dict['final_norm.weight'] = get_param(jax_params, final_ln_key, 'scale')
    state_dict['final_norm.bias'] = get_param(jax_params, final_ln_key, 'offset')
    
    final_lin_idx = mlp_start_idx + 2*num_layers
    load_linear_to_torch(final_lin_idx, 'output_proj')

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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
