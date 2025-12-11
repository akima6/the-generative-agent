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
    if key not in params:
        raise KeyError(f"Key '{key}' not found.")
    if subkey not in params[key]:
        # Fallback: Maybe the parameter IS the key value (no subkey)?
        # This happens if someone did params['key'] = array
        raise KeyError(f"Subkey '{subkey}' not found in '{key}'. Available: {list(params[key].keys())}")
    return jax_to_torch(params[key][subkey])

def get_direct_param(params, scope, key):
    """Gets a parameter that is directly stored (no w/b sub-dictionary)."""
    if scope not in params:
        raise KeyError(f"Scope '{scope}' not found.")
    if key not in params[scope]:
        raise KeyError(f"Key '{key}' not found in scope '{scope}'.")
    return jax_to_torch(params[scope][key])

def convert_weights(jax_params, torch_model):
    print("Converting weights (Mapped Structure - Fixed Embeddings)...")
    state_dict = torch_model.state_dict()
    
    # --- 1. EMBEDDINGS (Under '~') ---
    # hk.get_parameter stores weights directly under the name in the scope
    print("  - Converting Embeddings...")
    embeddings_scope = jax_params.get('~')
    if embeddings_scope is None:
        print("Error: Scope '~' not found. Cannot load embeddings.")
        sys.exit(1)

    state_dict['g_embeddings.weight'] = jax_to_torch(embeddings_scope['g_embeddings'])
    state_dict['w_embeddings.weight'] = jax_to_torch(embeddings_scope['w_embeddings'])
    state_dict['a_embeddings.weight'] = jax_to_torch(embeddings_scope['a_embeddings'])

    # --- 2. h0 MLP & PROJECTIONS ---
    print("  - Converting Projections...")
    
    # Check for w_params (if h0_size=0)
    # It would likely be in '~' if it exists
    if 'w_params' in embeddings_scope:
        print("    Found w_params (Table Lookup)")
        state_dict['w_params'] = jax_to_torch(embeddings_scope['w_params'])
        linear_offset = 0
    else:
        # h0 MLP exists, uses linear_0 and linear_1
        print("    Found h0 MLP (Linear 0 & 1)")
        linear_offset = 2

    def get_lin(idx, torch_name):
        suffix = f"_{idx}" if idx > 0 else ""
        key = f"linear{suffix}"
        w = get_param(jax_params, key, 'w')
        b = get_param(jax_params, key, 'b')
        state_dict[f'{torch_name}.weight'] = w.t()
        state_dict[f'{torch_name}.bias'] = b

    # If h0 MLP exists, load it
    if linear_offset == 2:
        get_lin(0, 'h0_mlp.0')
        get_lin(1, 'h0_mlp.2')

    # Projections (indices shift based on linear_offset)
    # fc_hW
    get_lin(linear_offset + 0, 'fc_hW')
    # fc_hA
    get_lin(linear_offset + 1, 'fc_hA')
    
    # fc_hXYZ (Average next 3)
    idx_base = linear_offset + 2
    w4 = get_param(jax_params, f'linear_{idx_base}', 'w')
    w5 = get_param(jax_params, f'linear_{idx_base+1}', 'w')
    w6 = get_param(jax_params, f'linear_{idx_base+2}', 'w')
    b4 = get_param(jax_params, f'linear_{idx_base}', 'b')
    b5 = get_param(jax_params, f'linear_{idx_base+1}', 'b')
    b6 = get_param(jax_params, f'linear_{idx_base+2}', 'b')
    
    state_dict['fc_hXYZ.weight'] = ((w4+w5+w6)/3.0).t()
    state_dict['fc_hXYZ.bias'] = (b4+b5+b6)/3.0

    # --- 3. TRANSFORMER BLOCKS ---
    print("  - Converting Transformer Blocks...")
    num_layers = len(torch_model.layers)
    # Start after the 5 projection layers (2 for h0 + 1 hW + 1 hA + 3 hXYZ = 7 linears consumed so far?)
    # Wait, indices:
    # 0,1 (h0)
    # 2 (hW)
    # 3 (hA)
    # 4,5,6 (hXYZ)
    # Next is 7.
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
        get_lin(l1_idx, f'layers.{i}.mlp.0')
        get_lin(l2_idx, f'layers.{i}.mlp.2')
        
        # Attention
        attn_key = f"multi_head_attention_{i}" if i > 0 else "multi_head_attention"
        attn_dict = jax_params[attn_key]
        
        def to_pt(w): 
            return jax_to_torch(w.reshape(w.shape[0], -1)).t()
            
        w_q = to_pt(attn_dict['query']['w'])
        w_k = to_pt(attn_dict['key']['w'])
        w_v = to_pt(attn_dict['value']['w'])
        
        b_q = jax_to_torch(attn_dict['query']['b'].reshape(-1))
        b_k = jax_to_torch(attn_dict['key']['b'].reshape(-1))
        b_v = jax_to_torch(attn_dict['value']['b'].reshape(-1))
        
        state_dict[f'layers.{i}.attn.in_proj_weight'] = torch.cat([w_q, w_k, w_v], dim=0)
        state_dict[f'layers.{i}.attn.in_proj_bias'] = torch.cat([b_q, b_k, b_v], dim=0)
        
        w_o = jax_to_torch(attn_dict['linear']['w']).t()
        b_o = jax_to_torch(attn_dict['linear']['b'])
        
        state_dict[f'layers.{i}.attn.out_proj.weight'] = w_o
        state_dict[f'layers.{i}.attn.out_proj.bias'] = b_o

    # --- 4. FINAL LAYERS ---
    print("  - Converting Final Layers...")
    final_ln_key = f"layer_norm_{2*num_layers}"
    state_dict['final_norm.weight'] = get_param(jax_params, final_ln_key, 'scale')
    state_dict['final_norm.bias'] = get_param(jax_params, final_ln_key, 'offset')
    
    final_lin_idx = mlp_start_idx + 2*num_layers
    get_lin(final_lin_idx, 'output_proj')

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
