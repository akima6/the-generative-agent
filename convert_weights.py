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
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error unpickling: {e}")
        sys.exit(1)
        
    params = data['params'] if 'params' in data else data
    return params

def jax_to_torch(param):
    return torch.tensor(np.array(param))

def find_key_with_suffix(params, suffix):
    """Recursively search for a key ending with suffix."""
    for k in params.keys():
        if k.endswith(suffix):
            return k
    return None

def get_param(params, scope, name):
    """Robustly fetch param given scope and name."""
    # Try exact match
    if scope in params and name in params[scope]:
        return params[scope][name]
    
    # Try finding scope with different slash convention
    # e.g., 'network/~/linear' vs 'network/linear'
    norm_scope = scope.replace('/~/', '/')
    for k in params.keys():
        if k.replace('/~/', '/') == norm_scope:
            if name in params[k]:
                return params[k][name]
            
    # Fail
    raise KeyError(f"Could not find scope '{scope}' or '{norm_scope}' with param '{name}'")

def convert_weights(jax_params, torch_model):
    print("Converting weights...")
    state_dict = torch_model.state_dict()
    
    # 1. Detect Prefix
    # Find where g_embeddings are stored to guess the prefix
    g_key = find_key_with_suffix(jax_params, 'g_embeddings')
    if g_key is None:
        print("ERROR: Could not find 'g_embeddings' in checkpoint.")
        print("Available Top-Level Keys:", list(jax_params.keys())[:20])
        sys.exit(1)
    
    # prefix is everything up to g_embeddings
    # e.g. 'network/~/g_embeddings' -> prefix='network/~/'
    prefix = g_key.replace('g_embeddings', '')
    print(f"Detected naming prefix: '{prefix}'")

    # Helper to fetch with prefix
    def get_w(module_name, param_name='w'):
        # Construct scope name (e.g. network/~/linear)
        scope = f"{prefix}{module_name}"
        # Haiku usually stores weights in 'w' or 'embeddings' or 'scale'
        if param_name == 'w' and 'w' not in jax_params.get(scope, {}):
             # Try 'embeddings' if it's an embedding layer
             if 'embeddings' in jax_params.get(scope, {}):
                 return jax_params[scope]['embeddings']
        
        try:
            return get_param(jax_params, scope, param_name)
        except KeyError:
            # Fallback for sequential/linear naming differences
            # Sometimes h0_mlp uses 'sequential/linear'
            if 'sequential' in module_name:
                # Try finding any key that looks right
                pass
            raise

    # --- EMBEDDINGS ---
    state_dict['g_embeddings.weight'] = jax_to_torch(get_w('g_embeddings', 'embeddings'))
    state_dict['w_embeddings.weight'] = jax_to_torch(get_w('w_embeddings', 'embeddings'))
    state_dict['a_embeddings.weight'] = jax_to_torch(get_w('a_embeddings', 'embeddings'))
    
    # --- H0 MLP (If exists) ---
    # Logic: If h0_size > 0, there is a Sequential block.
    # Usually: 'sequential/linear', 'sequential/linear_1'.
    # Check if 'sequential/linear' exists.
    if f"{prefix}sequential/linear" in jax_params:
        print("Found h0_mlp layers...")
        # Linear 1
        w = get_w('sequential/linear', 'w')
        b = get_w('sequential/linear', 'b')
        state_dict['h0_mlp.0.weight'] = jax_to_torch(w).t()
        state_dict['h0_mlp.0.bias'] = jax_to_torch(b)
        
        # Linear 2
        w = get_w('sequential/linear_1', 'w')
        b = get_w('sequential/linear_1', 'b')
        state_dict['h0_mlp.2.weight'] = jax_to_torch(w).t()
        state_dict['h0_mlp.2.bias'] = jax_to_torch(b)
    elif 'w_params' in str(jax_params.keys()):
        # Table fallback
        state_dict['w_params'] = jax_to_torch(get_w('w_params', 'w_params'))

    # --- PROJECTIONS ---
    # Identification of which "linear" is which.
    # Standard order in JAX code:
    # 1. h0_mlp (handled above)
    # 2. hW (linear) -> if h0 exists, this might be linear_2
    # 3. hA (linear_1) -> linear_3
    # 4. hXYZ (linear_2,3,4) -> linear_4,5,6
    
    # Let's verify by checking keys.
    all_keys = sorted(jax_params.keys())
    linear_keys = [k for k in all_keys if 'linear' in k and 'multi_head' not in k]
    
    # Filter out sequential linears
    main_linears = [k for k in linear_keys if 'sequential' not in k]
    # Sort by index suffix
    # linear, linear_1, linear_2 ...
    def sort_key(s):
        parts = s.split('_')
        if parts[-1].isdigit(): return int(parts[-1])
        return 0
    main_linears.sort(key=sort_key)
    
    print(f"Found Projection Linears: {main_linears}")
    
    # Mapping
    # hW -> 0
    # hA -> 1
    # hX -> 2
    # hY -> 3
    # hZ -> 4
    
    def load_lin(jax_key, torch_name):
        # jax_key is full path e.g. network/~/linear_5
        w = jax_params[jax_key]['w']
        b = jax_params[jax_key]['b']
        state_dict[f'{torch_name}.weight'] = jax_to_torch(w).t()
        state_dict[f'{torch_name}.bias'] = jax_to_torch(b)

    if len(main_linears) >= 5:
        load_lin(main_linears[0], 'fc_hW')
        load_lin(main_linears[1], 'fc_hA')
        
        # hXYZ (Average 3)
        w2 = jax_to_torch(jax_params[main_linears[2]]['w'])
        w3 = jax_to_torch(jax_params[main_linears[3]]['w'])
        w4 = jax_to_torch(jax_params[main_linears[4]]['w'])
        b2 = jax_to_torch(jax_params[main_linears[2]]['b'])
        b3 = jax_to_torch(jax_params[main_linears[3]]['b'])
        b4 = jax_to_torch(jax_params[main_linears[4]]['b'])
        
        state_dict['fc_hXYZ.weight'] = ((w2+w3+w4)/3.0).t()
        state_dict['fc_hXYZ.bias'] = (b2+b3+b4)/3.0
        
        # The rest of linears are in the transformer blocks
        block_linears = main_linears[5:]
    else:
        print("Error: Not enough linear layers found for projections.")
        sys.exit(1)

    # --- TRANSFORMER BLOCKS ---
    num_layers = len(torch_model.layers)
    
    # LayerNorms
    # Filter keys
    ln_keys = [k for k in all_keys if 'layer_norm' in k]
    ln_keys.sort(key=sort_key)
    
    # Blocks
    for i in range(num_layers):
        # LN
        # Each block has 2 LNs.
        ln1_key = ln_keys[2*i]
        ln2_key = ln_keys[2*i + 1]
        
        state_dict[f'layers.{i}.ln1.weight'] = jax_to_torch(jax_params[ln1_key]['scale'])
        state_dict[f'layers.{i}.ln1.bias'] = jax_to_torch(jax_params[ln1_key]['offset'])
        
        state_dict[f'layers.{i}.ln2.weight'] = jax_to_torch(jax_params[ln2_key]['scale'])
        state_dict[f'layers.{i}.ln2.bias'] = jax_to_torch(jax_params[ln2_key]['offset'])
        
        # MLP
        # 2 Linears per block.
        # Taken from block_linears list
        l1_key = block_linears[2*i]
        l2_key = block_linears[2*i + 1]
        
        w1 = jax_to_torch(jax_params[l1_key]['w']).t()
        b1 = jax_to_torch(jax_params[l1_key]['b'])
        state_dict[f'layers.{i}.mlp.0.weight'] = w1
        state_dict[f'layers.{i}.mlp.0.bias'] = b1
        
        w2 = jax_to_torch(jax_params[l2_key]['w']).t()
        b2 = jax_to_torch(jax_params[l2_key]['b'])
        state_dict[f'layers.{i}.mlp.2.weight'] = w2
        state_dict[f'layers.{i}.mlp.2.bias'] = b2
        
        # ATTENTION
        attn_suffix = f"_{i}" if i > 0 else ""
        # Look for multi_head_attention{suffix}
        # Be careful if there are other attention modules? No.
        attn_key = f"{prefix}multi_head_attention{attn_suffix}"
        
        jax_attn = jax_params[attn_key]
        
        def to_pt_attn(w):
            return jax_to_torch(w.reshape(w.shape[0], -1)).t()

        w_q = to_pt_attn(jax_attn['query']['w'])
        w_k = to_pt_attn(jax_attn['key']['w'])
        w_v = to_pt_attn(jax_attn['value']['w'])
        b_q = jax_to_torch(jax_attn['query']['b'].reshape(-1))
        b_k = jax_to_torch(jax_attn['key']['b'].reshape(-1))
        b_v = jax_to_torch(jax_attn['value']['b'].reshape(-1))
        
        state_dict[f'layers.{i}.attn.in_proj_weight'] = torch.cat([w_q, w_k, w_v], dim=0)
        state_dict[f'layers.{i}.attn.in_proj_bias'] = torch.cat([b_q, b_k, b_v], dim=0)
        
        state_dict[f'layers.{i}.attn.out_proj.weight'] = jax_to_torch(jax_attn['linear']['w']).t()
        state_dict[f'layers.{i}.attn.out_proj.bias'] = jax_to_torch(jax_attn['linear']['b'])

    # --- FINAL LAYERS ---
    # Final LN is the last one in ln_keys
    final_ln_key = ln_keys[-1]
    state_dict['final_norm.weight'] = jax_to_torch(jax_params[final_ln_key]['scale'])
    state_dict['final_norm.bias'] = jax_to_torch(jax_params[final_ln_key]['offset'])
    
    # Final Linear is the last one in block_linears (or main_linears)
    final_lin_key = block_linears[-1]
    state_dict['output_proj.weight'] = jax_to_torch(jax_params[final_lin_key]['w']).t()
    state_dict['output_proj.bias'] = jax_to_torch(jax_params[final_lin_key]['b'])

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
    jax_params = load_jax_weights(jax_path)
    
    model = convert_weights(jax_params, model)
    
    out_path = os.path.join(CURRENT_DIR, "pretrained_model", "epoch_005500.pt")
    torch.save(model.state_dict(), out_path)
    print(f"Saved PyTorch weights to {out_path}")

if __name__ == "__main__":
    main()
