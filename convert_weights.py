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
    # Ensure it's a numpy array first
    return torch.tensor(np.array(param))

def get_param(params, key, subkey):
    """Safely get param from dictionary."""
    if key not in params:
        raise KeyError(f"Key '{key}' not found in checkpoint.")
    val = params[key]
    if subkey not in val:
        raise KeyError(f"Subkey '{subkey}' not found in '{key}'. Available: {list(val.keys())}")
    return jax_to_torch(val[subkey])

def convert_weights(jax_params, torch_model):
    print("Converting weights...")
    state_dict = torch_model.state_dict()
    
    # --- 1. EMBEDDINGS ---
    # Scope '~' contains the embeddings directly
    print("  - Converting Embeddings...")
    embeddings_scope = jax_params.get('~')
    if embeddings_scope is None:
        print("Error: Scope '~' not found.")
        sys.exit(1)

    # DIRECT ACCESS (No ['embeddings'] subkey)
    state_dict['g_embeddings.weight'] = jax_to_torch(embeddings_scope['g_embeddings'])
    state_dict['w_embeddings.weight'] = jax_to_torch(embeddings_scope['w_embeddings'])
    state_dict['a_embeddings.weight'] = jax_to_torch(embeddings_scope['a_embeddings'])

    # --- 2. PROJECTIONS ---
    print("  - Converting Projections...")
    
    # Check for h0 MLP vs Table
    # If h0_size > 0, we expect linear_0 and linear_1 to be h0_mlp
    # If not, w_params would be in embeddings_scope
    
    if 'w_params' in embeddings_scope:
        print("    Found w_params (Table Lookup)")
        state_dict['w_params'] = jax_to_torch(embeddings_scope['w_params'])
        linear_offset = 0
    else:
        print("    Found h0 MLP (Linear 0 & 1)")
        linear_offset = 2

    # Helper to load linear layer
    def load_linear(jax_idx, torch_name):
        suffix = f"_{jax_idx}" if jax_idx > 0 else ""
        key = f"linear{suffix}"
        w = get_param(jax_params, key, 'w')
        b = get_param(jax_params, key, 'b')
        state_dict[f'{torch_name}.weight'] = w.t()
        state_dict[f'{torch_name}.bias'] = b

    # Load h0 MLP
    if linear_offset == 2:
        load_linear(0, 'h0_mlp.0')
        load_linear(1, 'h0_mlp.2')

    # fc_hW
    load_linear(linear_offset + 0, 'fc_hW')
    # fc_hA
    load_linear(linear_offset + 1, 'fc_hA')
    
    # fc_hXYZ (Average next 3)
    idx_base = linear_offset + 2
    w4 = get_param(jax_params, f"linear_{idx_base}", 'w')
    w5 = get_param(jax_params, f"linear_{idx_base+1}", 'w')
    w6 = get_param(jax_params, f"linear_{idx_base+2}", 'w')
    b4 = get_param(jax_params, f"linear_{idx_base}", 'b')
    b5 = get_param(jax_params, f"linear_{idx_base+1}", 'b')
    b6 = get_param(jax_params, f"linear_{idx_base+2}", 'b')
    
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
        load_linear(l1_idx, f'layers.{i}.mlp.0')
        load_linear(l2_idx, f'layers.{i}.mlp.2')
        
        # Attention
        # Construct flat keys: e.g. "multi_head_attention_1/query"
        attn_base = f"multi_head_attention_{i}" if i > 0 else "multi_head_attention"
        
        def get_attn_param(submodule, param):
            full_key = f"{attn_base}/{submodule}"
            return get_param(jax_params, full_key, param)

        w_q = get_attn_param('query', 'w')
        w_k = get_attn_param('key', 'w')
        w_v = get_attn_param('value', 'w')
        
        b_q = get_attn_param('query', 'b')
        b_k = get_attn_param('key', 'b')
        b_v = get_attn_param('value', 'b')
        
        # Reshape: (Embed, Heads, Dim) -> (Embed, Embed) -> Transpose
        def to_pt(w): 
            return w.reshape(w.shape[0], -1).t()
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
    load_linear(final_lin_idx, 'output_proj')

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
