import os
import sys
import yaml
import pickle
import torch
import numpy as np

# --- PATHS ---
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()

# If CURRENT_DIR ends with "generative_agent", go up one level
if CURRENT_DIR.replace("\\", "/").endswith("generative_agent"):
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
else:
    # Otherwise assume CURRENT_DIR *is* the project root
    PROJECT_ROOT = CURRENT_DIR

CRYSTALFORMER_DIR = os.path.join(PROJECT_ROOT, "CrystalFormer")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CRYSTALFORMER_DIR)

PROJECT_ROOT, CURRENT_DIR, CRYSTALFORMER_DIR

from crystalformer.src.transformer import make_transformer

def load_jax_weights(path):
    print(f"Loading JAX weights from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['params'] if 'params' in data else data

def jax_to_torch_linear(param):
    # JAX: (In, Out) -> PyTorch: (Out, In)
    return torch.tensor(np.array(param)).t()

def jax_to_torch_embed(param):
    # JAX: (Vocab, Dim) -> PyTorch: (Vocab, Dim)
    return torch.tensor(np.array(param))

def convert_weights(jax_params, model, config):
    print("Starting FINAL WEIGHT REPAIR (Split Coords + Transpose)...")
    state_dict = model.state_dict()
    
    # 1. Embeddings
    if '~' in jax_params:
        embed = jax_params['~']
        state_dict['g_embeddings.weight'] = jax_to_torch_embed(embed['g_embeddings'])
        state_dict['w_embeddings.weight'] = jax_to_torch_embed(embed['w_embeddings'])
        state_dict['a_embeddings.weight'] = jax_to_torch_embed(embed['a_embeddings'])
    
    # 2. Linear Projections
    def map_linear(jax_name, pt_name):
        w = jax_params[jax_name]['w']
        b = jax_params[jax_name]['b']
        state_dict[f'{pt_name}.weight'] = jax_to_torch_linear(w)
        state_dict[f'{pt_name}.bias'] = torch.tensor(np.array(b))

    map_linear('linear', 'h0_mlp.0')
    map_linear('linear_1', 'h0_mlp.2')
    map_linear('linear_2', 'fc_hW')
    map_linear('linear_3', 'fc_hA')
    
    # --- FIX FOR fc_hX, fc_hY, fc_hZ ---
    # Map JAX linear_4 -> fc_hX, linear_5 -> fc_hY, linear_6 -> fc_hZ
    print("  - Mapping coordinate projections (linear_4, 5, 6)...")
    
    def load_coord_linear(jax_idx, pt_name):
        w = jax_params[f'linear_{jax_idx}']['w']
        b = jax_params[f'linear_{jax_idx}']['b']
        state_dict[f'{pt_name}.weight'] = jax_to_torch_linear(w)
        state_dict[f'{pt_name}.bias'] = torch.tensor(np.array(b))

    load_coord_linear(4, 'fc_hX')
    load_coord_linear(5, 'fc_hY')
    load_coord_linear(6, 'fc_hZ')
    
    # 3. Transformer Blocks
    num_layers = config['transformer_layers']
    mlp_offset = 7
    
    for i in range(num_layers):
        # Norms
        ln1 = f"layer_norm_{2*i}" if 2*i>0 else "layer_norm"
        ln2 = f"layer_norm_{2*i+1}"
        state_dict[f'layers.{i}.ln1.weight'] = jax_to_torch_embed(jax_params[ln1]['scale'])
        state_dict[f'layers.{i}.ln1.bias'] = jax_to_torch_embed(jax_params[ln1]['offset'])
        state_dict[f'layers.{i}.ln2.weight'] = jax_to_torch_embed(jax_params[ln2]['scale'])
        state_dict[f'layers.{i}.ln2.bias'] = jax_to_torch_embed(jax_params[ln2]['offset'])
        
        # MLP
        l1 = f"linear_{mlp_offset + 2*i}"
        l2 = f"linear_{mlp_offset + 2*i + 1}"
        state_dict[f'layers.{i}.mlp.0.weight'] = jax_to_torch_linear(jax_params[l1]['w'])
        state_dict[f'layers.{i}.mlp.0.bias'] = torch.tensor(np.array(jax_params[l1]['b']))
        state_dict[f'layers.{i}.mlp.2.weight'] = jax_to_torch_linear(jax_params[l2]['w'])
        state_dict[f'layers.{i}.mlp.2.bias'] = torch.tensor(np.array(jax_params[l2]['b']))
        
        # Attention
        attn_name = f"multi_head_attention_{i}" if i>0 else "multi_head_attention"
        
        # Q, K, V (Simple Transpose logic works for PyTorch MultiheadAttention inputs)
        state_dict[f'layers.{i}.attn.q_proj.weight'] = jax_to_torch_linear(jax_params[f"{attn_name}/query"]['w'])
        state_dict[f'layers.{i}.attn.q_proj.bias'] = torch.tensor(np.array(jax_params[f"{attn_name}/query"]['b']))
        
        state_dict[f'layers.{i}.attn.k_proj.weight'] = jax_to_torch_linear(jax_params[f"{attn_name}/key"]['w'])
        state_dict[f'layers.{i}.attn.k_proj.bias'] = torch.tensor(np.array(jax_params[f"{attn_name}/key"]['b']))
        
        state_dict[f'layers.{i}.attn.v_proj.weight'] = jax_to_torch_linear(jax_params[f"{attn_name}/value"]['w'])
        state_dict[f'layers.{i}.attn.v_proj.bias'] = torch.tensor(np.array(jax_params[f"{attn_name}/value"]['b']))
        
        # Output Proj
        state_dict[f'layers.{i}.attn.o_proj.weight'] = jax_to_torch_linear(jax_params[f"{attn_name}/linear"]['w'])
        state_dict[f'layers.{i}.attn.o_proj.bias'] = torch.tensor(np.array(jax_params[f"{attn_name}/linear"]['b']))

    # 4. Final Layers
    final_ln = f"layer_norm_{2*num_layers}"
    final_lin = f"linear_{mlp_offset + 2*num_layers}"
    
    state_dict['final_norm.weight'] = jax_to_torch_embed(jax_params[final_ln]['scale'])
    state_dict['final_norm.bias'] = jax_to_torch_embed(jax_params[final_ln]['offset'])
    
    state_dict['output_proj.weight'] = jax_to_torch_linear(jax_params[final_lin]['w'])
    state_dict['output_proj.bias'] = torch.tensor(np.array(jax_params[final_lin]['b']))
    
    return state_dict

def main():
    config_path = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    model = make_transformer(
        None, config['Nf'], config['Kx'], config['Kl'], config['n_max'],
        config['h0_size'], config['transformer_layers'], config['num_heads'],
        config['key_size'], config['model_size'], config['embed_size'],
        config['atom_types'], config['wyck_types'], 0.0
    )
    
    jax_path = os.path.join(PROJECT_ROOT, "pretrained_model", "epoch_005500.pkl")
    if not os.path.exists(jax_path):
        print(f"Error: Original JAX checkpoint not found at {jax_path}")
        return

    with open(jax_path, 'rb') as f:
        jax_raw = pickle.load(f)
    jax_params = jax_raw['params'] if 'params' in jax_raw else jax_raw
    
    try:
        new_state = convert_weights(jax_params, model, config)
        out_path = os.path.join(PROJECT_ROOT, "pretrained_model", "epoch_005500_FIXED.pt")
        torch.save(new_state, out_path)
        print(f"SAVED REPAIRED MODEL TO: {out_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
