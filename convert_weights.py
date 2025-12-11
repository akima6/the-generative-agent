import os
import sys
import yaml
import pickle
import torch
import numpy as np
# We need JAX only to unpickle the arrays (Colab has JAX installed by default)
# If the pickle contains JAX DeviceArrays, we need jax loaded to deserialize them.
import jax
import jax.numpy as jnp

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CRYSTALFORMER_DIR = os.path.join(CURRENT_DIR, "CrystalFormer")
sys.path.append(CRYSTALFORMER_DIR)

from crystalformer.src.transformer import make_transformer

def load_jax_weights(path):
    print(f"Loading JAX weights from {path}...")
    # We use jax.default_backend() to ensure JAX is initialized for unpickling
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except ModuleNotFoundError as e:
        # Fallback: specific handling if some specific module is missing in unpickling
        print(f"Error unpickling: {e}")
        print("Note: If the checkpoint contains custom Haiku classes, you might need to 'pip install dm-haiku' temporarily.")
        sys.exit(1)
        
    # Handle if it's a dict with 'params' key or just params
    params = data['params'] if 'params' in data else data
    return params

def jax_to_torch(param):
    # Convert JAX array to Numpy, then to Torch
    return torch.tensor(np.array(param))

def convert_weights(jax_params, torch_model):
    print("Converting weights...")
    state_dict = torch_model.state_dict()
    
    # Helper to clean Haiku names
    # Haiku structure: {'module_name': {'param_name': array}}
    
    # 1. Embeddings
    state_dict['g_embeddings.weight'] = jax_to_torch(jax_params['network/~/g_embeddings']['embeddings'])
    state_dict['w_embeddings.weight'] = jax_to_torch(jax_params['network/~/w_embeddings']['embeddings'])
    state_dict['a_embeddings.weight'] = jax_to_torch(jax_params['network/~/a_embeddings']['embeddings'])
    
    # h0 / w_params
    if 'network/~/w_params' in jax_params:
        state_dict['w_params'] = jax_to_torch(jax_params['network/~/w_params']['w_params'])
    
    # 2. Projections
    def copy_linear(jax_name, torch_name):
        w = jax_params[f'network/~/{jax_name}']['w']
        b = jax_params[f'network/~/{jax_name}']['b']
        state_dict[f'{torch_name}.weight'] = jax_to_torch(w).t() # Transpose W for PyTorch
        state_dict[f'{torch_name}.bias'] = jax_to_torch(b)

    # fc_hW
    copy_linear('linear', 'fc_hW')
    
    # fc_hA
    copy_linear('linear_1', 'fc_hA')
    
    # fc_hXYZ
    w2 = jax_to_torch(jax_params['network/~/linear_2']['w'])
    w3 = jax_to_torch(jax_params['network/~/linear_3']['w'])
    w4 = jax_to_torch(jax_params['network/~/linear_4']['w'])
    
    b2 = jax_to_torch(jax_params['network/~/linear_2']['b'])
    b3 = jax_to_torch(jax_params['network/~/linear_3']['b'])
    b4 = jax_to_torch(jax_params['network/~/linear_4']['b'])
    
    # Average them for the shared layer
    w_avg = (w2 + w3 + w4) / 3.0
    b_avg = (b2 + b3 + b4) / 3.0
    
    state_dict['fc_hXYZ.weight'] = w_avg.t()
    state_dict['fc_hXYZ.bias'] = b_avg

    # 3. Transformer Layers
    num_layers = len(torch_model.layers)
    
    for i in range(num_layers):
        # Layer Norms
        ln1_idx = 2 * i
        ln2_idx = 2 * i + 1
        ln_suffix_1 = f"_{ln1_idx}" if ln1_idx > 0 else ""
        ln_suffix_2 = f"_{ln2_idx}" 
        
        # ln1
        state_dict[f'layers.{i}.ln1.weight'] = jax_to_torch(jax_params[f'network/~/layer_norm{ln_suffix_1}']['scale'])
        state_dict[f'layers.{i}.ln1.bias'] = jax_to_torch(jax_params[f'network/~/layer_norm{ln_suffix_1}']['offset'])
        
        # ln2
        state_dict[f'layers.{i}.ln2.weight'] = jax_to_torch(jax_params[f'network/~/layer_norm{ln_suffix_2}']['scale'])
        state_dict[f'layers.{i}.ln2.bias'] = jax_to_torch(jax_params[f'network/~/layer_norm{ln_suffix_2}']['offset'])
        
        # MLP
        mlp1_idx = 5 + 2 * i
        mlp2_idx = 5 + 2 * i + 1
        
        # MLP 1
        w1 = jax_to_torch(jax_params[f'network/~/linear_{mlp1_idx}']['w']).t()
        b1 = jax_to_torch(jax_params[f'network/~/linear_{mlp1_idx}']['b'])
        state_dict[f'layers.{i}.mlp.0.weight'] = w1
        state_dict[f'layers.{i}.mlp.0.bias'] = b1
        
        # MLP 2
        w2 = jax_to_torch(jax_params[f'network/~/linear_{mlp2_idx}']['w']).t()
        b2 = jax_to_torch(jax_params[f'network/~/linear_{mlp2_idx}']['b'])
        state_dict[f'layers.{i}.mlp.2.weight'] = w2
        state_dict[f'layers.{i}.mlp.2.bias'] = b2
        
        # Attention
        attn_suffix = f"_{i}" if i > 0 else ""
        jax_attn = jax_params[f'network/~/multi_head_attention{attn_suffix}']
        
        def to_pt_attn(w):
            return jax_to_torch(w.reshape(w.shape[0], -1)).t()

        w_q = to_pt_attn(jax_attn['query']['w'])
        w_k = to_pt_attn(jax_attn['key']['w'])
        w_v = to_pt_attn(jax_attn['value']['w'])
        
        b_q = jax_to_torch(jax_attn['query']['b'].reshape
