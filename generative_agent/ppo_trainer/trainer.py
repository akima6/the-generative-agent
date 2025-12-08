# generative_agent/ppo_trainer/trainer.py
import os
import sys
import yaml
import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np

# --- SETUP PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
CRYSTALFORMER_DIR = os.path.join(PROJECT_ROOT, "CrystalFormer")
sys.path.append(CRYSTALFORMER_DIR)

# --- CRYSTALFORMER IMPORTS (The Engine) ---
from crystalformer.src.transformer import make_transformer
from crystalformer.src.sample import sample_crystal
from crystalformer.src.loss import make_loss_fn
from crystalformer.src.lattice import norm_lattice
from crystalformer.reinforce.ppo import make_ppo_loss_fn

# --- LOCAL IMPORTS ---
from reward import RewardCalculator
from bridge import TensorBridge

# --- JAX SETUP ---
jax.config.update("jax_enable_x64", True)

# --- CONFIGURATION ---
CONFIG_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "epoch_005500.pkl")

# Training Hyperparameters (Office Laptop Safe Mode)
BATCH_SIZE = 4    # Crystals to imagine at once
PPO_EPOCHS = 2    # How many times to learn from the same batch
NUM_ITERATIONS = 5 # Run 5 full cycles (5 generations + 10 PPO updates)
LR = 1e-5         
CLIP_EPS = 0.2    
BETA = 0.05       
SPACE_GROUP = 225 # Fm-3m (Classic Cubic structures)

def load_config_and_model():
    """Loads the architecture config and the pretrained weights."""
    print(f"Loading Config from {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    print("Initializing JAX Model...")
    key = jax.random.PRNGKey(42)
    
    # Create the Transformer structure
    params, transformer = make_transformer(
        key,
        Nf=config['Nf'], Kx=config['Kx'], Kl=config['Kl'], n_max=config['n_max'],
        h0_size=config['h0_size'], num_layers=config['transformer_layers'],
        num_heads=config['num_heads'], key_size=config['key_size'],
        model_size=config['model_size'], embed_size=config['embed_size'],
        atom_types=config['atom_types'], wyck_types=config['wyck_types'],
        dropout_rate=config['dropout_rate']
    )
    
    print(f"Loading Checkpoint from {CHECKPOINT_PATH}")
    with open(CHECKPOINT_PATH, 'rb') as f:
        ckpt_data = pickle.load(f)
        params = ckpt_data['params'] if 'params' in ckpt_data else ckpt_data

    return config, params, transformer, key

def main():
    print("--- STARTING NOVAGEN DISCOVERY ENGINE (PHASE 2) ---")
    
    # 1. INITIALIZATION
    config, params, transformer, key = load_config_and_model()
    
    # Optimizer (Adam is standard for Transformers)
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)
    
    # 2. DEFINE LOSS FUNCTIONS
    _, logp_fn = make_loss_fn(
        config['n_max'], config['atom_types'], config['wyck_types'],
        config['Kx'], config['Kl'], transformer
    )
    
    ppo_loss_fn = make_ppo_loss_fn(logp_fn, CLIP_EPS, BETA)

    # Initialize Judges
    reward_calc = RewardCalculator()
    
    # Constants for Sampling
    atom_mask = jnp.zeros((config['atom_types']), dtype=int)
    atom_mask = jnp.stack([atom_mask] * config['n_max'], axis=0)
    constraints = jnp.arange(0, config['n_max'], 1) 
    
    # JIT Compile the Core Math
    calc_logp_jit = jax.jit(logp_fn, static_argnums=7) 
    
    # Define the Single Update Step
    @jax.jit
    def update_step(params, opt_state, key, x, old_logp, pretrain_logp, advantages):
        
        # Define the PPO Objective (to be MAXIMIZED)
        def objective_fn(p):
            return ppo_loss_fn(p, key, x, old_logp, pretrain_logp, advantages)
            
        # We want to MINIMIZE the negative objective
        def loss_fn(p):
            obj_val, kl_val = objective_fn(p)
            return -obj_val, (obj_val, kl_val)
            
        (neg_loss, (obj_val, kl_val)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Apply optimizer update
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Return the actual objective for logging (not the negative loss)
        return new_params, new_opt_state, obj_val, kl_val

    # Initialize a Running Average Baseline (crucial for non-zero advantages)
    # Start with a safe, low value (-3.0) to ensure non-zero advantage when reward is -5.0
    global_baseline = jnp.array(-3.0) 
    # Save original params to calculate KL Divergence (prevent model drift)
    pretrain_params = params

    # 3. THE TRAINING LOOP
    for it in range(NUM_ITERATIONS):
        print(f"\n=== Iteration {it+1}/{NUM_ITERATIONS} ===")
        
        # --- STEP A: DREAMING (Sampling) ---
        print("1. Sampling new crystals...")
        key, subkey = jax.random.split(key)
        
        # Generate raw tensors representing crystals
        XYZ, A, W, M, L_real = sample_crystal(
            subkey, transformer, params,
            config['n_max'], BATCH_SIZE, config['atom_types'], config['wyck_types'],
            config['Kx'], config['Kl'], SPACE_GROUP, 
            None, atom_mask, 1.0, 1.0, 1.0, constraints
        )
        
        G = jnp.full((BATCH_SIZE,), SPACE_GROUP)
        
        # --- STEP B: MEMORIZING (Log Probs) ---
        L_norm = norm_lattice(G, W, L_real)
        
        logp_w, logp_xyz, logp_a, logp_l = calc_logp_jit(
            params, subkey, G, L_norm, XYZ, A, W, False
        )
        old_logp = logp_w + logp_xyz + logp_a + logp_l
        
        p_logp_w, p_logp_xyz, p_logp_a, p_logp_l = calc_logp_jit(
            pretrain_params, subkey, G, L_norm, XYZ, A, W, False
        )
        pretrain_logp = p_logp_w + p_logp_xyz + p_logp_a + p_logp_l

        # --- STEP C: REALITY CHECK (Reward) ---
        print("2. Converting to Physical Structures...")
        structures = TensorBridge.batch_to_structures(G, L_real, XYZ, A, M)
        
        # DEBUG: Check if structures were successfully built
        valid_count = sum(1 for s in structures if s is not None)
        print(f"   [DEBUG] Successfully built {valid_count}/{BATCH_SIZE} Pymatgen structures.")
        if valid_count > 0:
            first_structure = next(s for s in structures if s is not None)
            print(f"   [DEBUG] Sample Composition: {first_structure.composition.reduced_formula}")
            print(f"   [DEBUG] Sample Density: {first_structure.density:.2f} g/cm3")
            
        print("3. Calculating Rewards (Relaxation + Band Gap)...")
        rewards_list = reward_calc.get_rewards(structures)
        rewards_jax = jnp.array(rewards_list)
        
        # Normalize Advantages with Running Average Baseline
        avg_reward = jnp.mean(rewards_jax)
        # Update running average (0.9 old baseline + 0.1 new average)
        # NOTE: Using a simple EMA, which is good for stability
        global_baseline = 0.9 * global_baseline + 0.1 * avg_reward 
        advantages = rewards_jax - global_baseline
        
        print(f"   > Batch Avg Reward: {avg_reward:.4f}")
        print(f"   > Running Baseline: {global_baseline:.4f}")
        print(f"   > Best Crystal: {max(rewards_list):.4f}")
        
        # --- STEP D: LEARNING (Update) ---
        print("4. Updating Neural Network Weights...")
        x_data = (G, L_norm, XYZ, A, W)
        
        for ppo_epoch in range(PPO_EPOCHS):
            key, subkey = jax.random.split(key)
            params, opt_state, obj_val, kl_val = update_step(
                params, opt_state, subkey, x_data, old_logp, pretrain_logp, advantages
            )
            print(f"   [Epoch {ppo_epoch+1}] PPO Obj: {obj_val:.6f} | Drift(KL): {kl_val:.6f}")

    # 4. SAVING
    print("\nTraining Complete.")
    save_path = os.path.join(PROJECT_ROOT, "pretrained_model", "finetuned_semiconductor.pkl")
    print(f"Saving new brain to: {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump({'params': params, 'opt_state': opt_state}, f)

if __name__ == "__main__":
    main()