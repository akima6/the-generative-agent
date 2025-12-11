# generative_agent/ppo_trainer/trainer.py
import os
import sys
import yaml
import torch
import torch.optim as optim
import numpy as np
import pickle
import copy

# --- SETUP PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
CRYSTALFORMER_DIR = os.path.join(PROJECT_ROOT, "CrystalFormer")
sys.path.append(CRYSTALFORMER_DIR)

# --- CRYSTALFORMER IMPORTS (Now using Rewritten Modules) ---
from crystalformer.src.transformer import make_transformer
from crystalformer.src.sample import sample_crystal
from crystalformer.src.loss import make_loss_fn
from crystalformer.src.lattice import norm_lattice
from crystalformer.reinforce.ppo import make_ppo_loss_fn

# --- LOCAL IMPORTS ---
from reward import RewardCalculator
from bridge import TensorBridge

# --- CONFIGURATION ---
CONFIG_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "epoch_005500.pt")

# Training Hyperparameters
BATCH_SIZE = 4    
PPO_EPOCHS = 2    
NUM_ITERATIONS = 5 
LR = 1e-5         
CLIP_EPS = 0.2    
BETA = 0.05       
SPACE_GROUP = 225 
TEMP = 1.0        

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config_and_model():
    """Loads the architecture config and the pretrained weights."""
    print(f"Loading Config from {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Initializing PyTorch Model on {DEVICE}...")
    
    transformer = make_transformer(
        key=None,
        Nf=config['Nf'], Kx=config['Kx'], Kl=config['Kl'], n_max=config['n_max'],
        h0_size=config['h0_size'], num_layers=config['transformer_layers'],
        num_heads=config['num_heads'], key_size=config['key_size'],
        model_size=config['model_size'], embed_size=config['embed_size'],
        atom_types=config['atom_types'], wyck_types=config['wyck_types'],
        dropout_rate=config['dropout_rate']
    ).to(DEVICE)
    
    # Load Weights
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading Checkpoint from {CHECKPOINT_PATH}")
        # FIX FOR PYTORCH 2.6+: Explicitly set weights_only=False
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't accept weights_only
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            
        if 'model_state_dict' in checkpoint:
            transformer.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assuming the file contains only the state_dict
            transformer.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found. Using random init.")

    return config, transformer

def main():
    # The rest of main() function is unchanged, running the PPO loop...
    # ... (omitted for brevity)
    
    print("--- STARTING NOVAGEN DISCOVERY ENGINE (PHASE 2 - PYTORCH INTEGRATED) ---")
    
    # 1. INITIALIZATION
    config, transformer = load_config_and_model()
    
    # Clone for pretrain reference
    transformer_pretrain = copy.deepcopy(transformer)
    transformer_pretrain.eval()
    for p in transformer_pretrain.parameters():
        p.requires_grad = False
        
    optimizer = optim.Adam(transformer.parameters(), lr=LR)
    
    _, logp_fn = make_loss_fn(
        config['n_max'], config['atom_types'], config['wyck_types'],
        config['Kx'], config['Kl'], transformer
    )
    
    ppo_loss_calc = make_ppo_loss_fn(logp_fn, CLIP_EPS, BETA)

    reward_calc = RewardCalculator()
    
    atom_mask = torch.zeros((config['n_max'], config['atom_types']), device=DEVICE) 
    constraints = torch.arange(config['n_max'], device=DEVICE)

    global_baseline = -3.0
    
    # 3. THE TRAINING LOOP
    for it in range(NUM_ITERATIONS):
        print(f"\n=== Iteration {it+1}/{NUM_ITERATIONS} ===")
        
        # --- STEP A: DREAMING (Sampling) ---
        print("1. Sampling new crystals...")
        
        with torch.no_grad():
            XYZ, A, W, M, L_real = sample_crystal(
                key=None, transformer=transformer, params=None,
                n_max=config['n_max'], batchsize=BATCH_SIZE,
                atom_types=config['atom_types'], wyck_types=config['wyck_types'],
                Kx=config['Kx'], Kl=config['Kl'], g=SPACE_GROUP,
                w_mask=None, atom_mask=None, top_p=1.0, temperature=TEMP, T1=TEMP,
                constraints=constraints
            )
        
        G = torch.full((BATCH_SIZE,), SPACE_GROUP, device=DEVICE)
        
        # --- STEP B: MEMORIZING (Log Probs) ---
        L_norm = norm_lattice(G, W, L_real)
        
        with torch.no_grad():
            lp_w, lp_xyz, lp_a, lp_l = logp_fn(
                transformer, G, L_norm, XYZ, A, W, is_train=False
            )
            old_logp = lp_w + lp_xyz + lp_a + lp_l
            
            pp_w, pp_xyz, pp_a, pp_l = logp_fn(
                transformer_pretrain, G, L_norm, XYZ, A, W, is_train=False
            )
            pretrain_logp = pp_w + pp_xyz + pp_a + pp_l

        # --- STEP C: REALITY CHECK (Reward) ---
        print("2. Converting to Physical Structures...")
        structures = TensorBridge.batch_to_structures(G, L_real, XYZ, A, M)
        
        valid_count = sum(1 for s in structures if s is not None)
        print(f"   [DEBUG] Successfully built {valid_count}/{BATCH_SIZE} Pymatgen structures.")
            
        print("3. Calculating Rewards...")
        rewards_list = reward_calc.get_rewards(structures)
        rewards_tensor = torch.tensor(rewards_list, device=DEVICE, dtype=torch.float32)
        
        avg_reward = rewards_tensor.mean().item()
        global_baseline = 0.9 * global_baseline + 0.1 * avg_reward 
        advantages = rewards_tensor - global_baseline
        
        print(f"   > Batch Avg Reward: {avg_reward:.4f}")
        print(f"   > Running Baseline: {global_baseline:.4f}")
        
        # --- STEP D: LEARNING (Update) ---
        print("4. Updating Neural Network Weights...")
        transformer.train()
        
        x_data = (G, L_norm, XYZ, A, W)
        
        for ppo_epoch in range(PPO_EPOCHS):
            optimizer.zero_grad()
            
            loss, kl = ppo_loss_calc(transformer, x_data, old_logp, pretrain_logp, advantages)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            
            print(f"   [Epoch {ppo_epoch+1}] PPO Loss: {loss.item():.6f} | Drift(KL): {kl.item():.6f}")

    # 4. SAVING
    print("\nTraining Complete.")
    save_path = os.path.join(PROJECT_ROOT, "pretrained_model", "finetuned_semiconductor.pt")
    print(f"Saving new brain to: {save_path}")
    torch.save(transformer.state_dict(), save_path)


if __name__ == "__main__":
    main()
