import os
import sys
import yaml
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
import time

# --- SETUP PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up TWO levels: ppo_trainer -> generative_agent -> root
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
CRYSTALFORMER_DIR = os.path.join(PROJECT_ROOT, "CrystalFormer")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CRYSTALFORMER_DIR)

# --- IMPORTS ---
from crystalformer.src.transformer import make_transformer
from crystalformer.src.sample import sample_crystal
from crystalformer.src.loss import make_loss_fn
from crystalformer.src.lattice import norm_lattice
from crystalformer.reinforce.ppo import make_ppo_loss_fn

from generative_agent.ppo_trainer.reward import RewardCalculator
from generative_agent.ppo_trainer.bridge import TensorBridge

# --- CONFIGURATION ---
CONFIG_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "epoch_005500.pt")
LOG_CSV_PATH = os.path.join(PROJECT_ROOT, "discovery_log.csv")

# --- HYPERPARAMETERS (DISCOVERY MODE) ---
BATCH_SIZE = 16      # Higher batch = better chance to find stable crystals
PPO_EPOCHS = 4       # More learning steps per batch
NUM_ITERATIONS = 200 # Long run for discovery
LR = 1e-5
CLIP_EPS = 0.2
BETA = 0.05
SPACE_GROUP = 225
TEMP = 1.0

# Logging Verbosity
VERBOSE = False # Set True to see every PPO epoch detail

# Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cpu' and 'cuda' in os.environ.get('ACCELERATOR_TYPE', '').lower():
    DEVICE = torch.device("cuda:0")

def load_system():
    """Loads config, model, and initializes optimizer."""
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
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            
        if 'model_state_dict' in checkpoint:
            transformer.load_state_dict(checkpoint['model_state_dict'])
        else:
            transformer.load_state_dict(checkpoint)
    else:
        print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}. Model is random!")

    return config, transformer

def run_integrity_check(transformer, config, reward_calc, device):
    """
    Runs a tiny batch through the ENTIRE pipeline to ensure 
    Model -> Bridge -> Relaxer -> Oracle -> Reward works.
    """
    print("\n--- SYSTEM INTEGRITY CHECK (PRE-FLIGHT) ---")
    try:
        # 1. Check Model Generation
        print("[1/4] Testing Model Generation...", end=" ")
        with torch.no_grad():
            constraints = torch.arange(config['n_max'], device=device)
            XYZ, A, W, M, L = sample_crystal(
                key=None, transformer=transformer, params=None,
                n_max=config['n_max'], batchsize=2, # Tiny batch
                atom_types=config['atom_types'], wyck_types=config['wyck_types'],
                Kx=config['Kx'], Kl=config['Kl'], g=SPACE_GROUP,
                w_mask=None, atom_mask=None, top_p=1.0, temperature=1.0, T1=1.0,
                constraints=constraints
            )
        print("PASS")

        # 2. Check Bridge
        print("[2/4] Testing TensorBridge...", end=" ")
        G = torch.full((2,), SPACE_GROUP, device=device)
        structures = TensorBridge.batch_to_structures(G, L, XYZ, A, M)
        if len(structures) != 2: raise ValueError("Bridge returned wrong number of structures")
        print("PASS")

        # 3. Check Reward/Workers
        print("[3/4] Testing Workers (Relaxer/Oracle)...", end=" ")
        # This will actually call the subprocesses!
        rewards = reward_calc.get_rewards(structures)
        if len(rewards) != 2: raise ValueError("Reward calculator failed")
        print("PASS")
        
        print("[4/4] Integrity Check Complete. All systems nominal.")
        return True
    except Exception as e:
        print(f"\nFATAL ERROR during Integrity Check: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("--- STARTING NOVAGEN DISCOVERY ENGINE (PHASE 2) ---")
    
    # 1. SETUP
    config, transformer = load_system()
    
    # Initialize Reward Calculator (The Workers)
    reward_calc = RewardCalculator()
    
    # Perform Integrity Check
    if not run_integrity_check(transformer, config, reward_calc, DEVICE):
        print("System check failed. Aborting.")
        return

    # Initialize CSV Log
    if not os.path.exists(LOG_CSV_PATH):
        pd.DataFrame(columns=["Iteration", "Formula", "Density", "Reward", "Valid"]).to_csv(LOG_CSV_PATH, index=False)
    print(f"Logging discovery data to: {LOG_CSV_PATH}")

    # Setup PPO
    transformer_pretrain = copy.deepcopy(transformer)
    transformer_pretrain.eval()
    for p in transformer_pretrain.parameters(): p.requires_grad = False
        
    optimizer = optim.Adam(transformer.parameters(), lr=LR)
    
    _, logp_fn = make_loss_fn(
        config['n_max'], config['atom_types'], config['wyck_types'],
        config['Kx'], config['Kl'], transformer
    )
    ppo_loss_calc = make_ppo_loss_fn(logp_fn, CLIP_EPS, BETA)
    
    atom_mask = torch.zeros((config['n_max'], config['atom_types']), device=DEVICE) 
    constraints = torch.arange(config['n_max'], device=DEVICE)
    global_baseline = -3.0

    print(f"\n--- BEGINNING {NUM_ITERATIONS} ITERATION DISCOVERY RUN ---")

    # 2. MAIN LOOP
    for it in range(NUM_ITERATIONS):
        iter_start = time.time()
        
        # --- A. SAMPLE ---
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
        
        # --- B. LOG PROBS ---
        L_norm = norm_lattice(G, W, L_real)
        with torch.no_grad():
            lp_w, lp_xyz, lp_a, lp_l = logp_fn(transformer, G, L_norm, XYZ, A, W, is_train=False)
            old_logp = lp_w + lp_xyz + lp_a + lp_l
            
            pp_w, pp_xyz, pp_a, pp_l = logp_fn(transformer_pretrain, G, L_norm, XYZ, A, W, is_train=False)
            pretrain_logp = pp_w + pp_xyz + pp_a + pp_l

        # --- C. REWARD & LOGGING ---
        structures = TensorBridge.batch_to_structures(G, L_real, XYZ, A, M)
        
        # Prepare Log Entries
        log_entries = []
        for s in structures:
            if s:
                log_entries.append({"Iteration": it+1, "Formula": s.composition.reduced_formula, "Density": s.density, "Valid": True})
            else:
                log_entries.append({"Iteration": it+1, "Formula": "INVALID", "Density": 0.0, "Valid": False})

        # Calculate Rewards (The Slow Part)
        rewards_list = reward_calc.get_rewards(structures)
        rewards_tensor = torch.tensor(rewards_list, device=DEVICE, dtype=torch.float32)
        
        # Update Log with Rewards and Save
        for i, entry in enumerate(log_entries):
            entry["Reward"] = rewards_list[i]
        pd.DataFrame(log_entries).to_csv(LOG_CSV_PATH, mode='a', header=False, index=False)
        
        # Calculate Stats
        avg_reward = rewards_tensor.mean().item()
        max_reward = rewards_tensor.max().item()
        global_baseline = 0.9 * global_baseline + 0.1 * avg_reward 
        advantages = rewards_tensor - global_baseline
        
        # --- D. UPDATE ---
        transformer.train()
        x_data = (G, L_norm, XYZ, A, W)
        
        total_loss = 0
        total_kl = 0
        for _ in range(PPO_EPOCHS):
            optimizer.zero_grad()
            loss, kl = ppo_loss_calc(transformer, x_data, old_logp, pretrain_logp, advantages)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_kl += kl.item()
            
        # Report
        iter_time = time.time() - iter_start
        status_symbol = "✅" if max_reward > -4.0 else "❌"
        
        print(f"Iter {it+1}/{NUM_ITERATIONS} [{status_symbol}] | "
              f"Reward: {avg_reward:.2f} (Max: {max_reward:.2f}) | "
              f"PPO Loss: {total_loss/PPO_EPOCHS:.4f} | "
              f"KL: {total_kl/PPO_EPOCHS:.4f} | "
              f"Time: {iter_time:.1f}s")
              
        if VERBOSE:
            print(f"   Sample Formulas: {[e['Formula'] for e in log_entries[:3]]}...")

    # SAVE
    print("\n--- DISCOVERY RUN COMPLETE ---")
    save_path = os.path.join(PROJECT_ROOT, "pretrained_model", "finetuned_semiconductor.pt")
    torch.save(transformer.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    print(f"Full log saved to: {LOG_CSV_PATH}")

if __name__ == "__main__":
    main()
