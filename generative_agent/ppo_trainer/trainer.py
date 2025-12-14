import os
import sys
import yaml
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
import time
from pymatgen.core import Element

# --- SETUP PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
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

# --- HYPERPARAMETERS ---
BATCH_SIZE = 16
PPO_EPOCHS = 4
NUM_ITERATIONS = 200
LR = 5e-6
CLIP_EPS = 0.2
BETA = 0.05
SPACE_GROUP = 225
TEMP = 1.5 # High temp for exploration

# --- ELEMENT UNIVERSE (TARGET LIST) ---
# Symbols provided by user
ELEMENT_SYMBOLS = [
    "B", "C", "N", "O", "F",
    "Al", "Si", "P", "S", "Cl",
    "Ga", "Ge", "As", "Se", "Br",
    "In", "Sn", "Sb", "Te", "I",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re", "Os"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cpu' and 'cuda' in os.environ.get('ACCELERATOR_TYPE', '').lower():
    DEVICE = torch.device("cuda:0")

def get_atom_mask(atom_types, device):
    """Creates a boolean mask allowing only the selected elements."""
    mask = torch.zeros((atom_types,), device=device)
    mask[0] = 1.0 # Always allow padding (0)
    
    count = 0
    for sym in ELEMENT_SYMBOLS:
        try:
            z = Element(sym).Z
            if z < atom_types:
                mask[z] = 1.0
                count += 1
        except:
            pass
    print(f"--- ATOM MASK CREATED ---")
    print(f"Allowed {count} elements from the provided Universe list.")
    return mask

def load_system():
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
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading Checkpoint from {CHECKPOINT_PATH}")
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            if 'model_state_dict' in ckpt: transformer.load_state_dict(ckpt['model_state_dict'])
            else: transformer.load_state_dict(ckpt)
        except Exception as e:
            print(f"WARNING: Checkpoint load failed: {e}")
    else:
        print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}")

    return config, transformer

def main():
    print("--- STARTING NOVAGEN DISCOVERY ENGINE (FINAL) ---")
    
    config, transformer = load_system()
    reward_calc = RewardCalculator()
    
    # Initialize Log with new Columns
    if not os.path.exists(LOG_CSV_PATH):
        columns = [
            "Iteration", "Formula", "Density", "Reward", "Valid",
            "Initial_Energy", "Final_Energy", "Band_Gap", "Formation_Energy"
        ]
        pd.DataFrame(columns=columns).to_csv(LOG_CSV_PATH, index=False)

    # Setup PPO
    transformer_pretrain = copy.deepcopy(transformer)
    transformer_pretrain.eval()
    for p in transformer_pretrain.parameters(): p.requires_grad = False
        
    optimizer = optim.Adam(transformer.parameters(), lr=LR)
    
    _, logp_fn = make_loss_fn(config['n_max'], config['atom_types'], config['wyck_types'], config['Kx'], config['Kl'], transformer)
    ppo_loss_calc = make_ppo_loss_fn(logp_fn, CLIP_EPS, BETA)
    
    # APPLY ELEMENT MASK
    valid_elements_mask = get_atom_mask(config['atom_types'], DEVICE)
    # Expand for batching
    # atom_mask shape expected: (n_max, atom_types)
    atom_mask = valid_elements_mask.unsqueeze(0).expand(config['n_max'], -1)
    
    constraints = torch.arange(config['n_max'], device=DEVICE)
    global_baseline = -5.0

    print(f"\n--- BEGINNING {NUM_ITERATIONS} ITERATION RUN ---")

    for it in range(NUM_ITERATIONS):
        iter_start = time.time()
        
        # --- A. SAMPLE ---
        try:
            with torch.no_grad():
                XYZ, A, W, M, L_real = sample_crystal(
                    key=None, transformer=transformer, params=None,
                    n_max=config['n_max'], batchsize=BATCH_SIZE,
                    atom_types=config['atom_types'], wyck_types=config['wyck_types'],
                    Kx=config['Kx'], Kl=config['Kl'], g=SPACE_GROUP,
                    w_mask=None, atom_mask=atom_mask, top_p=1.0, temperature=TEMP, T1=TEMP,
                    constraints=constraints
                )
        except RuntimeError as e:
            print(f"Sampling Error: {e}")
            continue

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
        
        # Calculate Rewards (and fill stats)
        rewards_list = reward_calc.get_rewards(structures)
        rewards_tensor = torch.tensor(rewards_list, device=DEVICE, dtype=torch.float32)
        
        # Construct Log Data
        batch_logs = []
        stats = reward_calc.last_batch_stats # Get the rich data from reward.py
        
        for i, s in enumerate(structures):
            entry = {
                "Iteration": it+1,
                "Formula": stats[i].get('formula', "INVALID"),
                "Density": s.density if s else 0.0,
                "Reward": rewards_list[i],
                "Valid": stats[i].get('valid', False),
                "Initial_Energy": stats[i].get('initial_E'),
                "Final_Energy": stats[i].get('final_E'),
                "Band_Gap": stats[i].get('band_gap'),
                "Formation_Energy": stats[i].get('formation_E')
            }
            batch_logs.append(entry)
            
        pd.DataFrame(batch_logs).to_csv(LOG_CSV_PATH, mode='a', header=False, index=False)
        
        # Stats
        avg_reward = rewards_tensor.mean().item()
        max_reward = rewards_tensor.max().item()
        global_baseline = 0.9 * global_baseline + 0.1 * avg_reward 
        
        # Advantages
        raw_advantages = rewards_tensor - global_baseline
        if raw_advantages.std() > 1e-5:
            advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
        else:
            advantages = raw_advantages
        
        # --- D. UPDATE ---
        transformer.train()
        x_data = (G, L_norm, XYZ, A, W)
        
        total_loss = 0
        skipped = False
        
        for _ in range(PPO_EPOCHS):
            optimizer.zero_grad()
            loss, kl = ppo_loss_calc(transformer, x_data, old_logp, pretrain_logp, advantages)
            
            if torch.isnan(loss) or torch.abs(loss) > 20.0:
                skipped = True
                break
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            
        iter_time = time.time() - iter_start
        status_symbol = "✅" if max_reward > -4.0 else "❌"
        loss_str = f"{total_loss/PPO_EPOCHS:.4f}" if not skipped else "SKIP"
        
        print(f"Iter {it+1}/{NUM_ITERATIONS} [{status_symbol}] | "
              f"Reward: {avg_reward:.2f} (Max: {max_reward:.2f}) | "
              f"Loss: {loss_str} | Time: {iter_time:.1f}s")

    print("\n--- DISCOVERY RUN COMPLETE ---")
    torch.save(transformer.state_dict(), os.path.join(PROJECT_ROOT, "pretrained_model", "finetuned_semiconductor.pt"))
    print(f"Full Data Log: {LOG_CSV_PATH}")

if __name__ == "__main__":
    main()
