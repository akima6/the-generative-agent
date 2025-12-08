#!/bin/bash

# --- NOVAGEN Colab Setup Script ---
# This script configures the 3-Environment Subprocess Architecture on a remote Linux/Colab machine.

# 1. Base Setup and Conda Initialization
# -------------------------------------
echo "1. Initializing Conda and Cloning Repository..."

# Install Conda (if not already installed, which is common on Colab)
if ! command -v conda &> /dev/null
then
    echo "Conda not found. Installing Miniconda..."
    wget -q -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash miniconda.sh -b -p /usr/local -f
    rm miniconda.sh
    # Re-initialize Conda in the current shell session
    export PATH="/usr/local/bin:$PATH"
    conda init bash
    # Source the new environment setup (essential for 'conda activate')
    source /usr/local/etc/profile.d/conda.sh
fi

# Ensure all future commands run through the conda base environment
conda activate base

# NOTE: Assuming you will clone the repository *before* running this script, 
# and that this script is executed from the project root.
PROJECT_DIR=$(pwd)
echo "Project root set to: $PROJECT_DIR"

# 2. Environment A (generative_agent / Base Environment)
# ---------------------------------------------------
echo "2. Configuring Environment A (Generative Agent/JAX/Base)..."
# We assume the base environment already has python 3.10 and necessary build tools
# We install JAX, Optax, and required dependencies here.
# NOTE: The JAX-CUDA version must match the Colab GPU's CUDA version.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install optax pandas pyyaml scipy
pip install pymatgen  # Needed by bridge.py

# Install CrystalFormer (assuming local setup is via editable install)
pip install -e $PROJECT_DIR/CrystalFormer

echo "Environment A (Base) setup complete."

# 3. Environment B (matgl_env) - The MatGL Relaxer
# ------------------------------------------------
ENV_B="matgl_env"
echo "3. Creating Environment B ($ENV_B - MatGL/M3GNet)..."
conda create -n $ENV_B python=3.10 -y
conda run -n $ENV_B pip install pymatgen torch numpy # PyTorch for matgl
conda run -n $ENV_B pip install matgl ase

# 4. Environment C (megnet_legacy_env) - The Legacy Band Gap Predictor
# -----------------------------------------------------------------
ENV_C="megnet_legacy_env"
echo "4. Creating Environment C ($ENV_C - MEGNET/TensorFlow Legacy)..."
conda create -n $ENV_C python=3.8 -y

# NOTE: Strict legacy requirements are enforced here.
conda run -n $ENV_C pip install numpy==1.23.5 pymatgen==2022.7.24
# TensorFlow 2.9 (needed for megnet's old models) and megnet
conda run -n $ENV_C pip install tensorflow-gpu==2.9.0 megnet==1.1.5

echo "Environment C ($ENV_C) setup complete."

# 5. Final Check and Instructions
# -------------------------------
echo ""
echo "============================================="
echo "NOVAGEN SETUP COMPLETE."
echo "============================================="
echo "You can now run the PPO trainer using:"
echo "python $PROJECT_DIR/generative_agent/ppo_trainer/trainer.py"
echo ""
echo "Verify your Python paths in the Colab cell before running the trainer."