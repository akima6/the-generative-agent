#!/bin/bash

# --- NOVAGEN Colab Setup Script (Final Robust Version) ---

# 1. Base Setup and Conda Initialization
# -------------------------------------
PROJECT_DIR=$(pwd)

# Install Conda (if not already installed)
if ! command -v conda &> /dev/null
then
    echo "Installing Miniconda..."
    wget -q -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash miniconda.sh -b -p /usr/local -f
    rm miniconda.sh
    export PATH="/usr/local/bin:$PATH"
    source /usr/local/etc/profile.d/conda.sh
fi

conda activate base
echo "Project root set to: $PROJECT_DIR"

# --- New: Terms of Service Fix (Critical) ---
echo "Accepting Conda Terms of Service for non-interactive mode..."
# This accepts the ToS for the default channels, which is mandatory now.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- Absolute Path Definitions (Crucial for Colab isolation) ---
mkdir -p $PROJECT_DIR/envs
ENV_B_PATH="$PROJECT_DIR/envs/matgl_env"
ENV_C_PATH="$PROJECT_DIR/envs/megnet_legacy_env"
PATH_FILE="$PROJECT_DIR/env_paths.txt"
echo "ENV_B_PATH=$ENV_B_PATH" > $PATH_FILE
echo "ENV_C_PATH=$ENV_C_PATH" >> $PATH_FILE
# ---------------------------------

# 2. Environment A (generative_agent / Base Environment)
echo "2. Configuring Environment A (Generative Agent/JAX/Base)..."
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install optax pandas pyyaml scipy pymatgen
pip install -e $PROJECT_DIR/CrystalFormer

# 3. Environment B (matgl_env) - Use -p to force local creation
echo "3. Creating Environment B ($ENV_B_PATH - MatGL/M3GNet)..."
# Use -p for absolute path and explicitly use conda-forge as a robust channel
conda create -p $ENV_B_PATH python=3.10 -c conda-forge -y
conda run -p $ENV_B_PATH pip install pymatgen torch numpy matgl ase

# 4. Environment C (megnet_legacy_env) - Use -p to force local creation
echo "4. Creating Environment C ($ENV_C_PATH - MEGNET/TensorFlow Legacy)..."
conda create -p $ENV_C_PATH python=3.8 -c conda-forge -y
conda run -p $ENV_C_PATH pip install numpy==1.23.5 pymatgen==2022.7.24 tensorflow-gpu==2.9.0 megnet==1.1.5

echo ""
echo "============================================="
echo "NOVAGEN SETUP COMPLETE. PATHS SAVED to $PATH_FILE"
echo "============================================="
