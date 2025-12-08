#!/bin/bash

# --- NOVAGEN Colab Persistent Setup Script ---

# 1. Mount Google Drive (Manual step in Colab cell)
# The user must execute 'from google.colab import drive; drive.mount("/content/drive")' first!

# 2. Define Project and Persistence Paths
PROJECT_DIR=$(pwd)
# We store persistent files in a specific folder on Google Drive
PERSISTENT_DIR="/content/drive/MyDrive/NOVAGEN_ENV"
CONDA_INSTALL_DIR="$PERSISTENT_DIR/miniconda3"
PROJECT_ENVS_DIR="$PROJECT_DIR/envs"
PATH_FILE="$PROJECT_DIR/env_paths.txt"

# 3. Conda Installation Check (If Conda doesn't exist on Drive, install it)
# --------------------------------------------------------------------------
if [ ! -d "$CONDA_INSTALL_DIR" ]; then
    echo "Conda not found on Drive. Performing FIRST-TIME INSTALL..."
    mkdir -p "$PERSISTENT_DIR"
    
    # 3a. Install Miniconda directly to the persistent directory
    wget -q -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash miniconda.sh -b -p "$CONDA_INSTALL_DIR" -f
    rm miniconda.sh
    
    # 3b. Initialize Conda (must be done after every reboot/install)
    export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
    $CONDA_INSTALL_DIR/bin/conda init bash
    # Source the new environment setup
    source $CONDA_INSTALL_DIR/etc/profile.d/conda.sh

    # 3c. Accept ToS (Crucial)
    echo "Accepting Conda Terms of Service..."
    $CONDA_INSTALL_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_INSTALL_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

    # 3d. Create the environment folders inside the project
    mkdir -p $PROJECT_ENVS_DIR
    
else
    # 4. Conda Found on Drive (REUSE INSTALLATION)
    echo "Conda found on Drive. Reusing installation..."
    export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
    source $CONDA_INSTALL_DIR/etc/profile.d/conda.sh
    
    # Check for existing envs inside the project and link them
    if [ -d "$PROJECT_ENVS_DIR" ]; then rm -rf "$PROJECT_ENVS_DIR"; fi
    ln -s "$PERSISTENT_DIR/envs" "$PROJECT_ENVS_DIR"
    
    # If the env_paths file exists, we are done with environment creation.
    if [ -f "$PATH_FILE" ]; then
        echo "Environments linked. Skipping creation steps."
        # Jump directly to step 5 (Base Env and CrystalFormer install)
        goto :finalize_base_env
    fi
fi

# 5. Environment Creation (Only runs once)
# ----------------------------------------
echo "Creating persistent environments..."
# Define specific paths for the envs *inside* the persistent directory
ENV_B_PATH="$PERSISTENT_DIR/envs/matgl_env"
ENV_C_PATH="$PERSISTENT_DIR/envs/megnet_legacy_env"
mkdir -p "$PERSISTENT_DIR/envs"

# Write paths for trainer.py
echo "ENV_B_PATH=$ENV_B_PATH" > $PATH_FILE
echo "ENV_C_PATH=$ENV_C_PATH" >> $PATH_FILE

# 5a. Environment B (matgl_env)
echo "Creating matgl_env..."
# Use -p to force creation in the persistent drive folder
$CONDA_INSTALL_DIR/bin/conda create -p $ENV_B_PATH python=3.10 --file matgl_env_deps.txt -c conda-forge -y

# 5b. Environment C (megnet_legacy_env)
echo "Creating megnet_legacy_env..."
# Use -p to force creation in the persistent drive folder
$CONDA_INSTALL_DIR/bin/conda create -p $ENV_C_PATH python=3.8 --file megnet_legacy_env_deps.txt -c conda-forge -y

# Link the created envs folder back to the project dir
ln -s "$PERSISTENT_DIR/envs" "$PROJECT_ENVS_DIR"

:finalize_base_env
# 6. Environment A (generative_agent / Base Environment)
# Always install JAX and CrystalFormer into the Colab base env
echo "Configuring Base Environment (JAX/CrystalFormer)..."
pip install -r generative_agent_deps.txt
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e $PROJECT_DIR/CrystalFormer

echo ""
echo "============================================="
echo "NOVAGEN PERSISTENT SETUP COMPLETE"
echo "============================================="
