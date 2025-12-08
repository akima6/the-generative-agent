#!/bin/bash

# --- NOVAGEN Colab Persistent Setup Script (Final YAML Version) ---

# 1. Mount Google Drive is a manual step (done in Colab Cell 1)

# 2. Define Project and Persistence Paths
PROJECT_DIR=$(pwd)
# We store persistent files in a specific folder on Google Drive
PERSISTENT_DIR="/content/drive/MyDrive/NOVAGEN_ENV"
CONDA_INSTALL_DIR="$PERSISTENT_DIR/miniconda3"
PROJECT_ENVS_DIR="$PROJECT_DIR/envs"
PATH_FILE="$PROJECT_DIR/env_paths.txt"

echo "Project root set to: $PROJECT_DIR"

# 3. Conda Installation Check (If Conda doesn't exist on Drive, install it)
# --------------------------------------------------------------------------
if [ ! -d "$CONDA_INSTALL_DIR" ]; then
    echo "Conda not found on Drive. Performing FIRST-TIME INSTALL..."
    mkdir -p "$PERSISTENT_DIR"
    
    # 3a. Install Miniconda directly to the persistent directory
    wget -q -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash miniconda.sh -b -p "$CONDA_INSTALL_DIR" -f
    rm miniconda.sh
    
    # 3b. Initialize Conda and accept ToS (CRUCIAL)
    export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
    $CONDA_INSTALL_DIR/bin/conda init bash
    source $CONDA_INSTALL_DIR/etc/profile.d/conda.sh

    echo "Accepting Conda Terms of Service..."
    $CONDA_INSTALL_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_INSTALL_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    
    # Create the environment folders inside the persistent directory
    mkdir -p "$PERSISTENT_DIR/envs"
    
else
    # 4. Conda Found on Drive (REUSE INSTALLATION)
    echo "Conda found on Drive. Reusing installation..."
    export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
    source $CONDA_INSTALL_DIR/etc/profile.d/conda.sh
    
    # Link the persistent envs folder to the project's 'envs' directory
    if [ ! -L "$PROJECT_ENVS_DIR" ]; then 
        if [ -d "$PROJECT_ENVS_DIR" ]; then rm -rf "$PROJECT_ENVS_DIR"; fi
        ln -s "$PERSISTENT_DIR/envs" "$PROJECT_ENVS_DIR"
    fi
    
    # If the env_paths file exists, assume creation is done
    if [ -f "$PATH_FILE" ]; then
        echo "Environments linked and paths loaded. Skipping full creation steps."
        # Jumps to base env install, which must happen every session
        goto :finalize_base_env 
    fi
fi

# 5. Environment Creation (Only runs once on first session)
# ----------------------------------------
echo "Creating persistent environments from YAML files..."

# Define specific paths for the envs *inside* the persistent directory
ENV_B_PATH="$PERSISTENT_DIR/envs/matgl_env"
ENV_C_PATH="$PERSISTENT_DIR/envs/megnet_legacy_env"

# Write paths for trainer.py to read
echo "ENV_B_PATH=$ENV_B_PATH" > $PATH_FILE
echo "ENV_C_PATH=$ENV_C_PATH" >> $PATH_FILE

# 5a. Environment B (matgl_env)
echo "Creating matgl_env from matgl_env.yaml..."
$CONDA_INSTALL_DIR/bin/conda env create -p $ENV_B_PATH -f matgl_env.yaml -y

# 5b. Environment C (megnet_legacy_env)
echo "Creating megnet_legacy_env from megnet_legacy_env.yaml (CRITICAL: Requires Py 3.8/TF 2.9)..."
$CONDA_INSTALL_DIR/bin/conda env create -p $ENV_C_PATH -f megnet_legacy_env.yaml -y

# Link the created envs folder back to the project dir (for the first time)
if [ ! -L "$PROJECT_ENVS_DIR" ]; then 
    ln -s "$PERSISTENT_DIR/envs" "$PROJECT_ENVS_DIR"
fi


:finalize_base_env
# 6. Environment A (generative_agent / Base Environment)
# Always install JAX and CrystalFormer into the Colab base env (Fastest step)
echo "Configuring Base Environment (JAX/CrystalFormer)..."
# Use the base env YAML to install all dependencies for the generative agent
$CONDA_INSTALL_DIR/bin/conda install -y --file generative_agent.yaml 
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e $PROJECT_DIR/CrystalFormer

echo ""
echo "============================================="
echo "NOVAGEN PERSISTENT SETUP COMPLETE"
echo "============================================="
