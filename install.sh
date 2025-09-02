#!/bin/bash
# Setup script for Neural Jacobian Field
echo "Starting Setup..."
pip install uv

# 2. Install NeRFStudio
echo "Installing NeRFStudio"
uv pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install --yes -c "nvidia/label/cuda-11.8.0" cuda-toolkit
uv pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
uv pip install git+https://github.com/sizhe-li/nerfstudio.git

# 3. Install Simulator
echo "Installing MuJoCo Simulator"
git submodule update --init --recursive
cd mujoco-phys-sim/phys_sim
uv pip install -r requirements.txt
pip install -e .
cd ../..

# 4. Install Jacobian Fields Codebase
echo "Installing Jacobian Fields Codebase"
cd project
uv pip install -r requirements.txt
pip install -e .
cd ..

echo "Setup Complete!"