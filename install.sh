#!/bin/bash
set -e

echo "==== Neural Jacobian Field Setup ===="

# ---------- Python tooling ----------
echo "[1/6] Installing uv + NumPy compatibility"
pip install -U uv
uv pip install "numpy<2"

# ---------- PyTorch (CUDA 11.8) ----------
# echo "[2/6] Installing PyTorch CUDA 11.8"
# uv pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
#   --index-url https://download.pytorch.org/whl/cu118

# ---------- PyTorch (Auto CUDA Match) ----------
echo "[2/6] Installing PyTorch (auto CUDA detection)"
uv pip install torch torchvision

# ---------- Nerfstudio + TinyCUDANN ----------
echo "[3/6] Installing Nerfstudio + tiny-cuda-nn"
uv pip install ninja
uv pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
uv pip install git+https://github.com/sizhe-li/nerfstudio.git

# ---------- MuJoCo Simulator ----------
echo "[4/6] Installing MuJoCo simulator"
git submodule update --init --recursive
cd mujoco-phys-sim/phys_sim
uv pip install -r requirements.txt
pip install -e .
cd ../..

# ---------- Jacobian Codebase ----------
echo "[5/6] Installing Neural Jacobian Field packages"
cd project
uv pip install -r requirements.txt
pip install -e .
cd ..

# ---------- Sanity Check ----------
echo "[6/6] Running import sanity check"
python - <<EOF
import torch
import jacobian
import neural_jacobian_field
import raft_wrapper
print("✅ All core packages imported successfully")
EOF

echo "==== Setup Complete ✅ ===="
