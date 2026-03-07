#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source ./setup_flash3dgs.sh
#
# If you run it with "bash ./setup_flash3dgs.sh", installation still works,
# but conda environment activation cannot persist in your current shell.

ENV_NAME="flash3dgs"
PYTHON_VERSION="3.10"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/examples"

is_sourced=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  is_sourced=1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found. Please install Miniconda/Anaconda first."
  return_or_exit=0
  if [[ ${is_sourced} -eq 1 ]]; then
    return_or_exit=1
  fi
  if [[ ${return_or_exit} -eq 1 ]]; then
    return 1
  else
    exit 1
  fi
fi

# Ensure conda activate works inside script
eval "$(conda shell.bash hook)"

echo "[1/6] Installing OS packages (requires sudo)..."
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git wget build-essential gcc-11 g++-11
else
  apt-get update
  apt-get install -y git wget build-essential gcc-11 g++-11
fi

echo "[2/6] Creating conda environment (${ENV_NAME}) if needed..."
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" -y python="${PYTHON_VERSION}"
fi

echo "[3/6] Activating conda environment (${ENV_NAME})..."
conda activate "${ENV_NAME}"

echo "[4/6] Installing Python/CUDA dependencies..."
python -m pip install -U pip
python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
python -m pip install --no-cache-dir "numpy==1.26.4" "setuptools==69.5.1" "wheel<0.43" ninja packaging

echo "[5/6] Installing FLASH3DGS from source..."
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
python -m pip install -e "${SCRIPT_DIR}" --no-build-isolation

echo "[6/6] Installing examples requirements and downloading dataset..."
python -m pip install -r "${EXAMPLES_DIR}/requirements.txt" --no-build-isolation
(
  cd "${EXAMPLES_DIR}"
  python datasets/download_dataset.py
)

echo ""
echo "Setup complete."
if [[ ${is_sourced} -eq 1 ]]; then
  echo "Conda environment '${ENV_NAME}' is active in this shell."
else
  echo "Conda environment activation cannot persist from a child shell."
  echo "Run this to activate it now: conda activate ${ENV_NAME}"
  echo "Tip: use 'source ./setup_flash3dgs.sh' next time."
fi
