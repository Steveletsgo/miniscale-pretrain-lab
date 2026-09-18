#!/usr/bin/env bash

# Source this file inside the Ubuntu-AI WSL distribution before development:
#   source /mnt/d/AI-Projects/miniscale-pretrain-lab/scripts/activate_wsl_env.sh

export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_CUDA_ARCH_LIST="8.9"

# Ubuntu-AI itself is stored under D:\WSL, so these Linux cache paths also
# consume D: rather than the Windows system drive.
export PIP_CACHE_DIR="${HOME}/.cache/pip"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TRITON_CACHE_DIR="${HOME}/.cache/triton"

mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}"
