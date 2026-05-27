#!/bin/bash
# Phase 2 step 1: extract layer-20 residual activations from
# Qwen2.5-7B-Instruct at the `?` token and the final prompt token, for
# every question in the scaled wm-person-death sweep. ~3 min on 1× H200.
#
# Submit:  sbatch jobs/run_extract_acts.sh
# Output:  exp/iphr/pd_full_acts.npz

#SBATCH --job-name=iphr_extract_acts
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --output=%x/../exp/logs/%x_%j.out

export HF_HOME=${HF_HOME:-/workspace-vast/$USER/hf_cache}
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

source ${VENV:-/workspace-vast/$USER/envs/.venv}/bin/activate
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

mkdir -p exp/iphr exp/logs

srun python scripts/iphr_extract_acts.py
