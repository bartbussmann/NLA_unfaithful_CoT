#!/bin/bash
# Phase 1: IPHR rollouts on Qwen2.5-7B-Instruct for 4 bias-prone templates.
# 30 (X,Y) tuples per pair-class × 2 phrasings × 2 classes × 4 templates
# × 10 rollouts = 4800 generations, ~20 min on 1× H200.
#
# Submit:  sbatch jobs/run_iphr_phase1.sh

#SBATCH --job-name=iphr_phase1
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
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

mkdir -p exp/iphr

srun python scripts/iphr_rollouts.py \
    --templates wm-nyt-pubdate wm-person-birth wm-person-death wm-us-natural-long \
    --buckets non-ambiguous-hard \
    --n-pairs 30 \
    --rollouts 10 \
    --batch-size 96 \
    --max-new-tokens 400 \
    --temperature 0.7 --top-p 0.9 \
    --seed 0 \
    --out exp/iphr/rollouts.jsonl
