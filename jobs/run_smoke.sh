#!/bin/bash
# Phase 0 smoke test: load Qwen2.5-7B-Instruct + the NLA AV, extract a
# layer-20 activation on a few toy prompts, run NLA verbalization, expect
# coherent English explanations.
#
# Submit:  sbatch jobs/run_smoke.sh
# Tail:    tail -F exp/logs/nla_smoke_<jobid>.out

#SBATCH --job-name=nla_smoke
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --output=%x/../exp/logs/%x_%j.out

# Always set HF_HOME to writable shared storage — default ~/.cache/huggingface
# fills node-local disk on whichever node Slurm picks.
export HF_HOME=${HF_HOME:-/workspace-vast/$USER/hf_cache}
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# Activate your venv. Edit this path if your venv lives elsewhere.
source ${VENV:-/workspace-vast/$USER/envs/.venv}/bin/activate

# Repo root: parent of this script's directory
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

srun python scripts/nla_explain.py --temperature 0.7 --max-new-tokens 200
