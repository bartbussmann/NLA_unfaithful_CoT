#!/bin/bash
# Phase 3: NLA-decode every layer-20 activation at the ? and last positions
# for all 800 wm-person-death questions. Batched generation, ~4 min on 1× H200.
#
# Submit:  sbatch jobs/run_nla_explain.sh
# Output:  exp/iphr/pd_full_nla.jsonl  (1600 rows)

#SBATCH --job-name=iphr_nla_explain
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0:45:00
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

srun python scripts/iphr_nla_explain.py --batch-size 24 --max-new-tokens 200
