#!/bin/bash
# Scaled-up wm-person-death sweep: ALL non-ambiguous-hard pairs (variants 1+2)
# = 400 pairs × 2 phrasings × 20 rollouts = 16,000 generations per GPU.
#
# This script runs the two variants in PARALLEL on two GPUs via a sbatch array.
# Submit:  sbatch jobs/run_iphr_pd_full.sh
# Each array task writes a separate JSONL; merge after.

#SBATCH --job-name=iphr_pd_full
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --array=1-2
#SBATCH --output=%x/../exp/logs/%x_%A_%a.out

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

# Variant 1 -> suffix "non-ambiguous-hard"
# Variant 2 -> suffix "non-ambiguous-hard-2"
if [ "$SLURM_ARRAY_TASK_ID" = "1" ]; then
    SUFFIX="non-ambiguous-hard"
    OUT="exp/iphr/pd_full_v1.jsonl"
    SEED=1
else
    SUFFIX="non-ambiguous-hard-2"
    OUT="exp/iphr/pd_full_v2.jsonl"
    SEED=2
fi

srun python scripts/iphr_rollouts.py \
    --templates wm-person-death \
    --buckets "$SUFFIX" \
    --n-pairs 100 \
    --rollouts 20 \
    --batch-size 96 \
    --max-new-tokens 400 \
    --temperature 0.7 --top-p 0.9 \
    --seed $SEED \
    --out $OUT
