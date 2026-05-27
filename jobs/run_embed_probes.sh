#!/bin/bash
# Phase 4: embed every NLA explanation with all-MiniLM-L6-v2 and train
# probes on the dense embeddings (mirrors the TF-IDF text-probe setup).
# ~3 min on 1× H200 (most of which is downloading the embedder first time).
#
# Submit:  sbatch jobs/run_embed_probes.sh
# Output:  exp/iphr/pd_full_nla_embeddings.npz
#          exp/iphr/embed_probe_results.json
#          exp/iphr/plots/fig13_act_vs_text_vs_embed.png

#SBATCH --job-name=iphr_embed_probes
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:15:00
#SBATCH --output=%x/../exp/logs/%x_%j.out

export HF_HOME=${HF_HOME:-/workspace-vast/$USER/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

source ${VENV:-/workspace-vast/$USER/envs/.venv}/bin/activate
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

mkdir -p exp/iphr exp/logs

srun python scripts/iphr_embed_probes.py
