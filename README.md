# NLA-CoT — Probing IPHR with Natural Language Autoencoders on Qwen2.5-7B-Instruct

Pipeline for testing whether **Implicit Post-Hoc Rationalization (IPHR)** —
the behavioural phenomenon documented by Arcuschin et al. (2025) where a
model commits to a Yes/No answer before reasoning, then constructs a
plausible-looking chain-of-thought to justify it — is detectable as a
*representational* phenomenon at the residual-stream level of
Qwen2.5-7B-Instruct, using the released
[Natural Language Autoencoder](https://transformer-circuits.pub/2026/nla/index.html)
checkpoints (Fraser-Taliente et al., 2026).

See `project_plan.md` for the full research plan.

## Requirements

- A single NVIDIA GPU with ≥ 32 GB memory (an H100, A100-80GB, or
  RTX 6000 Ada all work; we ran on H200s). Phase 0 / Phase 1 / Phase 2
  load Qwen2.5-7B in bf16 (~14 GB) plus a second 7B model for some phases
  (~28 GB peak); Phase 3 uses Qwen-7B alone; Phase 4 needs only a small
  embedder (≤ 4 GB).
- Python 3.11.
- ~60 GB free disk space for model weights + experiment outputs.
- Network access (for the first run, to download the HuggingFace
  checkpoints).

## Setup

### 1. Clone this repo and the two supporting repos

```bash
git clone <this repo url> NLA_CoT
cd NLA_CoT
git clone https://github.com/jettjaniak/chainscope.git
git clone https://github.com/kitft/natural_language_autoencoders.git nla
```

`chainscope` is the dataset and `nla` is the reference NLA implementation
(we only read its docs; all pipeline code lives in `scripts/`).

### 2. Create a Python 3.11 venv and install dependencies

`requirements.txt` pins torch to the CUDA 12.8 build. **You probably
need to change the suffix** to match your driver — check with
`nvidia-smi`, look at the CUDA version it reports, and pick the matching
[PyTorch wheel index](https://pytorch.org/get-started/locally/) (e.g.,
`cu121`, `cu124`, `cu128`).

Using [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
               --extra-index-url https://pypi.org/simple \
               -r requirements.txt
```

Or with stock `pip`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 \
            --extra-index-url https://pypi.org/simple \
            -r requirements.txt
```

The HuggingFace checkpoints (`Qwen/Qwen2.5-7B-Instruct`,
`kitft/nla-qwen2.5-7b-L20-av`, `sentence-transformers/all-MiniLM-L6-v2`)
will be downloaded automatically on first use. If you want to control
where they go, set `HF_HOME=/path/to/your/cache` before running.

## Running the pipeline

Every Python script in `scripts/` is a standalone CLI; `--help` on each
shows its flags. Outputs go to `exp/iphr/` by default (override with
`NLA_COT_EXP_DIR`).

### Phase 0 — smoke test the NLA decoding (~3 min on 1 GPU)

```bash
python scripts/nla_explain.py --temperature 0.7 --max-new-tokens 200
```

Expected: coherent English explanations for four toy prompts. If outputs
are CJK characters, the NLA injection path broke — see
`nla/docs/inference.md` for the failure-mode checklist.

### Phase 1 — IPHR rollouts on 4 bias-prone templates

```bash
# ~20 min on a single H200 — generates 4,800 CoT rollouts
python scripts/iphr_rollouts.py \
    --templates wm-nyt-pubdate wm-person-birth wm-person-death wm-us-natural-long \
    --buckets non-ambiguous-hard \
    --n-pairs 30 --rollouts 10 --batch-size 96 \
    --temperature 0.7 --top-p 0.9 \
    --out exp/iphr/rollouts.jsonl

# Form pairs and flag unfaithful ones (CPU, < 30 s)
python scripts/iphr_pairs.py

# Plot the 5 cross-template figures (CPU)
python scripts/iphr_plots.py        # fig1..fig5
```

### Phase 1 scaled — deep dive on wm-person-death

```bash
# Each variant: ~16 min on a single H200; run them serially or in
# parallel on two GPUs.
python scripts/iphr_rollouts.py \
    --templates wm-person-death --buckets non-ambiguous-hard \
    --n-pairs 100 --rollouts 20 --batch-size 96 --seed 1 \
    --out exp/iphr/pd_full_v1.jsonl

python scripts/iphr_rollouts.py \
    --templates wm-person-death --buckets non-ambiguous-hard-2 \
    --n-pairs 100 --rollouts 20 --batch-size 96 --seed 2 \
    --out exp/iphr/pd_full_v2.jsonl

cat exp/iphr/pd_full_v1.jsonl exp/iphr/pd_full_v2.jsonl \
    > exp/iphr/pd_full_combined.jsonl

python scripts/iphr_pairs.py \
    --rollouts exp/iphr/pd_full_combined.jsonl \
    --out-pairs exp/iphr/pd_full_pairs.jsonl \
    --out-unfaithful exp/iphr/pd_full_unfaithful.jsonl

python scripts/iphr_plots_pd.py     # fig6, fig7
python scripts/iphr_wrong_cots.py \
    --template wm-person-death --gap 0.20 \
    --out exp/iphr/wrong_cots_person-death.md
```

### Phase 2 — linear probes for pre-commitment

```bash
# ~3 min on 1 GPU: 800 forward passes of Qwen2.5-7B
python scripts/iphr_extract_acts.py

# CPU: ~30 s for the four probes
python scripts/iphr_probes.py
python scripts/iphr_probe_analysis.py    # fig8, fig9
python scripts/iphr_probe_cosines.py     # fig10: pairwise cosine sims
```

### Phase 3 — NLA-decode every activation (~4 min on 1 GPU)

```bash
python scripts/iphr_nla_explain.py --batch-size 24 --max-new-tokens 200
```

### Phase 4 — text probes on the NLA explanations

```bash
# TF-IDF probes (CPU, ~10 s)
python scripts/iphr_text_probes.py        # fig11, fig12

# Word-frequency-ratio analysis (CPU)
python scripts/iphr_word_ratios.py        # word_ratios.md

# MiniLM-embedding probes (~3 min on 1 GPU, mostly model download)
python scripts/iphr_embed_probes.py       # fig13
```

## Configuration

All scripts read these environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | Where HuggingFace caches checkpoints |
| `CHAINSCOPE_DATA` | `<repo>/chainscope/chainscope` | Path to chainscope dataset root |
| `CHAINSCOPE_QWEN_PATH` | `Qwen/Qwen2.5-7B-Instruct` | Base model — HF id or a local snapshot path |
| `NLA_COT_EXP_DIR` | `<repo>/exp/iphr` | Output directory for all results |

## Layout

```
NLA_CoT/
├── README.md                this file
├── LICENSE                  MIT
├── project_plan.md          full research plan + hypotheses
├── requirements.txt         pinned deps (edit cu128 to match your driver)
├── scripts/                 all analysis + experiment code
│   ├── nla_explain.py       NLAExplainer: (prompt, position) → NLA explanation
│   ├── iphr_rollouts.py     Phase 1: batched IPHR rollouts
│   ├── iphr_pairs.py        Phase 1.2: form pairs, flag unfaithful
│   ├── iphr_inspect.py      Side-by-side rollout dumps for hand-labeling
│   ├── iphr_wrong_cots.py   Dump wrong-answer CoTs per template
│   ├── iphr_plots.py        Phase 1 cross-template figures
│   ├── iphr_plots_pd.py     Phase 1 scaled wm-person-death figures
│   ├── iphr_extract_acts.py Phase 2 step 1: extract layer-20 activations
│   ├── iphr_probes.py       Phase 2 step 2: GT + Model probes, 10-fold CV
│   ├── iphr_probe_analysis.py     Phase 2 figures + subset analysis
│   ├── iphr_probe_cosines.py      Phase 2 follow-up: cosine sims
│   ├── iphr_nla_explain.py  Phase 3: batched NLA decode of all activations
│   ├── iphr_text_probes.py  Phase 4a: TF-IDF probes on NLA explanations
│   ├── iphr_word_ratios.py  Phase 4 follow-up: word-frequency-ratio analysis
│   └── iphr_embed_probes.py Phase 4b: MiniLM-embedding probes on explanations
├── jobs/                    sbatch templates (Slurm-specific; optional)
├── chainscope/              gitignored — clone per setup instructions
├── nla/                     gitignored — clone per setup instructions
└── exp/                     gitignored — experiment outputs land here
```

## Reproducibility notes

- **Seeds.** Sampling of (X, Y) tuples in `iphr_rollouts.py` is
  controlled by `--seed`. Generation uses `do_sample=True` and is not
  itself seeded; per-pair accuracy averages over `--rollouts` samples
  to estimate the model's bias.
- **Model.** Qwen2.5-7B-Instruct, snapshot
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- **NLA checkpoints.** `kitft/nla-qwen2.5-7b-L20-av` (Activation
  Verbalizer). The sidecar (`nla_meta.yaml`) is loaded at runtime;
  token IDs and scale factors are never hardcoded in our code.
- **Prompt template.** Exact IPHR template from Arcuschin et al.,
  embedded as `IPHR_TEMPLATE` in `iphr_rollouts.py` and the activation
  extractor.
- **Sampling.** Temperature 0.7, top-p 0.9 (matching the IPHR paper).
- **Cross-validation.** All probes use 10-fold `GroupKFold` keyed on
  `(x_name, y_name)` so both phrasings of an IPHR pair always share a
  fold. In `iphr_text_probes.py` the TF-IDF vectoriser is refit on
  each fold's training set to avoid entity-name leakage.

## Acknowledgements

- chainscope dataset and IPHR methodology — Arcuschin et al., 2025
  ([arXiv:2503.08679](https://arxiv.org/abs/2503.08679))
- NLA checkpoints + reference code — Fraser-Taliente, Kantamneni, Ong
  et al., 2026
  ([Transformer Circuits](https://transformer-circuits.pub/2026/nla/index.html))
