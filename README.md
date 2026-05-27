# NLA-CoT — Probing IPHR with Natural Language Autoencoders on Qwen2.5-7B-Instruct

Pipeline for testing whether **Implicit Post-Hoc Rationalization (IPHR)** —
the behavioural phenomenon documented by Arcuschin et al. (2025) where a
model commits to a Yes/No answer before reasoning, then constructs a
plausible-looking CoT to justify it — is detectable as a *representational*
phenomenon at the residual-stream level of Qwen2.5-7B-Instruct, using the
released [Natural Language Autoencoder](https://transformer-circuits.pub/2026/nla/index.html)
checkpoints (Fraser-Taliente et al., 2026).

See `project_plan.md` for the full research plan.

## Layout

```
NLA_CoT/
├── README.md
├── LICENSE                      MIT
├── project_plan.md              full research plan + hypotheses
├── requirements.txt             pinned deps (cu128 torch)
├── .gitignore                   excludes data, caches, cloned support repos
├── scripts/                     all analysis + experiment code
│   ├── nla_explain.py           NLAExplainer: (prompt, position) → NLA explanation (Phase 0 smoke test)
│   ├── iphr_rollouts.py         Phase 1: batched IPHR rollouts via HF transformers
│   ├── iphr_pairs.py            Phase 1.2: form pairs, flag unfaithful
│   ├── iphr_inspect.py          Side-by-side rollout dumps for hand-labeling
│   ├── iphr_wrong_cots.py       Dump verbatim wrong-answer CoTs per template
│   ├── iphr_plots.py            Phase 1 cross-template figures
│   ├── iphr_plots_pd.py         Phase 1 scaled wm-person-death figures
│   ├── iphr_extract_acts.py     Phase 2 step 1: extract layer-20 activations
│   ├── iphr_probes.py           Phase 2 step 2: GT + Model probes, 10-fold CV
│   ├── iphr_probe_analysis.py   Phase 2 figures + subset analysis
│   ├── iphr_probe_cosines.py    Phase 2 follow-up: cosine sims between probes
│   ├── iphr_nla_explain.py      Phase 3: batched NLA decode of all activations
│   ├── iphr_text_probes.py      Phase 4a: TF-IDF probes on NLA explanations
│   ├── iphr_word_ratios.py      Phase 4 follow-up: word-frequency-ratio analysis
│   └── iphr_embed_probes.py     Phase 4b: MiniLM-embedding probes on explanations
├── jobs/                        sbatch templates (one per heavy step)
│   ├── run_smoke.sh             Phase 0
│   ├── run_iphr_phase1.sh       Phase 1 (4 templates, ~20 min)
│   ├── run_iphr_pd_full.sh      Phase 1 scaled wm-person-death (sbatch array of 2)
│   ├── run_extract_acts.sh      Phase 2: activation extraction
│   ├── run_nla_explain.sh       Phase 3: NLA decode
│   └── run_embed_probes.sh      Phase 4b: sentence-embedding probes
├── chainscope/                  gitignored — clone per setup instructions
├── nla/                         gitignored — clone per setup instructions
└── exp/                         gitignored — experiment outputs land here
```

## Setup

### 1. Clone supporting repos

```bash
cd <repo-root>
git clone https://github.com/jettjaniak/chainscope.git
git clone https://github.com/kitft/natural_language_autoencoders.git nla
```

### 2. Create a Python 3.11 venv and install pinned deps

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt
source .venv/bin/activate
```

`cu128` matches the cluster's NVIDIA driver (12.8). On a different driver,
swap the suffix in `requirements.txt` and the `--index-url`. The HF
checkpoints (`Qwen/Qwen2.5-7B-Instruct`, `kitft/nla-qwen2.5-7b-L20-av`,
`sentence-transformers/all-MiniLM-L6-v2`) download automatically on first
use.

### 3. (Optional) Point caches at fast shared storage

```bash
export HF_HOME=/path/to/shared/hf_cache
```

The cluster CLAUDE.md notes that without this, downloads land on
node-local disk and fill it.

## Running

Each Slurm job below writes its outputs under `exp/iphr/` (created on
demand). All Python scripts also accept the env vars `CHAINSCOPE_DATA`
(default `<repo>/chainscope/chainscope`), `CHAINSCOPE_QWEN_PATH` (default
`Qwen/Qwen2.5-7B-Instruct`), `NLA_COT_EXP_DIR` (default `<repo>/exp/iphr`).

### Phase 0 — smoke test the NLA decoding

```bash
sbatch jobs/run_smoke.sh
tail -F exp/logs/nla_smoke_*.out
```

Expected: coherent English explanations for four toy prompts. If outputs
are CJK, the NLA-injection path broke — see `nla/docs/inference.md` for
the failure-mode checklist.

### Phase 1 — IPHR rollouts on 4 bias-prone templates

```bash
sbatch jobs/run_iphr_phase1.sh     # 4,800 generations, ~20 min on 1× H200
python scripts/iphr_pairs.py       # form pairs, flag unfaithful
python scripts/iphr_plots.py       # fig1..fig5
```

Cross-template figures land in `exp/iphr/plots/`.

### Phase 1 scaled — wm-person-death deep dive

Runs as a 2-task sbatch array (variant 1 + variant 2 in parallel on two
GPUs); each task generates 8,000 rollouts in ~16–25 min.

```bash
sbatch jobs/run_iphr_pd_full.sh
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
sbatch jobs/run_extract_acts.sh    # 800 forward passes, ~3 min
python scripts/iphr_probes.py      # 10-fold CV, GT + Model × {?, last}
python scripts/iphr_probe_analysis.py    # fig8, fig9
python scripts/iphr_probe_cosines.py     # fig10: cosine sims between probes
```

### Phase 3 — NLA-decode every activation

```bash
sbatch jobs/run_nla_explain.sh     # 1,600 decodes, ~4 min on 1× H200
```

### Phase 4 — text probes on the NLA explanations

```bash
python scripts/iphr_text_probes.py        # TF-IDF probes, fig11, fig12
python scripts/iphr_word_ratios.py        # word_ratios.md
sbatch jobs/run_embed_probes.sh           # MiniLM-embedding probes, fig13
```

## Env vars used by the scripts

| Variable | Default | Purpose |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | HF download cache (set to shared NVMe on a cluster) |
| `CHAINSCOPE_DATA` | `<repo>/chainscope/chainscope` | chainscope dataset root |
| `CHAINSCOPE_QWEN_PATH` | `Qwen/Qwen2.5-7B-Instruct` | Base-model HF id OR local snapshot path |
| `NLA_COT_EXP_DIR` | `<repo>/exp/iphr` | Where outputs go |
| `VENV` | `/workspace-vast/$USER/envs/.venv` | venv path used by the sbatch templates |

## Reproducibility notes

- **Seeds.** Sampling of (X, Y) tuples in `iphr_rollouts.py` is controlled
  by `--seed`. Generation uses `do_sample=True` (HF default) and is not
  itself seeded; per-pair accuracy averages over `--rollouts` samples to
  estimate the model's bias.
- **Model.** Qwen2.5-7B-Instruct, snapshot
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- **NLA checkpoints.** `kitft/nla-qwen2.5-7b-L20-av` (Activation
  Verbalizer). Sidecar (`nla_meta.yaml`) is loaded at runtime; token IDs
  and scale factors are not hardcoded.
- **Prompt template.** Exact IPHR template from Arcuschin et al.,
  embedded as `IPHR_TEMPLATE` in `iphr_rollouts.py` and the activation
  extractor.
- **Sampling.** Temperature 0.7, top-p 0.9 (matching the IPHR paper).
- **Cross-validation.** All probes use 10-fold `GroupKFold` keyed on the
  `(x_name, y_name)` tuple so both phrasings of an IPHR pair always
  share a fold. Refitting includes the TF-IDF vectoriser in
  `iphr_text_probes.py`.

## Acknowledgements

- chainscope dataset and IPHR methodology — Arcuschin et al., 2025
  ([arXiv:2503.08679](https://arxiv.org/abs/2503.08679))
- NLA checkpoints + reference code — Fraser-Taliente, Kantamneni, Ong
  et al., 2026 ([Transformer Circuits](https://transformer-circuits.pub/2026/nla/index.html))
