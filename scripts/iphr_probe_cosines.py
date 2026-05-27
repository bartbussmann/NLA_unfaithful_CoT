"""Compute cosine similarities between the four probe weight vectors.

For each (target, position) we fit a single logistic regression on the FULL
800-question dataset (with the same drop_mask as the CV pipeline) and pull
the 3584-dim coefficient vector. We then compute the 4×4 cosine-similarity
matrix.

Why full-data rather than fold-averaged: a single full-data fit gives one
clean direction per probe; averaging fold weights would underweight features
that flip sign across folds. For interpretation of "what direction in
residual stream does this probe care about", the full-data fit is more
natural.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from iphr_probes import build_labels, load_acts, load_jsonl

EXP = Path(os.environ.get("NLA_COT_EXP_DIR",
                          Path(__file__).resolve().parent.parent / "exp" / "iphr"))


def main() -> int:
    acts = load_acts(EXP / "pd_full_acts.npz")
    labels = build_labels(EXP / "pd_full_combined.jsonl", acts,
                          EXP / "pd_full_pairs.jsonl")
    print(f"loaded {len(acts['qid'])} questions")

    fits = {}
    for pos_name, X_key in [("qmark", "acts_qmark"), ("last", "acts_last")]:
        X = acts[X_key]
        for target_name, y, drop in [
            ("gt",    labels["gt_label"],       None),
            ("model", labels["model_majority"], labels["drop"]),
        ]:
            key = f"{target_name}_at_{pos_name}"
            mask = np.ones(len(y), dtype=bool) if drop is None else ~drop
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf.fit(X[mask], y[mask])
            w = clf.coef_[0]  # [d_model]
            fits[key] = w
            print(f"  {key:18s} w.shape={w.shape} ||w||={np.linalg.norm(w):.3f}")

    keys = ["gt_at_qmark", "gt_at_last", "model_at_qmark", "model_at_last"]
    n = len(keys)
    cos = np.zeros((n, n))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            wa, wb = fits[a], fits[b]
            cos[i, j] = float(np.dot(wa, wb) /
                              (np.linalg.norm(wa) * np.linalg.norm(wb)))

    # Print
    print("\nCosine-similarity matrix:")
    head = " " * 18 + " ".join(f"{k:>16s}" for k in keys)
    print(head)
    for i, k in enumerate(keys):
        row = "  ".join(f"{cos[i, j]:+.3f}".rjust(16) for j in range(n))
        print(f"{k:18s} {row}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=10)
    ax.set_yticklabels(keys, fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.annotate(f"{cos[i, j]:+.3f}", (j, i), ha="center", va="center",
                        color="white" if abs(cos[i, j]) > 0.5 else "black",
                        fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label("cosine similarity of probe weight vectors")
    ax.set_title("Cosine similarities between the four probe directions\n"
                 "(layer-20 residual stream, Qwen2.5-7B-Instruct, "
                 "wm-person-death)")
    fig.tight_layout()
    out = EXP / "plots" / "fig10_probe_cosines.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")

    # Save numerical results
    (EXP / "probe_cosines.json").write_text(json.dumps(
        {"keys": keys, "cosine_matrix": cos.tolist(),
         "norms": {k: float(np.linalg.norm(fits[k])) for k in keys}},
        indent=2))
    print(f"wrote {EXP / 'probe_cosines.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
