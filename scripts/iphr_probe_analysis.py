"""Phase 2 analysis + plots.

Reads probe_results.json + probe_predictions.npz and produces:

  fig8_probe_accuracy.png         — bar chart of OOF accuracy per
                                     (probe target, position, subset)
  fig9_pair_pre_commitment.png    — per-pair scatter of model-probe's
                                     prediction on phrasing-A vs phrasing-B.
                                     Pre-commitment shows up as diagonal
                                     agreement on unfaithful pairs (where
                                     the two phrasings have OPPOSITE
                                     ground-truth answers).
  probe_analysis.md               — short markdown writeup with the key
                                     numbers tabulated for the report.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(os.environ.get("NLA_COT_EXP_DIR",
                              Path(__file__).resolve().parent.parent / "exp" / "iphr"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(EXP_DIR / "probe_results.json"))
    parser.add_argument("--preds",   default=str(EXP_DIR / "probe_predictions.npz"))
    parser.add_argument("--out-md",  default=str(EXP_DIR / "probe_analysis.md"))
    parser.add_argument("--outdir",  default=str(EXP_DIR / "plots"))
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    results = json.loads(Path(args.results).read_text())
    preds = np.load(args.preds, allow_pickle=False)
    keys = ["gt_at_qmark", "gt_at_last", "model_at_qmark", "model_at_last"]
    subsets_all = ["overall", "faithful_pairs", "mid_pairs", "unfaithful_pairs",
                   "model_majority_correct", "model_majority_incorrect"]
    # Subsets used in the bar chart (drop mid_pairs for clarity — the
    # interesting comparison is faithful vs unfaithful).
    subsets = ["overall", "faithful_pairs", "unfaithful_pairs",
               "model_majority_correct", "model_majority_incorrect"]

    # ──────────────── fig8: probe AUC bar chart ────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(subsets))
    width = 0.2
    colors = {"gt_at_qmark":   "#1f77b4", "gt_at_last":    "#aec7e8",
              "model_at_qmark":"#d62728", "model_at_last": "#ff9896"}
    labels = {"gt_at_qmark":   "GT probe   @ ? token",
              "gt_at_last":    "GT probe   @ last token",
              "model_at_qmark":"Model probe @ ? token",
              "model_at_last": "Model probe @ last token"}
    for i, k in enumerate(keys):
        aucs = [results[k][s]["auc"] for s in subsets]
        ns   = [results[k][s]["n"]   for s in subsets]
        # Replace NaN AUCs (single-class subset) with 0 so the bar shows as
        # empty, and annotate explicitly.
        plot_vals = [a if not np.isnan(a) else 0.0 for a in aucs]
        bars = ax.bar(x + (i - 1.5) * width, plot_vals, width,
                      label=labels[k], color=colors[k], edgecolor="black")
        for b, a, n in zip(bars, aucs, ns):
            if np.isnan(a):
                ax.annotate(f"n/a\n(1 class)\nn={n}",
                            (b.get_x() + b.get_width()/2, 0.02),
                            ha="center", va="bottom", fontsize=7, color="grey")
            else:
                ax.annotate(f"{a:.2f}\nn={n}",
                            (b.get_x() + b.get_width()/2, a),
                            ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (AUC = 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in subsets],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("OOF AUC (10-fold CV, pairs grouped)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Probe AUC: ground-truth vs model-majority targets, "
                 "at two pre-CoT positions\n(wm-person-death, 800 questions, "
                 "layer 20)")
    ax.legend(ncols=2, loc="upper right", fontsize=8)
    fig.tight_layout()
    out = outdir / "fig8_probe_auc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # ──────────────── fig9: per-pair pre-commitment ────────────────
    # For each (x, y) tuple, find the gt-variant and lt-variant question rows,
    # plot model-probe predicted-P(YES) at the qmark position for variant_a vs
    # variant_b.  Color by pair kind.  For unfaithful pairs the two variants
    # have OPPOSITE ground-truth answers, so a faithful model probe would
    # appear on the anti-diagonal.  Points ON the diagonal mean the probe
    # predicts the SAME answer regardless of which phrasing — pre-commitment.
    pos_key = "model_at_qmark"  # the headline probe
    p_oof = preds[pos_key]
    qids = preds["qid"]
    bkts = preds["bucket"]
    xs   = preds["x_name"]
    ys   = preds["y_name"]
    pkin = preds["pair_kind"]
    gt   = preds["gt_label"]

    # Index by (x, y) → {bucket: idx}
    by_xy: dict[tuple, dict[str, int]] = defaultdict(dict)
    for i, (x_, y_, b_) in enumerate(zip(xs, ys, bkts)):
        by_xy[(str(x_), str(y_))][str(b_)] = i

    pair_points: list[tuple[float, float, str, str]] = []
    for (x_, y_), bidx in by_xy.items():
        for a, b in [("gt_NO_1", "lt_YES_1"), ("gt_YES_1", "lt_NO_1")]:
            if a in bidx and b in bidx:
                ia, ib = bidx[a], bidx[b]
                if np.isnan(p_oof[ia]) or np.isnan(p_oof[ib]):
                    continue
                kind = str(pkin[ia])
                pair_points.append((float(p_oof[ia]), float(p_oof[ib]),
                                    kind, f"{x_}||{y_}||{a}"))

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    color_for = {"faithful": "#2ca02c", "mid": "#888888",
                 "unfaithful": "#d62728", "unknown": "#cccccc"}
    label_for = {"faithful": "faithful (Δ<0.2)", "mid": "mid (0.2≤Δ≤0.5)",
                 "unfaithful": "unfaithful (Δ>0.5)", "unknown": "(no pair)"}
    rng = np.random.default_rng(0)
    for kind in ["mid", "faithful", "unfaithful"]:
        pts = [(a, b) for a, b, k, _ in pair_points if k == kind]
        if not pts: continue
        xs_ = np.array([p[0] for p in pts]) + rng.uniform(-0.005, 0.005, len(pts))
        ys_ = np.array([p[1] for p in pts]) + rng.uniform(-0.005, 0.005, len(pts))
        ax.scatter(xs_, ys_, color=color_for[kind], alpha=0.65, s=35,
                   edgecolors="black", linewidths=0.3, label=label_for[kind])
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1,
            label="agrees on both phrasings (y = x)")
    ax.plot([0, 1], [1, 0], ":", color="navy", linewidth=1,
            label="opposite on opposite phrasings (faithful: y = 1 − x)")
    ax.set_xlabel("Model-probe P(YES) on gt-phrasing (correct = NO)")
    ax.set_ylabel("Model-probe P(YES) on lt-phrasing (correct = YES)")
    ax.set_title("Pre-commitment signature in the residual stream\n"
                 "(model-probe predictions at the '?' token; "
                 "diagonal = same answer regardless of phrasing)")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out = outdir / "fig9_pair_pre_commitment.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ──────────────── short writeup ────────────────
    md = []
    md.append("# Phase 2 — linear probes for pre-commitment at the prompt")
    md.append("")
    md.append("We trained logistic-regression probes on layer-20 residual-stream")
    md.append("activations of Qwen2.5-7B-Instruct at two pre-CoT positions of the")
    md.append("IPHR prompt: the '?' token that ends the question, and the last")
    md.append("token of the chat-templated prompt (the Qwen analogue of the IPHR")
    md.append("paper's 'colon' position).")
    md.append("")
    md.append("Two targets:")
    md.append("- **GT probe**: predicts the ground-truth YES/NO answer.")
    md.append("- **Model probe**: predicts the answer the model itself emits in")
    md.append("  the MAJORITY of its 20 rollouts (questions with tied votes or")
    md.append("  all-UNKNOWN dropped).")
    md.append("")
    md.append("10-fold cross-validation, folds grouped by (X, Y) tuple so that")
    md.append("both phrasings of a pair always land in the same fold. Training")
    md.append("set per fold: ~720 questions; held-out: ~80 questions.")
    md.append("")
    md.append("## Headline results (OOF accuracy)")
    md.append("")
    md.append("| Subset | GT @ ? | GT @ last | Model @ ? | Model @ last |")
    md.append("|---|---|---|---|---|")
    for s in subsets_all:
        row = [s.replace("_", " ")]
        for k in keys:
            r = results[k][s]
            if np.isnan(r["acc"]):
                row.append("—")
            else:
                row.append(f"{r['acc']:.3f} (n={r['n']})")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("## Headline plots")
    md.append("")
    md.append("- `fig8_probe_accuracy.png` — probe accuracy bars across subsets.")
    md.append("- `fig9_pair_pre_commitment.png` — per-pair scatter of the model")
    md.append("  probe's P(YES) prediction on the two phrasings of each pair.")
    md.append("  Pre-commitment manifests as DIAGONAL clustering on unfaithful")
    md.append("  pairs (whose two phrasings have OPPOSITE ground truth).")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("The interesting comparisons:")
    md.append("- **GT vs Model on overall**: How much of the answer is")
    md.append("  representable at the prompt at all?")
    md.append("- **GT on unfaithful pairs**: How well does layer 20 represent")
    md.append("  the CORRECT answer when the model gets it wrong?")
    md.append("- **Model on unfaithful pairs**: How well does layer 20 represent")
    md.append("  the BIASED answer the model is going to give?  If this is")
    md.append("  high, that's evidence of pre-commitment encoded in the")
    md.append("  residual stream before any CoT has been generated.")
    md.append("- **GT @ last vs Model @ last**: Position-specific comparison —")
    md.append("  does the bias signal get stronger as the model approaches the")
    md.append("  CoT boundary?")

    Path(args.out_md).write_text("\n".join(md))
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
