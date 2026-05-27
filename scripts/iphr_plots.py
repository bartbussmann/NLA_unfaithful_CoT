"""Phase 1 plots for the UK AISI work-test report.

Reads the Phase 1 rollouts JSONL and the pair-analysis output and produces:

  fig1_yes_rate.png         — per-template YES rate bar chart
  fig2_answer_distribution.png — stacked YES/NO/UNK per template
  fig3_pair_scatter.png     — acc_correct_variant vs acc_other_variant, colored
                              by template; off-diagonal = unfaithful
  fig4_delta_hist.png       — Δaccuracy histograms, one panel per template
  fig5_unfaithful_counts.png — bar chart of unfaithful-pair counts per template

Usage:
    python iphr_plots.py [--rollouts ... --pairs ... --outdir ...]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(os.environ.get("NLA_COT_EXP_DIR",
                              Path(__file__).resolve().parent.parent / "exp" / "iphr"))


TEMPLATE_LABELS = {
    "wm-nyt-pubdate": "NYT article\npublication dates",
    "wm-person-birth": "Historical figure\nbirth dates",
    "wm-person-death": "Historical figure\ndeath dates",
    "wm-us-natural-long": "US natural feature\nlongitude (east/west)",
}

# Stable color per template
TEMPLATE_COLORS = {
    "wm-nyt-pubdate":     "#1f77b4",
    "wm-person-birth":    "#2ca02c",
    "wm-person-death":    "#d62728",
    "wm-us-natural-long": "#9467bd",
}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


def fig1_yes_rate(rollouts, outdir):
    by_tmpl: dict[str, Counter] = defaultdict(Counter)
    for r in rollouts:
        by_tmpl[r["template"]][r["parsed_answer"]] += 1

    tmpls = list(TEMPLATE_LABELS.keys())
    yes_rates = []
    for t in tmpls:
        c = by_tmpl[t]
        yr = c.get("YES", 0) / max(c.get("YES", 0) + c.get("NO", 0), 1)
        yes_rates.append(yr * 100)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [TEMPLATE_COLORS[t] for t in tmpls]
    bars = ax.bar(range(len(tmpls)), yes_rates, color=colors, edgecolor="black")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="Unbiased (50%)")
    ax.set_xticks(range(len(tmpls)))
    ax.set_xticklabels([TEMPLATE_LABELS[t] for t in tmpls], fontsize=9)
    ax.set_ylabel("YES rate among resolved answers (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-template YES bias on IPHR questions\n"
                 "(Qwen2.5-7B-Instruct, temp 0.7, top-p 0.9)")
    # Annotate
    for b, yr in zip(bars, yes_rates):
        ax.annotate(f"{yr:.1f}%", (b.get_x() + b.get_width()/2, yr),
                    ha="center", va="bottom" if yr < 95 else "top", fontsize=10,
                    color="black")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = outdir / "fig1_yes_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig2_answer_distribution(rollouts, outdir):
    by_tmpl: dict[str, Counter] = defaultdict(Counter)
    for r in rollouts:
        by_tmpl[r["template"]][r["parsed_answer"]] += 1
    tmpls = list(TEMPLATE_LABELS.keys())
    yes = np.array([by_tmpl[t].get("YES", 0) for t in tmpls], dtype=float)
    no  = np.array([by_tmpl[t].get("NO", 0)  for t in tmpls], dtype=float)
    unk = np.array([by_tmpl[t].get("UNKNOWN", 0) for t in tmpls], dtype=float)
    tot = yes + no + unk
    yes_p, no_p, unk_p = yes/tot*100, no/tot*100, unk/tot*100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(tmpls))
    ax.bar(x, yes_p, color="#2ca02c", edgecolor="black", label="YES")
    ax.bar(x, no_p,  bottom=yes_p, color="#d62728", edgecolor="black", label="NO")
    ax.bar(x, unk_p, bottom=yes_p + no_p, color="#888888", edgecolor="black",
           label="UNKNOWN")
    ax.set_xticks(x)
    ax.set_xticklabels([TEMPLATE_LABELS[t] for t in tmpls], fontsize=9)
    ax.set_ylabel("% of rollouts")
    ax.set_title("Answer distribution per template (10 rollouts × question)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0))
    for i, t in enumerate(tmpls):
        ax.annotate(f"{int(yes[i])}", (i, yes_p[i]/2), ha="center", va="center",
                    color="white", fontsize=9)
        ax.annotate(f"{int(no[i])}", (i, yes_p[i] + no_p[i]/2), ha="center",
                    va="center", color="white", fontsize=9)
        if unk[i] > 5:
            ax.annotate(f"{int(unk[i])}", (i, yes_p[i] + no_p[i] + unk_p[i]/2),
                        ha="center", va="center", color="white", fontsize=9)
    fig.tight_layout()
    out = outdir / "fig2_answer_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig3_pair_scatter(pairs, outdir):
    """Per pair, plot (accuracy on the variant where correct=YES,
                       accuracy on the variant where correct=NO).
    Each pair has exactly one of each variant. Corners:
      (1,1) = perfect
      (0,0) = always wrong on both (rare; UNKNOWN noise)
      (1,0) = always says YES (correct when YES is right, wrong when NO is right)
      (0,1) = always says NO  (correct when NO is right, wrong when YES is right)
    Off-diagonal corners are the IPHR signature.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    by_tmpl: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in pairs:
        if p["expected_a"] == "YES":
            yes_acc, no_acc = p["acc_a"], p["acc_b"]
        else:
            yes_acc, no_acc = p["acc_b"], p["acc_a"]
        by_tmpl[p["template"]].append((yes_acc, no_acc))

    rng = np.random.default_rng(0)
    for t in TEMPLATE_LABELS:
        pts = by_tmpl.get(t, [])
        if not pts: continue
        xs = [a + rng.uniform(-0.02, 0.02) for a, _ in pts]
        ys = [b + rng.uniform(-0.02, 0.02) for _, b in pts]
        ax.scatter(xs, ys, color=TEMPLATE_COLORS[t], alpha=0.7, s=40,
                   edgecolors="black", linewidths=0.4,
                   label=TEMPLATE_LABELS[t].replace("\n", " "))

    # Diagonal = "consistent across phrasings" (model gives same accuracy)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
            label="faithful: y = x")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("accuracy when correct answer is YES")
    ax.set_ylabel("accuracy when correct answer is NO")
    ax.set_title("IPHR signature: same (X,Y), two phrasings\n"
                 "(off-diagonal corners = answer fixed regardless of question)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    # Corner annotations
    ax.annotate("perfect", (0.97, 0.97), fontsize=8, color="darkgreen",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e0ffe0",
                          ec="darkgreen", lw=0.5))
    ax.annotate("model always\nanswers YES", (0.97, 0.03), fontsize=8,
                color="darkred", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffe0e0",
                          ec="darkred", lw=0.5))
    ax.annotate("model always\nanswers NO", (0.03, 0.97), fontsize=8,
                color="darkred", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffe0e0",
                          ec="darkred", lw=0.5))
    fig.tight_layout()
    out = outdir / "fig3_pair_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig4_delta_hist(pairs, outdir):
    tmpls = list(TEMPLATE_LABELS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, tmpls):
        deltas = [p["delta"] for p in pairs if p["template"] == t]
        ax.hist(deltas, bins=np.linspace(0, 1, 11),
                color=TEMPLATE_COLORS[t], edgecolor="black", alpha=0.85)
        n_unf = sum(1 for d in deltas if d > 0.5)
        ax.axvline(0.5, color="darkred", linestyle="--", linewidth=1,
                   label=f"unfaithful threshold (Δ>0.5)\n{n_unf} pairs")
        ax.set_title(TEMPLATE_LABELS[t].replace("\n", " "), fontsize=10)
        ax.set_xlabel("Δaccuracy between paired phrasings")
        ax.set_ylabel("# pairs")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Accuracy gap distribution across pairs (same X,Y, two phrasings)",
                 fontsize=12)
    fig.tight_layout()
    out = outdir / "fig4_delta_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig5_unfaithful_counts(pairs, outdir):
    tmpls = list(TEMPLATE_LABELS.keys())
    n_pairs = []
    n_unf = []
    for t in tmpls:
        ps = [p for p in pairs if p["template"] == t]
        n_pairs.append(len(ps))
        n_unf.append(sum(1 for p in ps if p["unfaithful"]))
    rates = [u/p*100 if p else 0 for u, p in zip(n_unf, n_pairs)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(tmpls))
    colors = [TEMPLATE_COLORS[t] for t in tmpls]
    bars = ax.bar(x, n_unf, color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([TEMPLATE_LABELS[t] for t in tmpls], fontsize=9)
    ax.set_ylabel("Unfaithful pairs (Δacc > 50%, against bias)")
    ax.set_title("Unfaithful pair count per template "
                 f"(out of {n_pairs[0]} pairs per template)")
    for b, u, r in zip(bars, n_unf, rates):
        ax.annotate(f"{u}\n({r:.0f}%)",
                    (b.get_x() + b.get_width()/2, u),
                    ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    out = outdir / "fig5_unfaithful_counts.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default=str(EXP_DIR / "rollouts.jsonl"))
    parser.add_argument("--pairs", default=str(EXP_DIR / "pairs.jsonl"))
    parser.add_argument("--outdir", default=str(EXP_DIR / "plots"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rollouts = load_jsonl(Path(args.rollouts))
    pairs = load_jsonl(Path(args.pairs))
    print(f"loaded {len(rollouts)} rollouts, {len(pairs)} pairs")

    fig1_yes_rate(rollouts, outdir)
    fig2_answer_distribution(rollouts, outdir)
    fig3_pair_scatter(pairs, outdir)
    fig4_delta_hist(pairs, outdir)
    fig5_unfaithful_counts(pairs, outdir)


if __name__ == "__main__":
    main()
