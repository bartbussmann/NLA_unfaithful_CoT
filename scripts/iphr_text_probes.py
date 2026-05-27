"""Phase 3 follow-up: train TF-IDF + logistic-regression probes on the NLA
explanation text itself, mirroring the activation-based probes from Phase 2.

For each (target ∈ {gt, model}) × (position ∈ {qmark, last}) we:
  1. Build a TF-IDF vectoriser on the explanation strings (uni+bigram).
  2. Run 10-fold CV with folds grouped by (X, Y) tuple (same split convention
     as the activation probes, so the comparison is apples-to-apples).
  3. Report accuracy and AUC overall and on the same subsets as Phase 2.

Then for each axis we fit a SINGLE full-data classifier and pull the words /
bigrams with the largest positive and negative coefficients.

Outputs:
  exp/iphr/text_probe_results.json
  exp/iphr/text_probe_predictions.npz
  exp/iphr/text_probe_words.md     — top discriminative tokens per axis
  exp/iphr/plots/fig11_text_vs_activation.png  — bar chart vs Phase 2
  exp/iphr/plots/fig12_discriminative_words.png — top-tokens visual
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from iphr_probes import build_labels, load_acts, load_jsonl

EXP = Path(os.environ.get("NLA_COT_EXP_DIR",
                          Path(__file__).resolve().parent.parent / "exp" / "iphr"))


def build_text_matrix(rows_by_qbk_pos: dict, qids, buckets, position: str,
                      vectoriser: TfidfVectorizer | None = None,
                      fit: bool = True):
    """Return (X_sparse, vectoriser, texts) aligned with acts ordering."""
    texts = []
    for q, b in zip(qids, buckets):
        r = rows_by_qbk_pos[(str(q), str(b), position)]
        # Use the parsed explanation if present, otherwise the raw text.
        t = r.get("explanation") or r.get("raw_text") or ""
        texts.append(t)
    if fit:
        if vectoriser is None:
            vectoriser = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2),
                min_df=5, max_df=0.95,
                stop_words="english",
                token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b",
            )
        X = vectoriser.fit_transform(texts)
    else:
        assert vectoriser is not None
        X = vectoriser.transform(texts)
    return X, vectoriser, texts


def cv_predict_text(rows_by_qbk_pos, qids, buckets, position, y, groups,
                    drop_mask, n_splits=10, C=1.0):
    """Run grouped 10-fold CV refitting the vectoriser on each fold's train set.

    Fitting the vectoriser only on train is important: vocabulary leakage from
    held-out (X,Y) tuples could trivially help, especially with name-like
    tokens.
    """
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(np.zeros((n, 1)), y, groups)):
        tr = tr[~drop_mask[tr]]
        te = te[~drop_mask[te]]
        if len(tr) < 10 or len(np.unique(y[tr])) < 2:
            continue
        # Fit vectoriser on train texts
        tr_qids   = qids[tr];   tr_bkts = buckets[tr]
        te_qids   = qids[te];   te_bkts = buckets[te]
        vec = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2),
            min_df=3, max_df=0.95, stop_words="english",
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b",
        )
        Xtr, _, _ = build_text_matrix(rows_by_qbk_pos, tr_qids, tr_bkts, position,
                                       vectoriser=vec, fit=True)
        Xte, _, _ = build_text_matrix(rows_by_qbk_pos, te_qids, te_bkts, position,
                                       vectoriser=vec, fit=False)
        clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return oof


def summarise(y, oof, mask):
    mask = mask & ~np.isnan(oof)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "acc": float("nan"), "auc": float("nan"),
                "frac_pos": float("nan")}
    y_sub = y[mask]; p_sub = oof[mask]
    acc = float(((p_sub > 0.5).astype(int) == y_sub).mean())
    auc = float("nan")
    if len(np.unique(y_sub)) >= 2:
        auc = float(roc_auc_score(y_sub, p_sub))
    return {"n": n, "acc": acc, "auc": auc, "frac_pos": float(y_sub.mean())}


def top_words(vectoriser, coef, k=20):
    feat = np.array(vectoriser.get_feature_names_out())
    order = np.argsort(coef)
    pos = feat[order[-k:][::-1]];  pos_w = coef[order[-k:][::-1]]
    neg = feat[order[:k]];          neg_w = coef[order[:k]]
    return list(zip(pos, pos_w.tolist())), list(zip(neg, neg_w.tolist()))


def main():
    acts = load_acts(EXP / "pd_full_acts.npz")
    labels = build_labels(EXP / "pd_full_combined.jsonl", acts,
                          EXP / "pd_full_pairs.jsonl")
    qids = acts["qid"]; buckets = acts["bucket"]; xs = acts["x_name"]; ys = acts["y_name"]
    groups = np.array([f"{x}||{y}" for x, y in zip(xs, ys)])
    print(f"loaded {len(qids)} questions, {len(set(groups))} unique (X,Y)")

    # Index NLA rows by (qid, bucket, position)
    nla = load_jsonl(EXP / "pd_full_nla.jsonl")
    by_qbk_pos = {(r["qid"], r["bucket"], r["position"]): r for r in nla}
    print(f"loaded {len(nla)} NLA explanations")

    results = {}
    preds = {}
    full_data_fits = {}  # for word analysis

    for pos_name in ["qmark", "last"]:
        for target_name, y_arr, drop in [
            ("gt",    labels["gt_label"],       np.zeros(len(qids), dtype=bool)),
            ("model", labels["model_majority"], labels["drop"]),
        ]:
            key = f"{target_name}_at_{pos_name}_text"
            print(f"\n=== {key} ===")
            oof = cv_predict_text(by_qbk_pos, qids, buckets, pos_name,
                                  y_arr, groups, drop)
            preds[key] = oof

            # subsets
            ones = np.ones(len(y_arr), dtype=bool)
            unf  = labels["pair_kind"] == "unfaithful"
            mid  = labels["pair_kind"] == "mid"
            fai  = labels["pair_kind"] == "faithful"
            corr = labels["model_correct_majority"]
            inc  = (~corr) & (~labels["drop"])
            res = {
                "overall":            summarise(y_arr, oof, ones),
                "faithful_pairs":     summarise(y_arr, oof, fai),
                "mid_pairs":          summarise(y_arr, oof, mid),
                "unfaithful_pairs":   summarise(y_arr, oof, unf),
                "model_majority_correct":   summarise(y_arr, oof, corr),
                "model_majority_incorrect": summarise(y_arr, oof, inc),
            }
            results[key] = res
            for sub, r in res.items():
                print(f"  {sub:30s}  n={r['n']:4d}  acc={r['acc']:.3f}  "
                      f"auc={r['auc']:.3f}  frac+={r['frac_pos']:.2f}")

            # full-data fit for word analysis
            mask = ~drop
            vec = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2),
                min_df=5, max_df=0.95, stop_words="english",
                token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b",
            )
            Xfull, _, _ = build_text_matrix(by_qbk_pos, qids[mask], buckets[mask],
                                            pos_name, vectoriser=vec, fit=True)
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
            clf.fit(Xfull, y_arr[mask])
            full_data_fits[key] = (vec, clf.coef_[0])

    # ─────── compare to Phase 2 activation probes ───────
    act_results = json.loads((EXP / "probe_results.json").read_text())
    subsets = ["overall", "faithful_pairs", "unfaithful_pairs",
               "model_majority_correct", "model_majority_incorrect"]
    keys_act = ["gt_at_qmark", "gt_at_last", "model_at_qmark", "model_at_last"]
    keys_txt = [k + "_text" for k in keys_act]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(subsets))
    width = 0.1
    offset = -3.5 * width
    metric = "auc"
    for k_act, k_txt, color_act, color_txt, label in [
        ("gt_at_qmark",   "gt_at_qmark_text",   "#1f77b4", "#1f77b4", "GT @ ?"),
        ("gt_at_last",    "gt_at_last_text",    "#aec7e8", "#aec7e8", "GT @ last"),
        ("model_at_qmark","model_at_qmark_text","#d62728", "#d62728", "Model @ ?"),
        ("model_at_last", "model_at_last_text", "#ff9896", "#ff9896", "Model @ last"),
    ]:
        vals_act = [act_results[k_act][s][metric] for s in subsets]
        vals_txt = [results[k_txt][s][metric] for s in subsets]
        # Replace NaN AUCs (single-class subsets) with 0 — handled by the
        # auto-annotation below.
        v_act_plot = [v if not np.isnan(v) else 0 for v in vals_act]
        v_txt_plot = [v if not np.isnan(v) else 0 for v in vals_txt]
        b1 = ax.bar(x + offset, v_act_plot, width, label=f"{label} (act)",
                    color=color_act, edgecolor="black")
        offset += width
        b2 = ax.bar(x + offset, v_txt_plot, width, label=f"{label} (text)",
                    color=color_txt, edgecolor="black", hatch="///")
        offset += width
        for bar, v in list(zip(b1, vals_act)) + list(zip(b2, vals_txt)):
            if np.isnan(v):
                ax.annotate("n/a", (bar.get_x() + bar.get_width()/2, 0.02),
                            ha="center", va="bottom", fontsize=6, color="grey")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1,
               label="chance (AUC = 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in subsets],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("OOF AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Probe targets recoverable from layer-20 residual\n"
                 "vs from the NLA explanation text alone (TF-IDF)",
                 fontsize=12)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5),
              ncols=1)
    fig.tight_layout()
    out = EXP / "plots" / "fig11_text_vs_activation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # ─────── discriminative words ───────
    md_lines = []
    md_lines.append("# Discriminative tokens per probe axis")
    md_lines.append("")
    md_lines.append("Top-K coefficients of a logistic regression trained on TF-IDF "
                    "uni+bigrams of the NLA explanation text. Positive direction is "
                    "predicted-YES; negative direction is predicted-NO. For the GT "
                    "probe the target is the ground-truth answer; for the Model "
                    "probe it is the model's majority answer across 20 rollouts.")
    md_lines.append("")
    for key in ["gt_at_qmark_text", "gt_at_last_text",
                "model_at_qmark_text", "model_at_last_text"]:
        vec, coef = full_data_fits[key]
        pos, neg = top_words(vec, coef, k=15)
        md_lines.append(f"## {key}")
        md_lines.append("")
        md_lines.append("| Predicts YES (top) | weight | Predicts NO (top) | weight |")
        md_lines.append("|---|---|---|---|")
        for (pw, pc), (nw, nc) in zip(pos, neg):
            md_lines.append(f"| `{pw}` | {pc:+.2f} | `{nw}` | {nc:+.2f} |")
        md_lines.append("")
    (EXP / "text_probe_words.md").write_text("\n".join(md_lines))
    print(f"wrote {EXP / 'text_probe_words.md'}")

    # Bar plot of top-words (model_at_qmark_text — the headline)
    vec, coef = full_data_fits["model_at_qmark_text"]
    pos, neg = top_words(vec, coef, k=12)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    pos_words = [w for w, _ in pos][::-1]; pos_w = [c for _, c in pos][::-1]
    neg_words = [w for w, _ in neg];        neg_w = [c for _, c in neg]
    axes[0].barh(range(len(pos_words)), pos_w, color="#2ca02c", edgecolor="black")
    axes[0].set_yticks(range(len(pos_words))); axes[0].set_yticklabels(pos_words, fontsize=9)
    axes[0].set_title("Predict YES (model probe @ ? token)")
    axes[0].set_xlabel("LR coefficient")
    axes[1].barh(range(len(neg_words)), [-c for c in neg_w], color="#d62728",
                  edgecolor="black")
    axes[1].set_yticks(range(len(neg_words))); axes[1].set_yticklabels(neg_words, fontsize=9)
    axes[1].set_title("Predict NO (model probe @ ? token)")
    axes[1].set_xlabel("|LR coefficient|")
    fig.suptitle("Top discriminative tokens — NLA explanation → model's answer",
                 fontsize=11)
    fig.tight_layout()
    out = EXP / "plots" / "fig12_discriminative_words.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Save
    (EXP / "text_probe_results.json").write_text(json.dumps(results, indent=2))
    np.savez(EXP / "text_probe_predictions.npz", **preds,
             gt_label=labels["gt_label"],
             model_majority=labels["model_majority"],
             drop=labels["drop"],
             pair_kind=np.array([str(k) for k in labels["pair_kind"]]),
             model_correct_majority=labels["model_correct_majority"],
             qid=qids, bucket=buckets, x_name=xs, y_name=ys,
             expected_answer=acts["expected_answer"])
    print(f"wrote {EXP / 'text_probe_results.json'}, "
          f"{EXP / 'text_probe_predictions.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
