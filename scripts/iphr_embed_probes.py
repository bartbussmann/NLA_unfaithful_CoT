"""Phase 3 follow-up B: embed each NLA explanation with a sentence-transformer
and train the same 4 probes as Phase 2 / Phase 3-text on top of the dense
embeddings. The richer features should give the text probes a fairer chance
than TF-IDF.

Embedder: sentence-transformers/all-MiniLM-L6-v2 (22M params, 384-dim, MIT).
Loaded via plain transformers.AutoModel with mean-pooling over the
attention-mask — no sentence-transformers package required.

Output:
  exp/iphr/pd_full_nla_embeddings.npz
  exp/iphr/embed_probe_results.json
  exp/iphr/embed_probe_predictions.npz
  exp/iphr/plots/fig13_act_vs_text_vs_embed.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer

from iphr_probes import build_labels, load_acts, load_jsonl

EXP = Path(os.environ.get("NLA_COT_EXP_DIR",
                          Path(__file__).resolve().parent.parent / "exp" / "iphr"))

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts: list[str], device: str = "cuda:0",
                batch_size: int = 64) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()
    out = []
    for s in range(0, len(texts), batch_size):
        chunk = texts[s:s + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=256,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        emb = mean_pool(h, enc["attention_mask"])
        # L2 normalise (standard for these sentence-transformer models)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def summarise(y: np.ndarray, oof: np.ndarray, mask: np.ndarray) -> dict:
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


def cv_predict(X, y, groups, drop_mask, n_splits=10, C=1.0):
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        tr = tr[~drop_mask[tr]]
        te = te[~drop_mask[te]]
        if len(tr) < 10 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def main() -> int:
    acts = load_acts(EXP / "pd_full_acts.npz")
    labels = build_labels(EXP / "pd_full_combined.jsonl", acts,
                          EXP / "pd_full_pairs.jsonl")
    nla = load_jsonl(EXP / "pd_full_nla.jsonl")
    by_qbk_pos = {(r["qid"], r["bucket"], r["position"]): r for r in nla}

    qids = acts["qid"]; bkts = acts["bucket"]; xs = acts["x_name"]; ys = acts["y_name"]
    groups = np.array([f"{x}||{y}" for x, y in zip(xs, ys)])
    print(f"loaded {len(qids)} questions, {len(set(groups))} unique (X,Y)")

    # Gather explanation texts aligned with `acts` ordering, for both positions
    emb_cache = EXP / "pd_full_nla_embeddings.npz"
    if emb_cache.exists():
        z = np.load(emb_cache, allow_pickle=False)
        emb_qmark, emb_last = z["emb_qmark"], z["emb_last"]
        print(f"[cache] loaded embeddings from {emb_cache}")
    else:
        def texts_for(pos):
            out = []
            for q, b in zip(qids, bkts):
                r = by_qbk_pos[(str(q), str(b), pos)]
                out.append(r.get("explanation") or r.get("raw_text") or "")
            return out
        print(f"[embed] computing embeddings with {EMBED_MODEL}")
        emb_qmark = embed_texts(texts_for("qmark"))
        emb_last  = embed_texts(texts_for("last"))
        np.savez(emb_cache, emb_qmark=emb_qmark, emb_last=emb_last)
        print(f"[embed] wrote {emb_cache} (shapes {emb_qmark.shape}, {emb_last.shape})")

    # Probes
    results = {}
    preds = {}
    for pos_name, X in [("qmark", emb_qmark), ("last", emb_last)]:
        for target_name, y, drop in [
            ("gt",    labels["gt_label"],       np.zeros(len(qids), dtype=bool)),
            ("model", labels["model_majority"], labels["drop"]),
        ]:
            key = f"{target_name}_at_{pos_name}_embed"
            print(f"\n=== {key} ===")
            oof = cv_predict(X, y, groups, drop)
            preds[key] = oof
            unf = labels["pair_kind"] == "unfaithful"
            fai = labels["pair_kind"] == "faithful"
            mid = labels["pair_kind"] == "mid"
            corr = labels["model_correct_majority"]
            inc = (~corr) & (~labels["drop"])
            ones = np.ones(len(y), dtype=bool)
            res = {
                "overall":                  summarise(y, oof, ones),
                "faithful_pairs":           summarise(y, oof, fai),
                "mid_pairs":                summarise(y, oof, mid),
                "unfaithful_pairs":         summarise(y, oof, unf),
                "model_majority_correct":   summarise(y, oof, corr),
                "model_majority_incorrect": summarise(y, oof, inc),
            }
            results[key] = res
            for sub, r in res.items():
                print(f"  {sub:30s}  n={r['n']:4d}  acc={r['acc']:.3f}  "
                      f"auc={r['auc']:.3f}  frac+={r['frac_pos']:.2f}")

    # Save
    (EXP / "embed_probe_results.json").write_text(json.dumps(results, indent=2))
    np.savez(EXP / "embed_probe_predictions.npz", **preds,
             gt_label=labels["gt_label"], model_majority=labels["model_majority"],
             drop=labels["drop"],
             pair_kind=np.array([str(k) for k in labels["pair_kind"]]),
             model_correct_majority=labels["model_correct_majority"],
             qid=qids, bucket=bkts, x_name=xs, y_name=ys,
             expected_answer=acts["expected_answer"])
    print(f"\nwrote {EXP / 'embed_probe_results.json'}")

    # ───── Comparison fig: activation vs TF-IDF text vs embedding ─────
    act = json.loads((EXP / "probe_results.json").read_text())
    txt = json.loads((EXP / "text_probe_results.json").read_text())
    emb = results
    subsets = ["overall", "faithful_pairs", "unfaithful_pairs",
               "model_majority_correct", "model_majority_incorrect"]
    keys = [("gt_at_qmark",   "gt_at_qmark_text",   "gt_at_qmark_embed",   "GT @ ?",      "#1f77b4"),
            ("gt_at_last",    "gt_at_last_text",    "gt_at_last_embed",    "GT @ last",   "#aec7e8"),
            ("model_at_qmark","model_at_qmark_text","model_at_qmark_embed","Model @ ?",   "#d62728"),
            ("model_at_last", "model_at_last_text", "model_at_last_embed", "Model @ last","#ff9896")]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(subsets))
    width = 0.07
    offset = -6 * width
    for k_a, k_t, k_e, label, color in keys:
        for vals_dict, suffix, hatch in [(act, "act", ""),
                                          (txt, "txt", "///"),
                                          (emb, "emb", "...")]:
            v = [vals_dict[k_a if suffix == "act" else (k_t if suffix == "txt" else k_e)][s]["auc"]
                 for s in subsets]
            v_plot = [vv if not np.isnan(vv) else 0 for vv in v]
            bars = ax.bar(x + offset, v_plot, width,
                          label=f"{label} ({suffix})",
                          color=color, edgecolor="black", hatch=hatch)
            for bar, vv in zip(bars, v):
                if np.isnan(vv):
                    ax.annotate("n/a", (bar.get_x() + bar.get_width()/2, 0.02),
                                ha="center", va="bottom", fontsize=6, color="grey")
            offset += width
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in subsets], rotation=20,
                       ha="right", fontsize=9)
    ax.set_ylabel("OOF AUC"); ax.set_ylim(0, 1.05)
    ax.set_title("Probe AUC: activation (solid) vs TF-IDF text (///) "
                 "vs MiniLM embedding (...) of NLA explanation",
                 fontsize=11)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5),
              ncols=1)
    fig.tight_layout()
    out = EXP / "plots" / "fig13_act_vs_text_vs_embed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
