"""Phase 2: train linear probes at two pre-CoT token positions and analyse
performance on faithful vs unfaithful pairs.

Targets:
  gt_probe     — predicts the ground-truth answer (YES/NO)
  model_probe  — predicts what the model answers in the MAJORITY of rollouts
                 (UNKNOWN-only questions are dropped; tied questions are
                 broken toward the template's bias direction — see code).

For each (target × position) we run 10-fold cross-validation with folds
GROUPED BY (x_name, y_name) — both phrasings of a pair always go to the
same fold. This is the key falsification check: a probe must not learn to
recognise individual (X, Y) tuples and infer the answer from them.

Output:
  exp/iphr/probe_results.json     — per-fold + aggregate metrics
  exp/iphr/probe_predictions.npz  — out-of-fold predictions for each q
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

EXP_DIR = Path(os.environ.get("NLA_COT_EXP_DIR",
                              Path(__file__).resolve().parent.parent / "exp" / "iphr"))


def load_acts(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def build_labels(rollouts_path: Path, acts: dict, pairs_path: Path):
    """For each question (matched by (qid, bucket) order in acts):
        gt_label   — 1 if expected="YES", 0 if "NO"
        model_majority — 1 if YES > NO, 0 if NO > YES (resolved only).
                         Tied or all-UNKNOWN questions are flagged in
                         drop_for_model_probe.
        pair_kind  — 'unfaithful', 'faithful', 'mid' (based on Δacc thresholds)
        model_correct_majority — does majority answer match GT?
    """
    # Rollouts: gather per (qid, bucket) → Counter(YES/NO/UNKNOWN)
    rollouts = load_jsonl(rollouts_path)
    by_qbk: dict[tuple, Counter] = defaultdict(Counter)
    for r in rollouts:
        by_qbk[(r["qid"], r["bucket"])][r["parsed_answer"]] += 1

    # Pairs: per (template, x, y) → unfaithful flag (looking up in pairs file)
    pairs = load_jsonl(pairs_path)
    pair_kind_by_xy: dict[tuple, str] = {}
    for p in pairs:
        delta = p["delta"]
        if p["unfaithful"]:
            kind = "unfaithful"
        elif delta < 0.20:
            kind = "faithful"
        else:
            kind = "mid"
        pair_kind_by_xy[(p["template"], p["x_name"], p["y_name"])] = kind

    n = len(acts["qid"])
    gt_label = np.zeros(n, dtype=np.int64)
    model_majority = np.zeros(n, dtype=np.int64)
    drop = np.zeros(n, dtype=bool)
    pair_kind = np.empty(n, dtype=object)
    model_correct_majority = np.zeros(n, dtype=bool)
    n_yes = np.zeros(n, dtype=np.int32)
    n_no  = np.zeros(n, dtype=np.int32)
    n_unk = np.zeros(n, dtype=np.int32)

    for i in range(n):
        qid = str(acts["qid"][i])
        bkt = str(acts["bucket"][i])
        x   = str(acts["x_name"][i])
        y   = str(acts["y_name"][i])
        exp_ans = str(acts["expected_answer"][i])
        gt_label[i] = 1 if exp_ans == "YES" else 0

        c = by_qbk.get((qid, bkt), Counter())
        n_yes[i] = c.get("YES", 0)
        n_no[i]  = c.get("NO", 0)
        n_unk[i] = c.get("UNKNOWN", 0)

        if n_yes[i] > n_no[i]:
            model_majority[i] = 1
        elif n_no[i] > n_yes[i]:
            model_majority[i] = 0
        else:
            # tied or both-zero → drop from model_probe training/eval
            drop[i] = True
        if not drop[i]:
            model_correct_majority[i] = (model_majority[i] == gt_label[i])

        # Note: the pairs file uses the original "wm-person-death" template
        # name. acts has the same template.
        pair_kind[i] = pair_kind_by_xy.get(("wm-person-death", x, y), "unknown")

    return {
        "gt_label": gt_label,
        "model_majority": model_majority,
        "drop": drop,
        "pair_kind": pair_kind,
        "model_correct_majority": model_correct_majority,
        "n_yes": n_yes, "n_no": n_no, "n_unk": n_unk,
    }


def cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               n_splits: int = 10, C: float = 1.0,
               drop_mask: np.ndarray | None = None) -> np.ndarray:
    """Return out-of-fold predicted-probabilities for the positive class.

    drop_mask: bool array same length as y. Excluded examples are not
    trained on AND get NaN as their OOF prediction. All exclusions are
    test-time too — we don't evaluate on them.
    """
    if drop_mask is None:
        drop_mask = np.zeros_like(y, dtype=bool)
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        tr = tr[~drop_mask[tr]]
        # We still PREDICT on dropped test items (label=NaN) — keep them out:
        te = te[~drop_mask[te]]
        if len(tr) < 10 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def summarise(name: str, y: np.ndarray, oof: np.ndarray,
              mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = ~np.isnan(oof)
    else:
        mask = mask & ~np.isnan(oof)
    y_sub = y[mask]
    p_sub = oof[mask]
    if len(y_sub) == 0:
        return {"n": 0, "acc": float("nan"), "auc": float("nan"),
                "frac_pos": float("nan")}
    pred = (p_sub > 0.5).astype(int)
    acc = float((pred == y_sub).mean())
    auc = float("nan")
    if len(np.unique(y_sub)) >= 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_sub, p_sub))
    return {"n": int(mask.sum()), "acc": acc, "auc": auc,
            "frac_pos": float(y_sub.mean())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", default=str(EXP_DIR / "pd_full_acts.npz"))
    parser.add_argument("--rollouts",
                        default=str(EXP_DIR / "pd_full_combined.jsonl"))
    parser.add_argument("--pairs", default=str(EXP_DIR / "pd_full_pairs.jsonl"))
    parser.add_argument("--out", default=str(EXP_DIR / "probe_results.json"))
    parser.add_argument("--out-preds", default=str(EXP_DIR / "probe_predictions.npz"))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    acts = load_acts(Path(args.acts))
    labels = build_labels(Path(args.rollouts), acts, Path(args.pairs))

    # Grouping by (x_name, y_name) — both phrasings of a pair → same fold.
    group_keys = [f"{x}||{y}"
                  for x, y in zip(acts["x_name"], acts["y_name"])]
    groups = np.array(group_keys)
    print(f"loaded {len(groups)} questions, {len(set(group_keys))} unique (X,Y)")

    # Drop summary
    n_drop_model = int(labels["drop"].sum())
    print(f"dropping {n_drop_model} questions from model_probe "
          f"(tied or all-UNKNOWN)")

    results = {}
    preds: dict[str, np.ndarray] = {}

    for pos_name, X_key in [("qmark", "acts_qmark"), ("last", "acts_last")]:
        X = acts[X_key]
        for target_name, y, drop in [
            ("gt",    labels["gt_label"],       None),
            ("model", labels["model_majority"], labels["drop"]),
        ]:
            key = f"{target_name}_at_{pos_name}"
            print(f"\n=== {key} ===")
            oof = cv_predict(X, y, groups, n_splits=args.n_splits,
                             C=args.C, drop_mask=drop)
            preds[key] = oof

            # Subsets
            overall = summarise(key, y, oof)
            unf_mask  = labels["pair_kind"] == "unfaithful"
            mid_mask  = labels["pair_kind"] == "mid"
            fai_mask  = labels["pair_kind"] == "faithful"
            corr_mask = labels["model_correct_majority"]
            inc_mask  = (~corr_mask) & (~labels["drop"])

            res = {
                "overall":            summarise(key, y, oof),
                "faithful_pairs":     summarise(key, y, oof, fai_mask),
                "mid_pairs":          summarise(key, y, oof, mid_mask),
                "unfaithful_pairs":   summarise(key, y, oof, unf_mask),
                "model_majority_correct":   summarise(key, y, oof, corr_mask),
                "model_majority_incorrect": summarise(key, y, oof, inc_mask),
            }
            results[key] = res
            for sub, r in res.items():
                print(f"  {sub:30s}  n={r['n']:4d}  acc={r['acc']:.3f}  "
                      f"auc={r['auc']:.3f}  frac+={r['frac_pos']:.2f}")

    # Save
    Path(args.out).write_text(json.dumps(results, indent=2))
    np.savez(args.out_preds, **preds,
             gt_label=labels["gt_label"],
             model_majority=labels["model_majority"],
             drop=labels["drop"],
             pair_kind=np.array([str(k) for k in labels["pair_kind"]]),
             model_correct_majority=labels["model_correct_majority"],
             qid=acts["qid"], bucket=acts["bucket"],
             x_name=acts["x_name"], y_name=acts["y_name"],
             expected_answer=acts["expected_answer"])

    print(f"\nwrote {args.out}, {args.out_preds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
