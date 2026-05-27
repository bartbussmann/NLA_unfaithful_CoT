"""Simple word-frequency-ratio analysis of the NLA explanations.

For each category split (gt YES vs gt NO; model YES vs model NO) and each
position (`?` token, last token), we tokenise the explanations, count
unigrams, and report the words whose frequency is most skewed toward one
class. To avoid noise, words must appear in at least N distinct
explanations across the split.

Output: exp/iphr/word_ratios.md
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

from iphr_probes import build_labels, load_acts, load_jsonl

EXP = Path(os.environ.get("NLA_COT_EXP_DIR",
                          Path(__file__).resolve().parent.parent / "exp" / "iphr"))

TOKEN_RE = re.compile(r"[a-z][a-z]+")

STOPWORDS = set("""
the a an and or but of to in on at by for from with as that this these those
is are was were be been being am do does did doing have has had having
it its their there here what which who whom where when why how
not no nor so if then than because while until about over under
i you he she we they me him her us them my your his their our
can could should would may might must will shall just only also even
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS
            and len(t) >= 3]


def explanation_counts(rows: list[dict]) -> Counter:
    """Document-frequency: how many explanations contain each token."""
    c = Counter()
    for r in rows:
        text = r.get("explanation") or r.get("raw_text") or ""
        tokens = set(tokenize(text))
        for t in tokens:
            c[t] += 1
    return c


def ratio_table(pos_rows: list[dict], neg_rows: list[dict],
                min_docs: int = 20, k: int = 20) -> tuple[list, list]:
    """Return (positive-class tokens, negative-class tokens) ranked by ratio.

    ratio = (df_pos / N_pos) / (df_neg / N_neg)   (log2, additive smoothing)
    A token must appear in at least min_docs total explanations.
    """
    N_pos = len(pos_rows); N_neg = len(neg_rows)
    df_pos = explanation_counts(pos_rows)
    df_neg = explanation_counts(neg_rows)
    keys = set(df_pos) | set(df_neg)
    rows = []
    for w in keys:
        total = df_pos.get(w, 0) + df_neg.get(w, 0)
        if total < min_docs:
            continue
        p_pos = (df_pos.get(w, 0) + 1) / (N_pos + 2)
        p_neg = (df_neg.get(w, 0) + 1) / (N_neg + 2)
        log_ratio = float(np.log2(p_pos / p_neg))
        rows.append({
            "word": w, "df_pos": df_pos.get(w, 0), "df_neg": df_neg.get(w, 0),
            "log_ratio": log_ratio,
        })
    rows.sort(key=lambda r: -r["log_ratio"])
    top_pos = rows[:k]
    top_neg = rows[-k:][::-1]
    return top_pos, top_neg


def fmt(rows, total_pos, total_neg, header_pos, header_neg) -> list[str]:
    out = []
    out.append(f"| Word | log₂ ratio | docs in {header_pos} (N={total_pos}) "
               f"| docs in {header_neg} (N={total_neg}) |")
    out.append("|---|---|---|---|")
    for r in rows:
        out.append(f"| `{r['word']}` | {r['log_ratio']:+.2f} | "
                   f"{r['df_pos']} | {r['df_neg']} |")
    return out


def main() -> int:
    acts = load_acts(EXP / "pd_full_acts.npz")
    labels = build_labels(EXP / "pd_full_combined.jsonl", acts,
                          EXP / "pd_full_pairs.jsonl")
    nla = load_jsonl(EXP / "pd_full_nla.jsonl")
    by_qbk_pos = {(r["qid"], r["bucket"], r["position"]): r for r in nla}

    qids = acts["qid"]; bkts = acts["bucket"]
    gt = labels["gt_label"]; mm = labels["model_majority"]; drop = labels["drop"]

    md = []
    md.append("# Word-frequency-ratio analysis of NLA explanations")
    md.append("")
    md.append("For each category split and each position we tokenise the NLA "
              "explanation strings (lowercase, drop stopwords and tokens of "
              "length < 3), compute document-frequency for each word in each "
              "class, and rank by additive-smoothed log₂ ratio.  Words must "
              "appear in ≥ 20 explanations across both classes.")
    md.append("")
    md.append("Reading: a positive `log₂ ratio` means the word is more common "
              "in the YES-class explanations than in the NO-class; +1.0 means "
              "twice as common.")
    md.append("")

    for pos_name in ["qmark", "last"]:
        for target_name, y, drop_mask, pos_label, neg_label in [
            ("gt",    gt, np.zeros(len(qids), bool), "GT = YES",    "GT = NO"),
            ("model", mm, drop,                      "Model = YES", "Model = NO"),
        ]:
            # Gather rows
            pos_rows = []
            neg_rows = []
            for i, (q, b) in enumerate(zip(qids, bkts)):
                if drop_mask[i]:
                    continue
                r = by_qbk_pos.get((str(q), str(b), pos_name))
                if r is None:
                    continue
                if y[i] == 1:
                    pos_rows.append(r)
                else:
                    neg_rows.append(r)
            top_pos, top_neg = ratio_table(pos_rows, neg_rows,
                                            min_docs=20, k=20)
            md.append(f"## {target_name} @ {pos_name} — "
                      f"{pos_label} vs {neg_label}")
            md.append("")
            md.append(f"### Words more frequent in **{pos_label}** explanations")
            md.append("")
            md.extend(fmt(top_pos, len(pos_rows), len(neg_rows),
                          pos_label, neg_label))
            md.append("")
            md.append(f"### Words more frequent in **{neg_label}** explanations")
            md.append("")
            md.extend(fmt(top_neg, len(pos_rows), len(neg_rows),
                          pos_label, neg_label))
            md.append("")

    out = EXP / "word_ratios.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
