"""Dump every CoT for wm-person-death pairs where the model is inconsistent
(accuracy delta > threshold). For each pair, show the X/Y/accuracy summary,
then print every rollout where the model gave the WRONG answer — these are
the CoTs that justify the biased same-answer-both-ways behavior.

Usage:
    python iphr_wrong_cots.py --template wm-person-death --gap 0.20 \
        --out /workspace-vast/$USER/exp/iphr/wrong_cots_person-death.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path(os.environ.get("NLA_COT_EXP_DIR",
                              Path(__file__).resolve().parent.parent / "exp" / "iphr"))


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


PAIR_PARTNERS = {
    "gt_NO_1": "lt_YES_1",
    "lt_YES_1": "gt_NO_1",
    "gt_YES_1": "lt_NO_1",
    "lt_NO_1": "gt_YES_1",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default=str(EXP_DIR / "rollouts.jsonl"))
    parser.add_argument("--pairs", default=str(EXP_DIR / "pairs.jsonl"))
    parser.add_argument("--template", default="wm-person-death")
    parser.add_argument("--gap", type=float, default=0.20,
                        help="Minimum accuracy delta to include the pair.")
    parser.add_argument("--include-correct-context", action="store_true",
                        default=True,
                        help="Also include 1 correct CoT from the high-acc variant.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pairs = [p for p in load_jsonl(Path(args.pairs))
             if p["template"] == args.template and p["delta"] >= args.gap]
    pairs.sort(key=lambda p: -p["delta"])
    print(f"pairs matching template={args.template} delta>={args.gap}: {len(pairs)}")

    rollouts = load_jsonl(Path(args.rollouts))
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in rollouts:
        by_key[(r["template"], r["bucket"], r["x_name"], r["y_name"])].append(r)
    for k in by_key:
        by_key[k].sort(key=lambda r: r["rollout_idx"])

    lines: list[str] = []
    lines.append(f"# wm-person-death — wrong-answer CoTs (Δacc ≥ {args.gap:.2f})")
    lines.append("")
    lines.append(f"Source: rollouts at temperature 0.7, top-p 0.9, "
                 f"Qwen2.5-7B-Instruct.")
    lines.append(f"Pairs shown: {len(pairs)} (sorted by Δaccuracy, descending).")
    lines.append("")
    lines.append("Each pair shows two phrasings of the same (X, Y, death-date) "
                 "comparison. For the *wrong* variant, every rollout where the "
                 "model answered incorrectly is included verbatim — these are "
                 "the CoTs that fabricate facts, switch frames, or flip "
                 "answers. For context the first correct rollout from the *right* "
                 "variant is also shown.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, p in enumerate(pairs, 1):
        # Determine the "wrong" variant (lower acc) and the "right" variant
        if p["acc_a"] <= p["acc_b"]:
            wrong_v, wrong_acc, wrong_exp = p["variant_a"], p["acc_a"], p["expected_a"]
            right_v, right_acc, right_exp = p["variant_b"], p["acc_b"], p["expected_b"]
        else:
            wrong_v, wrong_acc, wrong_exp = p["variant_b"], p["acc_b"], p["expected_b"]
            right_v, right_acc, right_exp = p["variant_a"], p["acc_a"], p["expected_a"]

        lines.append(f"## Pair {i}: X = {p['x_name']!r} vs Y = {p['y_name']!r}")
        lines.append("")
        lines.append(f"- Δaccuracy = **{p['delta']:.2f}**")
        lines.append(f"- Template bias direction: **{p['template_bias']}**")
        lines.append(f"- WRONG variant: **{wrong_v}** (expected={wrong_exp}, "
                     f"acc={wrong_acc:.2f}) — model contradicts ground truth here")
        lines.append(f"- RIGHT variant: {right_v} (expected={right_exp}, "
                     f"acc={right_acc:.2f})")
        lines.append("")

        # Show ALL wrong CoTs from the WRONG variant
        wrong_rolls = by_key.get((args.template, wrong_v, p["x_name"], p["y_name"]), [])
        wrong_ans_rolls = [r for r in wrong_rolls
                           if r["parsed_answer"] != wrong_exp
                           and r["parsed_answer"] in ("YES", "NO")]
        right_ans_in_wrong_variant = [r for r in wrong_rolls
                                       if r["parsed_answer"] == wrong_exp]

        # The question text for the wrong variant
        if wrong_rolls:
            q = wrong_rolls[0]["q_str"]
            # Strip "about historical figures:\n\n" if present
            q_clean = q.replace("about historical figures:\n\n", "").strip()
            lines.append(f"### Wrong variant question")
            lines.append("")
            lines.append(f"> {q_clean}")
            lines.append("")
            lines.append(f"### Wrong rollouts on this variant "
                         f"({len(wrong_ans_rolls)} of {len(wrong_rolls)} — "
                         f"{len(right_ans_in_wrong_variant)} correct)")
            lines.append("")

            for r in wrong_ans_rolls:
                lines.append(f"#### Rollout {r['rollout_idx']} (parsed = "
                             f"`{r['parsed_answer']}`, expected `{wrong_exp}`)")
                lines.append("")
                lines.append("```")
                lines.append(r["completion"].rstrip())
                lines.append("```")
                lines.append("")

        # Context: 1 right rollout from the RIGHT variant
        if args.include_correct_context:
            right_rolls = by_key.get((args.template, right_v, p["x_name"], p["y_name"]), [])
            good = [r for r in right_rolls if r["parsed_answer"] == right_exp]
            if good:
                r = good[0]
                q_clean = r["q_str"].replace("about historical figures:\n\n", "").strip()
                lines.append(f"### For comparison: correct CoT on the OTHER "
                             f"phrasing ({right_v})")
                lines.append("")
                lines.append(f"> {q_clean}")
                lines.append("")
                lines.append(f"#### Rollout {r['rollout_idx']} (parsed = "
                             f"`{r['parsed_answer']}`, expected `{right_exp}`)")
                lines.append("")
                lines.append("```")
                lines.append(r["completion"].rstrip())
                lines.append("```")
                lines.append("")

        lines.append("---")
        lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
