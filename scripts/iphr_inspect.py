"""Phase 1.3: dump rollouts for unfaithful pairs side-by-side so we can
hand-classify the IPHR pattern (fact manipulation / argument switching /
answer flipping).

Usage:
    python iphr_inspect.py --top 6           # top-6 unfaithful by delta
    python iphr_inspect.py --template wm-us-natural-long --top 3
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default=str(EXP_DIR / "rollouts.jsonl"))
    parser.add_argument("--pairs", default=str(EXP_DIR / "unfaithful_pairs.jsonl"))
    parser.add_argument("--template", default=None)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--rollouts-per-variant", type=int, default=2)
    parser.add_argument("--max-completion-chars", type=int, default=1500)
    args = parser.parse_args()

    pairs = load_jsonl(Path(args.pairs))
    if args.template:
        pairs = [p for p in pairs if p["template"] == args.template]
    pairs.sort(key=lambda p: -p["delta"])
    pairs = pairs[:args.top]
    print(f"showing {len(pairs)} unfaithful pairs")

    # Index rollouts by (template, bucket, x, y) → list
    rollouts = load_jsonl(Path(args.rollouts))
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in rollouts:
        by_key[(r["template"], r["bucket"], r["x_name"], r["y_name"])].append(r)

    for i, p in enumerate(pairs):
        print("\n" + "█" * 90)
        print(f"PAIR {i+1}: template={p['template']} Δ={p['delta']:.2f} "
              f"bias={p['template_bias']}")
        print(f"  X = {p['x_name']!r}")
        print(f"  Y = {p['y_name']!r}")
        print(f"  variant_a = {p['variant_a']} (expected={p['expected_a']}, "
              f"acc={p['acc_a']:.2f})")
        print(f"  variant_b = {p['variant_b']} (expected={p['expected_b']}, "
              f"acc={p['acc_b']:.2f})")
        for v in (p["variant_a"], p["variant_b"]):
            rolls = by_key.get((p["template"], v, p["x_name"], p["y_name"]), [])
            # Sort by rollout_idx; show first N
            rolls.sort(key=lambda r: r["rollout_idx"])
            print("\n" + "─" * 90)
            print(f"VARIANT {v}: expected={rolls[0]['expected_answer'] if rolls else '?'}  "
                  f"({sum(1 for r in rolls if r['parsed_answer']==rolls[0]['expected_answer'])}/"
                  f"{len(rolls)} correct)")
            for r in rolls[:args.rollouts_per_variant]:
                print(f"\n  [rollout {r['rollout_idx']}] parsed={r['parsed_answer']}")
                comp = r["completion"][:args.max_completion_chars]
                # Indent the completion for readability
                for line in comp.split("\n"):
                    print(f"    {line}")
                if len(r["completion"]) > args.max_completion_chars:
                    print(f"    ...[truncated, {len(r['completion'])} total chars]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
