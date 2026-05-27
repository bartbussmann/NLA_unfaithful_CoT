"""Phase 3: NLA-decode every layer-20 activation at the ? and last positions
for all 800 wm-person-death questions.

Loads pd_full_acts.npz (produced by iphr_extract_acts.py) and runs the
released kitft/nla-qwen2.5-7b-L20-av in batched mode. The AV's canonical
prompt is the same for every input — only the injected activation differs
— so we build the prompt embeddings once and inject in-batch.

Output: pd_full_nla.jsonl, one record per (question, position):

    {qid, bucket, x_name, y_name, expected_answer,
     position: "qmark" | "last",
     explanation: str,         # extracted from <explanation>…</explanation>
     raw_text: str}            # full AV output for debugging
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP_DIR = Path(os.environ.get("NLA_COT_EXP_DIR",
                              Path(__file__).resolve().parent.parent / "exp" / "iphr"))

DEFAULT_AV = "kitft/nla-qwen2.5-7b-L20-av"

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def resolve_local_snapshot(repo_id: str) -> str:
    """Return a local HF snapshot path if present, else repo_id."""
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
    cache = Path(hf_home) / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if cache.exists():
        snaps = sorted(cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snaps:
            return str(snaps[0])
    return repo_id


def load_sidecar(av_dir: str):
    meta = yaml.safe_load((Path(av_dir) / "nla_meta.yaml").read_text())
    t = meta["tokens"]
    return {
        "d_model": meta["d_model"],
        "injection_char": t["injection_char"],
        "injection_token_id": t["injection_token_id"],
        "left_id": t["injection_left_neighbor_id"],
        "right_id": t["injection_right_neighbor_id"],
        "injection_scale": float(meta["extraction"]["injection_scale"]),
        "actor_prompt_template": meta["prompt_templates"]["av"],
    }


def build_av_prompt_ids(tok, sidecar) -> tuple[list[int], int]:
    content = sidecar["actor_prompt_template"].format(
        injection_char=sidecar["injection_char"])
    out = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    ids = list(out["input_ids"]) if hasattr(out, "keys") else list(out)
    matches = [i for i, t in enumerate(ids)
               if t == sidecar["injection_token_id"]
               and i > 0 and i < len(ids) - 1
               and ids[i - 1] == sidecar["left_id"]
               and ids[i + 1] == sidecar["right_id"]]
    assert len(matches) == 1, f"expected 1 injection site, got {len(matches)}"
    return ids, matches[0]


@torch.no_grad()
def batched_verbalize(
    model, tok, embed_layer, prompt_ids: list[int], inj_pos: int,
    activations: np.ndarray, injection_scale: float,
    *, max_new_tokens: int = 200, temperature: float = 1.0, top_p: float = 1.0,
    device: str = "cuda:0",
) -> list[str]:
    """activations: [B, d_model] fp32.  Returns list of raw text generations."""
    B = activations.shape[0]
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    base_embeds = embed_layer(ids).float()  # [T, d]
    T = base_embeds.shape[0]
    embeds = base_embeds.unsqueeze(0).expand(B, -1, -1).clone()  # [B, T, d]

    # Rescale + inject
    v = torch.from_numpy(activations).to(device).float()  # [B, d]
    norms = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v_scaled = v * (injection_scale / norms)
    embeds[:, inj_pos, :] = v_scaled

    model_dtype = next(model.parameters()).dtype
    embeds = embeds.to(model_dtype)
    attn_mask = torch.ones(B, T, device=device, dtype=torch.long)
    gen = model.generate(
        inputs_embeds=embeds, attention_mask=attn_mask,
        do_sample=temperature > 0.0,
        temperature=temperature if temperature > 0.0 else 1.0,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    # When inputs_embeds is provided, HF returns ONLY new tokens (no prompt prefix).
    return tok.batch_decode(gen, skip_special_tokens=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", default=str(EXP_DIR / "pd_full_acts.npz"))
    parser.add_argument("--av-model", default=DEFAULT_AV)
    parser.add_argument("--out", default=str(EXP_DIR / "pd_full_nla.jsonl"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--positions", nargs="+", default=["qmark", "last"])
    args = parser.parse_args()

    av_path = resolve_local_snapshot(args.av_model)
    print(f"[load] AV {args.av_model} → {av_path}")
    sidecar = load_sidecar(av_path)
    tok = AutoTokenizer.from_pretrained(av_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        av_path, dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    embed_layer = model.get_input_embeddings()

    prompt_ids, inj_pos = build_av_prompt_ids(tok, sidecar)
    print(f"[load] AV prompt len={len(prompt_ids)} inj_pos={inj_pos} "
          f"inj_scale={sidecar['injection_scale']}")

    print(f"[load] activations {args.acts}")
    acts = np.load(args.acts, allow_pickle=False)
    n_q = len(acts["qid"])
    print(f"[load] {n_q} questions")

    # Build the work list: (record_index_in_acts, position_name, vec)
    pos_to_key = {"qmark": "acts_qmark", "last": "acts_last"}
    work: list[tuple[int, str]] = []
    for pos_name in args.positions:
        assert pos_name in pos_to_key, f"unknown position {pos_name!r}"
        for i in range(n_q):
            work.append((i, pos_name))
    print(f"[gen] total NLA inferences: {len(work)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = out_path.open("w")

    t0 = time.time()
    for start in range(0, len(work), args.batch_size):
        batch = work[start:start + args.batch_size]
        positions = [pn for _, pn in batch]
        # Could be mixed positions; group by position so we vectorise the lookup
        # Simplest: just gather per item (already fast since acts is in RAM).
        vecs = np.stack([acts[pos_to_key[pn]][i] for i, pn in batch]).astype(np.float32)

        texts = batched_verbalize(
            model, tok, embed_layer, prompt_ids, inj_pos, vecs,
            sidecar["injection_scale"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
        )
        for (idx, pn), txt in zip(batch, texts):
            m = EXPLANATION_RE.search(txt)
            explanation = m.group(1).strip() if m else None
            f_out.write(json.dumps({
                "qid":        str(acts["qid"][idx]),
                "bucket":     str(acts["bucket"][idx]),
                "x_name":     str(acts["x_name"][idx]),
                "y_name":     str(acts["y_name"][idx]),
                "expected_answer": str(acts["expected_answer"][idx]),
                "position":   pn,
                "explanation": explanation,
                "raw_text":   txt,
            }, ensure_ascii=False) + "\n")
        f_out.flush()
        done = start + len(batch)
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        eta_min = (len(work) - done) / max(rate, 1e-6) / 60.0
        print(f"[gen] {done}/{len(work)} done ({rate:.2f}/s, ETA {eta_min:.1f} min)",
              flush=True)
    f_out.close()
    print(f"[done] wrote {out_path} ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
