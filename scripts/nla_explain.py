"""NLA explainer for IPHR / chain-of-thought experiments.

End-to-end: given (prompt_text, token_position) pairs, extract layer-20
activations from Qwen2.5-7B-Instruct and verbalize them with the released NLA
AV (kitft/nla-qwen2.5-7b-L20-av). HuggingFace transformers throughout — no
SGLang dependency.

Usage:
    from nla_explain import NLAExplainer
    expl = NLAExplainer()
    out = expl.explain([
        ("How are you today?", -1),       # last token of the prompt
        ("The quick brown fox.", 3),
    ])
    for text in out:
        print(text)

Sidecar contract (kitft/nla-qwen2.5-7b-L20-av/nla_meta.yaml):
- d_model=3584, layer 20 of 28.
- injection_char='㈎', injection_token_id=149705, injection_scale=150.0
- Left/right neighbors (29, 522) gate against the injection char appearing
  elsewhere in the prompt; we verify them per the reference impl.

If the AV outputs nothing but CJK on English activations, the injection path
broke — check the sidecar values against the live tokenizer.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_AV_MODEL = "kitft/nla-qwen2.5-7b-L20-av"
DEFAULT_LAYER = 20


@dataclass(frozen=True)
class NLASidecar:
    d_model: int
    layer: int
    injection_char: str
    injection_token_id: int
    left_id: int
    right_id: int
    injection_scale: float
    actor_prompt_template: str


def load_sidecar(av_dir: str | Path) -> NLASidecar:
    meta = yaml.safe_load((Path(av_dir) / "nla_meta.yaml").read_text())
    t = meta["tokens"]
    return NLASidecar(
        d_model=meta["d_model"],
        layer=meta["extraction_layer_index"],
        injection_char=t["injection_char"],
        injection_token_id=t["injection_token_id"],
        left_id=t["injection_left_neighbor_id"],
        right_id=t["injection_right_neighbor_id"],
        injection_scale=float(meta["extraction"]["injection_scale"]),
        actor_prompt_template=meta["prompt_templates"]["av"],
    )


def resolve_local_path(repo_id: str) -> str:
    """Resolve to a local HF cache snapshot dir if present, otherwise return repo_id."""
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
    cache_dir = Path(hf_home) / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if cache_dir.exists():
        snaps = sorted(cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snaps:
            return str(snaps[0])
    return repo_id


class NLAExplainer:
    """Joint host for base model (Qwen2.5-7B-Instruct) and the NLA AV.

    Both Qwen-architecture models fit on a single H200 in bf16 (~14GB each).
    By default base + AV both go on cuda:0; pass distinct devices for split.
    """

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        av_model: str = DEFAULT_AV_MODEL,
        base_device: str = "cuda:0",
        av_device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        layer: int | None = None,
    ) -> None:
        base_path = resolve_local_path(base_model)
        av_path = resolve_local_path(av_model)

        self.sidecar = load_sidecar(av_path)
        self.layer = layer if layer is not None else self.sidecar.layer

        self.base_device = base_device
        self.av_device = av_device

        print(f"[NLAExplainer] loading base {base_model} → {base_device}")
        self.base_tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        self.base = AutoModelForCausalLM.from_pretrained(
            base_path, dtype=dtype, device_map=base_device, trust_remote_code=True,
        )
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)

        print(f"[NLAExplainer] loading AV {av_model} → {av_device}")
        self.av_tok = AutoTokenizer.from_pretrained(av_path, trust_remote_code=True)
        self.av = AutoModelForCausalLM.from_pretrained(
            av_path, dtype=dtype, device_map=av_device, trust_remote_code=True,
        )
        self.av.eval()
        for p in self.av.parameters():
            p.requires_grad_(False)

        d_model = self.av.config.hidden_size
        assert d_model == self.sidecar.d_model, (
            f"AV hidden_size {d_model} != sidecar d_model {self.sidecar.d_model}"
        )

        self._validate_sidecar_against_tokenizer()
        self._cached_av_prompt_ids = self._build_av_prompt_ids()
        self._injection_pos = self._find_injection_position(self._cached_av_prompt_ids)

        print(
            f"[NLAExplainer] ready: layer={self.layer} d_model={d_model} "
            f"inj_char={self.sidecar.injection_char!r} "
            f"inj_token_id={self.sidecar.injection_token_id} "
            f"inj_pos_in_av_prompt={self._injection_pos} "
            f"inj_scale={self.sidecar.injection_scale}"
        )

    def _validate_sidecar_against_tokenizer(self) -> None:
        ids = self.av_tok.encode(self.sidecar.injection_char, add_special_tokens=False)
        assert ids == [self.sidecar.injection_token_id], (
            f"tokenizer drift: {self.sidecar.injection_char!r} -> {ids}, "
            f"sidecar says [{self.sidecar.injection_token_id}]"
        )

    def _build_av_prompt_ids(self) -> list[int]:
        content = self.sidecar.actor_prompt_template.format(
            injection_char=self.sidecar.injection_char
        )
        out = self.av_tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True,
        )
        # transformers 5.9 returns a BatchEncoding (Mapping, not dict subclass)
        # with input_ids+attention_mask. Earlier versions returned a flat list.
        if hasattr(out, "keys"):
            return list(out["input_ids"])
        return list(out)

    def _find_injection_position(self, ids: list[int]) -> int:
        matches = [
            i for i, t in enumerate(ids)
            if t == self.sidecar.injection_token_id
            and i > 0 and i < len(ids) - 1
            and ids[i - 1] == self.sidecar.left_id
            and ids[i + 1] == self.sidecar.right_id
        ]
        assert len(matches) == 1, (
            f"expected exactly one valid injection position in canonical AV prompt, "
            f"got {len(matches)}"
        )
        return matches[0]

    # ─── Extraction ────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract(self, prompt: str, position: int) -> torch.Tensor:
        """Return layer-`self.layer` activation at `position` (negatives OK) as
        an fp32 [d_model] tensor on CPU."""
        enc = self.base_tok(prompt, return_tensors="pt").to(self.base_device)
        out = self.base(**enc, output_hidden_states=True, use_cache=False)
        # hidden_states is a tuple len = n_layers+1 (incl. embeddings).
        # Convention from the inference docs: hidden_states[20] = output of
        # block index 19 — i.e. residual stream AFTER 20 transformer blocks.
        h = out.hidden_states[self.layer][0]  # [seq, d_model]
        seq_len = h.shape[0]
        if position < 0:
            position = seq_len + position
        assert 0 <= position < seq_len, (
            f"position {position} out of range for seq_len {seq_len}"
        )
        return h[position].float().cpu()

    @torch.no_grad()
    def extract_batch(self, pairs: list[tuple[str, int]]) -> list[torch.Tensor]:
        return [self.extract(p, pos) for p, pos in pairs]

    # ─── Verbalization (AV inference) ──────────────────────────────────────

    @staticmethod
    def _normalize(v: torch.Tensor, scale: float) -> torch.Tensor:
        norm = v.float().norm().clamp_min(1e-12)
        return (v.float() * (scale / norm))

    @torch.no_grad()
    def verbalize(
        self,
        activation: torch.Tensor,
        *,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 1.0,
        extract_explanation: bool = True,
    ) -> str:
        """Run the AV on a single [d_model] activation vector."""
        v = activation.float().view(-1)
        assert v.numel() == self.sidecar.d_model

        ids = torch.tensor(self._cached_av_prompt_ids, dtype=torch.long,
                           device=self.av_device).unsqueeze(0)
        embed_layer = self.av.get_input_embeddings()
        embeds = embed_layer(ids).float()  # [1, T, d] in fp32 for inject math

        v_scaled = self._normalize(v, self.sidecar.injection_scale).to(self.av_device)
        embeds[0, self._injection_pos] = v_scaled

        embeds_dtype = embeds.to(next(self.av.parameters()).dtype)

        attn_mask = torch.ones_like(ids)
        gen = self.av.generate(
            inputs_embeds=embeds_dtype,
            attention_mask=attn_mask,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.av_tok.eos_token_id,
        )
        # When using inputs_embeds, HF returns ONLY the new tokens (no echo
        # of the prompt), so we decode the full output.
        text = self.av_tok.decode(gen[0], skip_special_tokens=False)

        if not extract_explanation:
            return text
        m = EXPLANATION_RE.search(text)
        if m is None:
            print(f"[NLAExplainer] WARNING: no <explanation> tags. raw[:200]={text[:200]!r}")
            return text
        return m.group(1).strip()

    def explain(
        self,
        pairs: list[tuple[str, int]],
        **sampling,
    ) -> list[str]:
        """End-to-end: (prompt, token_position) → NLA explanation, per pair."""
        acts = self.extract_batch(pairs)
        return [self.verbalize(a, **sampling) for a in acts]


# ─── Smoke test ───────────────────────────────────────────────────────────────


def _smoke_test() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--base-device", default="cuda:0")
    parser.add_argument("--av-device", default="cuda:0")
    args = parser.parse_args()

    pairs = [
        ("The quick brown fox jumps over the lazy dog.", -1),
        ("Paris is the capital of France.", -1),
        ("def fibonacci(n):\n    if n < 2:\n        return n\n", -1),
        ("Q: Is the year 1999 later than the year 2000? A:", -1),
    ]

    expl = NLAExplainer(base_device=args.base_device, av_device=args.av_device)
    outs = expl.explain(
        pairs,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    for (prompt, pos), text in zip(pairs, outs):
        print("=" * 78)
        print(f"PROMPT (pos={pos}): {prompt!r}")
        print(f"EXPLANATION:\n{text}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
