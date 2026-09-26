"""Deployment-config evaluation for mlx-lm models (Visai addition; no Gate 1 because no kernel changes)."""

from __future__ import annotations

import gc
from typing import Any

import mlx.core as mx

from visai.models.mlx_model import load, measure_decode
from visai.verify.perplexity import perplexity

ALLOWED = {
    "bits": {None, 3, 4, 5, 6, 8},
    "group_size": {32, 64, 128},
    "kv_bits": {None, 4, 8},
    "kv_group_size": {32, 64},
}

SEARCH_SPACE_DOC = """Deployment config keys (all optional; omitted = baseline behaviour):
- bits: weight quantization bits for Linear layers: 3 | 4 | 5 | 6 | 8 | null (keep bf16)
- group_size: quantization group size: 32 | 64 | 128
- keep_high: list of module-path substrings kept in bf16 (mixed precision), e.g. ["lm_head", "embed_tokens",
  "layers.0.", "layers.27.", "mlp.down_proj"]
- kv_bits: KV-cache quantization: 4 | 8 | null ; kv_group_size: 32 | 64
- prefill_step_size: 256..4096 (prefill chunking)
- kernels: list of promoted kernel ops to patch in, e.g. ["add_rmsnorm"] (only ops with a promoted winner)
"""


def validate(config: dict[str, Any]) -> list[str]:
    errs = []
    for k in ("bits", "group_size", "kv_bits", "kv_group_size"):
        if k in config and config[k] not in ALLOWED[k]:
            errs.append(f"{k}={config[k]!r} not in {sorted(v for v in ALLOWED[k] if v is not None)} (or null)")
    if "prefill_step_size" in config and config["prefill_step_size"] is not None:
        v = int(config["prefill_step_size"])
        if not 256 <= v <= 4096:
            errs.append("prefill_step_size must be in [256, 4096]")
    if config.get("keep_high") and not isinstance(config["keep_high"], list):
        errs.append("keep_high must be a list of strings")
    unknown = set(config) - {"bits", "group_size", "keep_high", "kv_bits", "kv_group_size", "prefill_step_size", "kernels"}
    if unknown:
        errs.append(f"unknown keys: {sorted(unknown)}")
    return errs


def gen_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if config.get("kv_bits"):
        kw["kv_bits"] = int(config["kv_bits"])
        kw["kv_group_size"] = int(config.get("kv_group_size") or 64)
    if config.get("prefill_step_size"):
        kw["prefill_step_size"] = int(config["prefill_step_size"])
    return kw


def free() -> None:
    gc.collect()
    mx.clear_cache()


def evaluate_config(
    model_id: str,
    config: dict[str, Any],
    *,
    patcher_factory=None,
    input_tokens: int = 256,
    output_tokens: int = 128,
    runs: int = 3,
) -> dict[str, Any]:
    """Load model with config, measure decode speed and perplexity. Returns candidate metrics."""
    quant = {"bits": config.get("bits"), "group_size": config.get("group_size", 64), "keep_high": config.get("keep_high")}
    model, tok = load(model_id, quant if config.get("bits") else None)
    patcher = None
    try:
        if patcher_factory and config.get("kernels"):
            patcher = patcher_factory(model, config["kernels"])
            patcher.apply()
        speed = measure_decode(model, tok, input_tokens=input_tokens, output_tokens=output_tokens, runs=runs,
                               gen_kwargs=gen_kwargs(config))
        q = perplexity(model, tok)
        return {
            "throughput": speed["median_decode_tok_s"],
            "unit": "decode tok/s",
            "prefill_tok_s": speed["median_prefill_tok_s"],
            "peak_memory_gb": speed["peak_memory_gb"],
            "perplexity": q["perplexity"],
            "quality": q["quality"],
            "n_errors": 0,
        }
    finally:
        if patcher:
            patcher.revert()
        del model, tok
        free()


def remeasure_speed(model_id: str, config: dict[str, Any], patcher_factory=None) -> float:
    quant = {"bits": config.get("bits"), "group_size": config.get("group_size", 64), "keep_high": config.get("keep_high")}
    model, tok = load(model_id, quant if config.get("bits") else None)
    patcher = None
    try:
        if patcher_factory and config.get("kernels"):
            patcher = patcher_factory(model, config["kernels"])
            patcher.apply()
        return measure_decode(model, tok, runs=3, gen_kwargs=gen_kwargs(config))["median_decode_tok_s"]
    finally:
        if patcher:
            patcher.revert()
        del model, tok
        free()
