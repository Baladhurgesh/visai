"""mlx-lm model helpers: load (optionally quantized), measure decode speed, profile op families."""

from __future__ import annotations

import importlib
import statistics
import time
from collections import defaultdict
from typing import Any

import mlx.core as mx
import mlx.nn as nn

DEFAULT_PROMPT = (
    "You are solving grade-school math problems. Show every step. "
    "Problem: A store sells apples at $2 each and oranges at $3 each. "
    "Maya buys 14 apples and 9 oranges, then returns 3 apples. How much does she pay in total? "
)


def load(model_id: str, quant: dict[str, Any] | None = None):
    """Load an mlx-lm model; quant = {"bits": 4, "group_size": 64, "keep_high": [...]} quantizes in memory."""
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(model_id)
    if quant and quant.get("bits"):
        bits, group = int(quant["bits"]), int(quant.get("group_size", 64))
        keep = tuple(quant.get("keep_high") or ())

        def predicate(path: str, module: nn.Module) -> bool:
            if not hasattr(module, "to_quantized"):
                return False
            if any(k in path for k in keep):
                return False
            w = getattr(module, "weight", None)
            return w is not None and w.shape[-1] % group == 0

        nn.quantize(model, group_size=group, bits=bits, class_predicate=predicate)
        mx.eval(model.parameters())
    return model, tokenizer


def model_type(model) -> str:
    return getattr(getattr(model, "args", None), "model_type", "") or model.__class__.__module__.split(".")[-1]


def _prompt_tokens(tokenizer, prompt: str, input_tokens: int | None) -> list[int]:
    ids = tokenizer.encode(prompt)
    if input_tokens:
        while len(ids) < input_tokens:
            ids = ids + ids
        ids = ids[:input_tokens]
    return ids


def measure_decode(
    model,
    tokenizer,
    *,
    prompt: str = DEFAULT_PROMPT,
    input_tokens: int = 256,
    output_tokens: int = 128,
    runs: int = 3,
    gen_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Median decode/prefill tok/s over `runs` greedy generations (kernel-forge Baseline shape)."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    gen_kwargs = dict(gen_kwargs or {})
    ids = _prompt_tokens(tokenizer, prompt, input_tokens)
    sampler = make_sampler(temp=0.0)
    samples = []
    for i in range(runs + 1):  # first run is warmup
        last = None
        t0 = time.perf_counter()
        for resp in stream_generate(model, tokenizer, ids, max_tokens=output_tokens, sampler=sampler, **gen_kwargs):
            last = resp
        dt = time.perf_counter() - t0
        if i == 0 or last is None:
            continue
        samples.append(
            {
                "total_s": dt,
                "prompt_tokens": last.prompt_tokens,
                "completion_tokens": last.generation_tokens,
                "prefill_tok_s": last.prompt_tps,
                "decode_tok_s": last.generation_tps,
                "peak_memory_gb": last.peak_memory,
            }
        )
    med = lambda k: statistics.median(s[k] for s in samples)  # noqa: E731
    return {
        "warmup": 1,
        "timed_runs": len(samples),
        "median_decode_tok_s": med("decode_tok_s"),
        "median_prefill_tok_s": med("prefill_tok_s"),
        "median_total_s": med("total_s"),
        "peak_memory_gb": max(s["peak_memory_gb"] for s in samples),
        "runs": samples,
    }


def profile_families(model, tokenizer, *, input_tokens: int = 128, decode_steps: int = 32) -> list[dict]:
    """Per-op timing of decode steps. Returns rows {name, time_ns, launches} for summarize_rows().

    Each instrumented op is synchronized (inputs evaluated untimed, then op timed with mx.eval), so
    the split is indicative, not exact. Everything unattributed is reported as residual/elementwise.
    """
    from mlx_lm.models.cache import make_prompt_cache

    times: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    saved: list[tuple[Any, str, Any]] = []

    def _flat(x):
        return [a for a in (x if isinstance(x, (tuple, list)) else [x]) if isinstance(a, mx.array)]

    def timed(label: str, fn):
        def wrapper(*args, **kwargs):
            mx.eval(*[a for a in args if isinstance(a, mx.array)])
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            mx.eval(*_flat(out))
            times[label] += time.perf_counter() - t0
            counts[label] += 1
            return out

        return wrapper

    def patch(owner, attr, label):
        orig = getattr(owner, attr)
        saved.append((owner, attr, orig))
        setattr(owner, attr, timed(label, orig))

    mod = importlib.import_module(model.__class__.__module__)
    patch(nn.Linear, "__call__", "gemm:Linear")
    patch(nn.QuantizedLinear, "__call__", "gemm:QuantizedLinear")
    patch(nn.RMSNorm, "__call__", "rmsnorm:RMSNorm")
    patch(nn.RoPE, "__call__", "rope:RoPE")
    if hasattr(mod, "scaled_dot_product_attention"):
        patch(mod, "scaled_dot_product_attention", "attention:scaled_dot_product_attention")
    if hasattr(mod, "swiglu"):
        patch(mod, "swiglu", "swiglu:swiglu")

    step_total = 0.0
    try:
        cache = make_prompt_cache(model)
        ids = mx.array(_prompt_tokens(tokenizer, DEFAULT_PROMPT, input_tokens))[None]
        logits = model(ids, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        times.clear()
        counts.clear()
        for _ in range(decode_steps):
            t0 = time.perf_counter()
            logits = model(tok[:, None], cache=cache)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            step_total += time.perf_counter() - t0
    finally:
        for owner, attr, orig in reversed(saved):
            setattr(owner, attr, orig)

    attributed = sum(times.values())
    rows = [{"name": k, "time_ns": v * 1e9, "launches": counts[k]} for k, v in times.items()]
    rows.append(
        {
            "name": "residual_add+cache_update+embedding+sampling (unattributed elementwise)",
            "time_ns": max(step_total - attributed, 0.0) * 1e9,
            "launches": decode_steps,
        }
    )
    return rows
