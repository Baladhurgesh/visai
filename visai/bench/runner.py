"""Microbenchmark runner for MLX callables.

Calls are batched K at a time into one lazy graph and evaluated together, so per-call
Python / eval overhead does not dominate small memory-bound kernels.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import mlx.core as mx


def _flatten(out: Any) -> list:
    return list(out) if isinstance(out, (tuple, list)) else [out]


def time_fn(
    fn: Callable[..., Any],
    args: list,
    kwargs: dict | None = None,
    *,
    warmup: int = 5,
    reps: int = 15,
    inner: int = 50,
) -> dict[str, float]:
    kwargs = kwargs or {}
    mx.eval(*args)
    for _ in range(warmup):
        mx.eval(*_flatten(fn(*args, **kwargs)))
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = []
        for _ in range(inner):
            outs.extend(_flatten(fn(*args, **kwargs)))
        mx.eval(*outs)
        samples.append((time.perf_counter() - t0) / inner * 1e6)
    samples.sort()
    return {
        "median_us": statistics.median(samples),
        "p10_us": samples[max(0, len(samples) // 10)],
        "p90_us": samples[min(len(samples) - 1, (len(samples) * 9) // 10)],
    }
