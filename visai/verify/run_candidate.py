#!/usr/bin/env python3
"""Subprocess entry: gate and/or benchmark one MLX candidate. stdout is JSON.

Runs out-of-process so a hung or faulting Metal kernel cannot take down the agent.
The hidden-test seed arrives only via VISAI_HIDDEN_SEED from the parent.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mlx.core as mx

from visai.bench.runner import time_fn
from visai.jsonio import dump
from visai.tasks.ops import DTYPES, get_op
from visai.verify.kernel_gate import format_table, load_candidate, run_gate


def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(max(x, 1e-9)) for x in xs) / len(xs)) if xs else 0.0


def bench_all(op_name: str, candidate: Path | None, dtype: str) -> dict:
    """Bench every bench shape; summary times are geometric means across shapes."""
    op = get_op(op_name)
    per_shape = [bench(op_name, candidate, dtype, shape) for shape in op.bench_shapes]
    out: dict = {"dtype": dtype, "shapes": per_shape}
    for key in ("reference_us", "runtime_us", "candidate_us"):
        vals = [s[key] for s in per_shape if key in s]
        if vals:
            out[key] = _geomean(vals)
    if candidate is not None:
        out["timed_output_ok"] = all(s.get("timed_output_ok") for s in per_shape)
    return out


def bench(op_name: str, candidate: Path | None, dtype: str, shape: dict) -> dict:
    op = get_op(op_name)
    args, kwargs = op.make_inputs(shape, DTYPES[dtype], 2024, 1.0)
    out: dict = {"shape": shape, "dtype": dtype}
    out["reference_us"] = time_fn(op.reference, args, kwargs)["median_us"]
    out["runtime_us"] = time_fn(op.runtime, args, kwargs)["median_us"]
    if candidate is not None:
        kernel = load_candidate(candidate).kernel
        out["candidate_us"] = time_fn(kernel, args, kwargs)["median_us"]
        # re-check the exact tensors that were timed (guards against timing-only shortcuts)
        ref = op.reference(*args, **kwargs)
        got = kernel(*args, **kwargs)
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = got if isinstance(got, (tuple, list)) else (got,)
        out["timed_output_ok"] = all(
            bool(mx.allclose(g.astype(mx.float32), r.astype(mx.float32), atol=2e-2, rtol=2e-2).item())
            for g, r in zip(got, ref)
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--mode", choices=("gate", "bench", "all"), default="all")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    op = get_op(args.op)
    payload: dict = {"op": args.op, "dtype": args.dtype}
    pretty = None
    if args.mode in ("gate", "all"):
        if not args.candidate:
            raise SystemExit("--candidate required for gate")
        gate = run_gate(op, args.candidate, args.dtype, args.seed)
        payload["correctness"] = gate
        pretty = format_table(gate)
        if gate["status"] != "pass":
            dump(payload, None, pretty)
            return 1
    if args.mode in ("bench", "all"):
        payload["bench"] = bench_all(args.op, args.candidate, args.dtype)
    dump(payload, None, pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
