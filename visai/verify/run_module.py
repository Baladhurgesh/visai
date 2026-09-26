#!/usr/bin/env python3
"""Subprocess: Gate 1 + benchmark for a layer-kernel candidate against the model's own layer.

Reference = the layer class's original __call__ on activations captured from the real model.
Hidden cases (seed from VISAI_HIDDEN_SEED): scaled x3, tiny x0.01, negated, and fresh same-statistics noise.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from pathlib import Path

import mlx.core as mx

from visai.backends.module_patch import load_forward, resolve_class
from visai.bench.runner import time_fn
from visai.jsonio import dump
from visai.models.adapters import get_adapter
from visai.profiler.layers import load_capture

ALLOWED_PREFIXES = ("mlx", "math", "functools", "typing", "mlx_lm.models", "mlx_audio.vad.models", "parakeet_mlx", "dataclasses")
FORBIDDEN = {"open", "exec", "eval", "__import__", "compile", "globals", "locals", "setattr", "delattr", "breakpoint", "input"}
TOL = {"mlx.core.bfloat16": 2.5e-2, "mlx.core.float16": 1e-2, "mlx.core.float32": 1e-3}


def static_checks(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems, fwd = [], None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                if not n.startswith(ALLOWED_PREFIXES):
                    problems.append(f"import not allowed: {n}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN:
            problems.append(f"forbidden builtin: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr == "__call__":
            problems.append("calling __call__ directly is not allowed (it recurses once the layer is patched)")
        elif isinstance(node, ast.Global):
            problems.append("global state is not allowed")
        elif isinstance(node, ast.FunctionDef) and node.name == "forward":
            fwd = node
    if fwd is None:
        return problems + ["candidate must define forward(module, *args, **kwargs)"]
    first = fwd.args.args[0].arg if fwd.args.args else None
    for node in ast.walk(fwd):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (first, "type"):
            problems.append(f"calling the layer itself ({node.func.id}(...)) would recurse once patched; call its children or ops")
    return problems


def _flat(out) -> list:
    if isinstance(out, mx.array):
        return [out]
    if isinstance(out, (tuple, list)):
        return [a for o in out for a in _flat(o)]
    return []


def _compare(ref, cand, tol: float) -> dict:
    r, c = _flat(ref), _flat(cand)
    if len(r) != len(c):
        return {"pass": False, "error": f"expected {len(r)} array outputs, got {len(c)}"}
    worst = 0.0
    for a, b in zip(r, c):
        if tuple(a.shape) != tuple(b.shape):
            return {"pass": False, "error": f"shape {tuple(b.shape)} != {tuple(a.shape)}"}
        if a.dtype != b.dtype:
            return {"pass": False, "error": f"dtype {b.dtype} != {a.dtype}"}
        af, bf = a.astype(mx.float32), b.astype(mx.float32)
        if bool(mx.all(mx.isfinite(af)).item()) and not bool(mx.all(mx.isfinite(bf)).item()):
            return {"pass": False, "error": "non-finite output where reference is finite"}
        peak = float(mx.max(mx.abs(af)).item()) if af.size else 0.0
        err = float(mx.max(mx.abs(af - bf)).item()) / max(peak, 1.0) if af.size else 0.0
        worst = max(worst, err)
    return {"pass": worst <= tol, "rel_err": worst, "tol": tol}


def _perturb(args: list, kind: str, seed: int) -> list:
    out, done = [], False
    for a in args:
        if not done and isinstance(a, mx.array) and mx.issubdtype(a.dtype, mx.floating):
            if kind == "x3":
                a = a * 3
            elif kind == "small":
                a = a * 0.01
            elif kind == "negated":
                a = -a
            elif kind == "noise":
                af = a.astype(mx.float32)
                a = (mx.random.normal(a.shape, key=mx.random.key(seed)) * mx.std(af) + mx.mean(af)).astype(a.dtype)
            done = True
        out.append(a)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--config", default="{}")
    p.add_argument("--cls", required=True)
    p.add_argument("--captures", required=True, help="comma-separated capture stems")
    p.add_argument("--candidate", type=Path)
    p.add_argument("--mode", choices=("gate", "bench", "all"), default="all")
    a = p.parse_args()

    adapter = get_adapter(a.adapter)
    h = adapter.load(adapter.model_cfg(json.loads(a.config)))
    modules = dict(adapter.root(h).named_modules())
    orig = resolve_class(a.cls).__call__
    caps = [load_capture(Path(c)) for c in a.captures.split(",") if c]
    payload: dict = {"tool": "layer_correctness", "class": a.cls, "n_captures": len(caps)}

    fwd = None
    if a.candidate:
        problems = static_checks(a.candidate.read_text())
        if problems:
            payload["correctness"] = {"status": "rejected", "problems": problems, "n": 0, "n_pass": 0}
            dump(payload)
            return 1
        try:
            fwd = load_forward(str(a.candidate))
        except Exception as exc:  # noqa: BLE001
            payload["correctness"] = {"status": "compile_error", "error": f"{type(exc).__name__}: {exc}"}
            dump(payload)
            return 1

    if a.mode in ("gate", "all") and fwd is not None:
        hidden_seed = int(os.environ.get("VISAI_HIDDEN_SEED") or 7)
        cases = []
        for i, (args, kwargs, spec) in enumerate(caps):
            mod = modules[spec["module_path"]]
            out_dtype = str(_flat(orig(mod, *args, **kwargs))[0].dtype)
            tol = TOL.get(out_dtype, 2.5e-2)
            for kind in ("real", "x3", "small", "negated", "noise"):
                hidden = kind != "real"
                pargs = args if kind == "real" else _perturb(args, kind, hidden_seed + i)
                rec = {"case": f"capture{i}_{kind}" if not hidden else f"hidden_{kind}", "hidden": hidden}
                try:
                    ref = orig(mod, *pargs, **kwargs)
                    mx.eval(*_flat(ref))
                except Exception:  # noqa: BLE001  perturbation invalid for this layer
                    continue
                try:
                    cand = fwd(mod, *pargs, **kwargs)
                    mx.eval(*_flat(cand))
                    rec.update(_compare(ref, cand, tol))
                except Exception as exc:  # noqa: BLE001
                    rec.update({"pass": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                if hidden:
                    rec.pop("rel_err", None)
                cases.append(rec)
        n_ok = sum(1 for c in cases if c.get("pass"))
        payload["correctness"] = {
            "status": "pass" if cases and n_ok == len(cases) else "fail",
            "n": len(cases), "n_pass": n_ok, "n_fail": len(cases) - n_ok,
            "n_errors": sum(1 for c in cases if c.get("error")),
            "n_hidden_fail": sum(1 for c in cases if c["hidden"] and not c.get("pass")),
            "hidden_failure_kinds": sorted({c["case"] for c in cases if c["hidden"] and not c.get("pass")}),
            "visible_failures": [c for c in cases if not c["hidden"] and not c.get("pass")][:4],
            "first_error": next((c["error"] for c in cases if c.get("error")), None),
        }
        if payload["correctness"]["status"] != "pass":
            dump(payload)
            return 1

    if a.mode in ("bench", "all"):
        per = []
        for i, (args, kwargs, spec) in enumerate(caps):
            mod = modules[spec["module_path"]]
            row = {"capture": i, "signature": spec.get("signature", "")[:160],
                   "runtime_us": time_fn(lambda *x, **k: orig(mod, *x, **k), args, kwargs, inner=20)["median_us"]}
            if fwd is not None:
                row["candidate_us"] = time_fn(lambda *x, **k: fwd(mod, *x, **k), args, kwargs, inner=20)["median_us"]
                ref, got = orig(mod, *args, **kwargs), fwd(mod, *args, **kwargs)
                row["timed_output_ok"] = _compare(ref, got, TOL.get(str(_flat(ref)[0].dtype), 2.5e-2))["pass"]
            per.append(row)
        gm = lambda k: math.exp(sum(math.log(max(r[k], 1e-9)) for r in per) / len(per))  # noqa: E731
        bench = {"shapes": per, "runtime_us": gm("runtime_us"), "reference_us": gm("runtime_us")}
        if fwd is not None:
            bench["candidate_us"] = gm("candidate_us")
            bench["timed_output_ok"] = all(r.get("timed_output_ok") for r in per)
        payload["bench"] = bench
    dump(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
