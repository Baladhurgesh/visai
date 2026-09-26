"""Gate 1: kernel correctness (generalized from kernel-forge lib/gdn_correctness.py).

Same shape as kernel-forge: load candidate by path, run the same seeded tensors through
reference and candidate, report max_abs / mean_abs / max_rel per case, status=pass only if
every case passes. Added (KernelBench-Verified style):
- hidden shapes the writer never sees, drawn with a hidden seed from the parent process
- scaled / tiny / negated input distributions
- static anti-cheat checks on the candidate source
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import mlx.core as mx

from visai.tasks.ops import DISTRIBUTIONS, DTYPES, TOL, OpSpec

ALLOWED_IMPORTS = {"mlx", "mlx.core", "mlx.nn", "math", "functools", "typing", "mlx_lm.models.activations"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "__import__", "compile", "globals", "locals", "vars", "setattr",
    "getattr", "delattr", "breakpoint", "input",
}
FORBIDDEN_ATTRS = {"system", "popen", "_reference", "reference"}


def static_checks(source: str) -> list[str]:
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    has_kernel = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    problems.append(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod not in ALLOWED_IMPORTS:
                problems.append(f"import not allowed: from {mod}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"forbidden builtin: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"forbidden attribute: .{node.attr}")
        elif isinstance(node, ast.Global):
            problems.append("global state (possible output caching) is not allowed")
        elif isinstance(node, ast.FunctionDef) and node.name == "kernel":
            has_kernel = True
    if not has_kernel:
        problems.append("candidate must define a top-level function `kernel(...)`")
    return problems


def load_candidate(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"visai_cand_{abs(hash(str(path)))}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "kernel"):
        raise RuntimeError("candidate has no kernel()")
    return mod


def _as_tuple(out: Any) -> tuple:
    return tuple(out) if isinstance(out, (tuple, list)) else (out,)


def _stats(a: mx.array, b: mx.array) -> dict[str, float]:
    af, bf = a.astype(mx.float32), b.astype(mx.float32)
    d = mx.abs(af - bf)
    denom = mx.maximum(mx.abs(bf), 1e-6)
    return {
        "max_abs": float(mx.max(d).item()),
        "mean_abs": float(mx.mean(d).item()),
        "max_rel": float(mx.max(d / denom).item()),
    }


def run_case(op: OpSpec, kernel, shape: dict, dtype: str, seed: int, dist: str, hidden: bool) -> dict[str, Any]:
    name = "_".join(f"{k}{v}" for k, v in shape.items()) + f"_{dist}"
    rec: dict[str, Any] = {"name": name if not hidden else f"hidden_{dist}", "hidden": hidden, "pass": False}
    atol, rtol = TOL[dtype]
    try:
        args, kwargs = op.make_inputs(shape, DTYPES[dtype], seed, DISTRIBUTIONS[dist])
        ref = _as_tuple(op.reference(*args, **kwargs))
        cand = _as_tuple(kernel(*args, **kwargs))
        mx.eval(*ref, *cand)
        if len(ref) != len(cand):
            rec["error"] = f"expected {len(ref)} outputs, got {len(cand)}"
            return rec
        ok = True
        outs = []
        for i, (r, c) in enumerate(zip(ref, cand)):
            if tuple(c.shape) != tuple(r.shape):
                rec["error"] = f"output {i} shape {tuple(c.shape)} != {tuple(r.shape)}"
                return rec
            if c.dtype != r.dtype:
                rec["error"] = f"output {i} dtype {c.dtype} != {r.dtype}"
                return rec
            finite_ref = bool(mx.all(mx.isfinite(r.astype(mx.float32))).item())
            finite_c = bool(mx.all(mx.isfinite(c.astype(mx.float32))).item())
            close = bool(mx.allclose(c.astype(mx.float32), r.astype(mx.float32), atol=atol, rtol=rtol).item())
            st = _stats(c, r)
            ok = ok and close and (finite_c or not finite_ref)
            outs.append({**st, "close": close, "finite": finite_c})
        rec.update({"pass": ok, "outputs": outs})
        if hidden:  # never leak hidden shapes back to the writer
            rec["outputs"] = [{"close": o["close"], "finite": o["finite"]} for o in outs]
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    return rec


def run_gate(op: OpSpec, candidate: Path, dtype: str = "float16", seed: int = 42) -> dict[str, Any]:
    source = candidate.read_text()
    problems = static_checks(source)
    base = {"tool": "kernel_correctness", "op": op.name, "dtype": dtype, "candidate": str(candidate)}
    if problems:
        return {**base, "status": "rejected", "problems": problems, "n": 0, "n_pass": 0, "n_fail": 0}
    try:
        kernel = load_candidate(candidate).kernel
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "compile_error", "error": f"{type(exc).__name__}: {exc}"}

    hidden_seed = int(os.environ.get("VISAI_HIDDEN_SEED") or 0) or 1234567
    cases = []
    i = 0
    for shape in op.visible_shapes:
        for dist in ("normal", "x3"):
            cases.append(run_case(op, kernel, shape, dtype, seed + i, dist, hidden=False))
            i += 1
    for j, shape in enumerate(op.hidden_shapes):
        for dist in DISTRIBUTIONS:
            cases.append(run_case(op, kernel, shape, dtype, hidden_seed + 97 * j + i, dist, hidden=True))
            i += 1
    n_ok = sum(1 for c in cases if c.get("pass"))
    n_err = sum(1 for c in cases if c.get("error"))
    visible_fail = [c for c in cases if not c["pass"] and not c["hidden"]]
    hidden_fail = [c for c in cases if not c["pass"] and c["hidden"]]
    return {
        **base,
        "status": "pass" if n_ok == len(cases) else "fail",
        "atol_rtol": TOL[dtype],
        "n": len(cases),
        "n_pass": n_ok,
        "n_fail": len(cases) - n_ok,
        "n_errors": n_err,
        "n_hidden_fail": len(hidden_fail),
        "visible_failures": visible_fail[:6],
        "hidden_failure_kinds": sorted({c["name"] for c in hidden_fail}),
        "first_error": next((c["error"] for c in cases if c.get("error")), None),
        "gate": "accept candidate only if status==pass",
    }


def format_table(payload: dict[str, Any]) -> str:
    lines = [
        f"kernel correctness {payload.get('op')}  {payload.get('status')}  "
        f"{payload.get('n_pass')}/{payload.get('n')}  hidden_fail={payload.get('n_hidden_fail')}"
    ]
    for p in payload.get("problems") or []:
        lines.append(f"  REJECT {p}")
    for c in payload.get("visible_failures") or []:
        lines.append(f"  NO {c['name']} {c.get('error') or c.get('outputs')}")
    return "\n".join(lines)
