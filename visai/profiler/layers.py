"""Layer profiler: exclusive time per layer class during a real forward pass + input capture.

Every nn.Module subclass in the model gets a synchronizing timing wrapper. Exclusive time excludes
time spent in child modules, so "ConformerConvolution 18%" means 18% spent in that layer's own ops.
For the top layers we also capture real input activations (up to 3 distinct shape signatures per class)
so Gate 1 can compare a candidate kernel against the model's own layer on real data.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SCALAR = (int, float, bool, type(None))


def _flat(out) -> list:
    if isinstance(out, mx.array):
        return [out]
    if isinstance(out, (tuple, list)):
        return [a for o in out for a in _flat(o)]
    return []


def class_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _capturable(args: tuple, kwargs: dict) -> bool:
    return all(isinstance(a, mx.array) or isinstance(a, SCALAR) for a in args) and all(
        isinstance(v, mx.array) or isinstance(v, SCALAR) for v in kwargs.values()
    )


def _sig(args: tuple, kwargs: dict) -> str:
    parts = [f"{tuple(a.shape)}:{a.dtype}" if isinstance(a, mx.array) else repr(a) for a in args]
    parts += [f"{k}={tuple(v.shape)}" if isinstance(v, mx.array) else f"{k}={v!r}" for k, v in sorted(kwargs.items())]
    return "|".join(parts)


def save_capture(path: Path, args: tuple, kwargs: dict, meta: dict) -> None:
    arrays, spec = {}, {"args": [], "kwargs": {}, **meta}
    for i, a in enumerate(args):
        if isinstance(a, mx.array):
            arrays[f"a{i}"] = np.array(a.astype(mx.float32)) if a.dtype == mx.bfloat16 else np.array(a)
            spec["args"].append({"array": f"a{i}", "dtype": str(a.dtype)})
        else:
            spec["args"].append({"value": a})
    for k, v in kwargs.items():
        if isinstance(v, mx.array):
            arrays[f"k_{k}"] = np.array(v.astype(mx.float32)) if v.dtype == mx.bfloat16 else np.array(v)
            spec["kwargs"][k] = {"array": f"k_{k}", "dtype": str(v.dtype)}
        else:
            spec["kwargs"][k] = {"value": v}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), **arrays)
    path.with_suffix(".json").write_text(json.dumps(spec, indent=1, default=str))


_DT = {"mlx.core.bfloat16": mx.bfloat16, "mlx.core.float16": mx.float16, "mlx.core.float32": mx.float32}


def load_capture(path: Path) -> tuple[list, dict, dict]:
    spec = json.loads(Path(path).with_suffix(".json").read_text())
    data = np.load(Path(path).with_suffix(".npz"))

    def conv(s):
        if "array" in s:
            a = mx.array(data[s["array"]])
            dt = _DT.get(s["dtype"])
            return a.astype(dt) if dt is not None else a
        return s["value"]

    return [conv(s) for s in spec["args"]], {k: conv(s) for k, s in spec["kwargs"].items()}, spec


def profile_layers(root: nn.Module, run, capture_dir: Path | None = None, top_k_capture: int = 6) -> dict[str, Any]:
    """Run `run()` with every module class instrumented. Returns per-class exclusive time table."""
    paths = {id(m): p for p, m in root.named_modules()}
    classes = {type(m) for _, m in root.named_modules() if type(m) is not type(root)}
    excl: dict[type, float] = defaultdict(float)
    calls: dict[type, int] = defaultdict(int)
    samples: dict[type, dict[str, tuple]] = defaultdict(dict)
    stack: list[float] = []
    saved = []

    def make(cls):
        orig = cls.__call__

        def wrapper(self, *args, **kwargs):
            mx.eval(*[a for a in args if isinstance(a, mx.array)])
            stack.append(0.0)
            t0 = time.perf_counter()
            out = orig(self, *args, **kwargs)
            mx.eval(*_flat(out))
            dt = time.perf_counter() - t0
            child = stack.pop()
            excl[cls] += dt - child
            calls[cls] += 1
            if stack:
                stack[-1] += dt
            sig = _sig(args, kwargs) if _capturable(args, kwargs) else None
            if sig and sig not in samples[cls]:
                # keep the first 3 distinct shapes plus the most recent one (e.g. decode after prefill)
                if len(samples[cls]) >= 4:
                    samples[cls].pop(list(samples[cls])[-1])
                samples[cls][sig] = (paths.get(id(self), "?"), args, dict(kwargs))
            return out

        return orig, wrapper

    for cls in classes:
        orig, w = make(cls)
        saved.append((cls, orig))
        cls.__call__ = w
    t_start = time.perf_counter()
    try:
        run()
    finally:
        for cls, orig in saved:
            cls.__call__ = orig
    wall = time.perf_counter() - t_start
    attributed = sum(excl.values())
    rows = []
    for cls, t in sorted(excl.items(), key=lambda kv: -kv[1]):
        rows.append({
            "class": class_path(cls),
            "name": cls.__name__,
            "exclusive_ms": t * 1e3,
            "pct": 100 * t / max(attributed, 1e-9),
            "calls": calls[cls],
            "us_per_call": t / max(calls[cls], 1) * 1e6,
            "capturable": bool(samples.get(cls)),
            "example_path": next(iter(samples[cls].values()))[0] if samples.get(cls) else None,
            "n_params": sum(v.size for _, v in _params(root, paths, cls)),
        })
    captures: dict[str, list[str]] = {}
    if capture_dir:
        for r in [r for r in rows if r["capturable"]][:top_k_capture]:
            cls = next(c for c in excl if class_path(c) == r["class"])
            files = []
            for i, (sig, (mpath, args, kwargs)) in enumerate(samples[cls].items()):
                f = capture_dir / f"{cls.__name__}_{i}"
                save_capture(f, args, kwargs, {"module_path": mpath, "class": r["class"], "signature": sig})
                files.append(str(f))
            captures[r["class"]] = files
    return {"wall_s": wall, "attributed_s": attributed, "layers": rows, "captures": captures}


def _params(root, paths, cls):
    from mlx.utils import tree_flatten

    for p, m in root.named_modules():
        if type(m) is cls:
            yield from tree_flatten(m.parameters())
            break


def format_table(prof: dict[str, Any], n: int = 15) -> str:
    lines = [f"{'layer':<34} {'self%':>6} {'ms':>9} {'calls':>7} {'us/call':>9}  capture", "-" * 80]
    for r in prof["layers"][:n]:
        lines.append(f"{r['name'][:34]:<34} {r['pct']:6.1f} {r['exclusive_ms']:9.2f} {r['calls']:7d} {r['us_per_call']:9.1f}  "
                     f"{'yes' if r['capturable'] else 'no'}")
    return "\n".join(lines)
