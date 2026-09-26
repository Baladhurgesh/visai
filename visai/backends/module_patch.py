"""Layer-kernel patches: replace a layer class's __call__ with a candidate `forward(module, ...)`.

Patch spec kinds:
  {"kind": "module", "class": "pkg.mod:Qual.Name", "candidate": "path.py"}  -> all instances of that class
  {"kind": "op", "op": "add_rmsnorm", "candidate": "path.py"}              -> MLXModulePatcher fusion (LLMs)
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


def resolve_class(path: str) -> type:
    mod, qual = path.split(":")
    obj: Any = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def load_forward(path: str):
    spec = importlib.util.spec_from_file_location(f"visai_layer_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "forward"):
        raise RuntimeError(f"{path} has no forward(module, ...)")
    return mod.forward


class PatchSet:
    def __init__(self) -> None:
        self._saved: list[tuple[Any, str, Any]] = []
        self._others: list[Any] = []

    def set(self, owner: Any, attr: str, value: Any) -> None:
        self._saved.append((owner, attr, getattr(owner, attr)))
        setattr(owner, attr, value)

    def revert(self) -> None:
        while self._saved:
            owner, attr, value = self._saved.pop()
            setattr(owner, attr, value)
        for o in self._others:
            o.revert()
        self._others.clear()


def apply_patches(handle, adapter, patches: list[dict[str, Any]]) -> PatchSet:
    ps = PatchSet()
    for p in patches:
        if p.get("kind", "module") == "module":
            from visai.memory.kernels import resolve_patch_path

            cls = resolve_class(p["class"])
            fwd = load_forward(resolve_patch_path(p))

            def __call__(self, *args, _fwd=fwd, **kwargs):
                return _fwd(self, *args, **kwargs)

            ps.set(cls, "__call__", __call__)
        elif p["kind"] == "op":
            from visai.backends.patchers import MLXModulePatcher
            from visai.models.mlx_model import model_type

            mp = MLXModulePatcher(model_type(handle.model), [{"op": p["op"], "candidate": p["candidate"]}])
            mp.apply()
            ps._others.append(mp)
    return ps
