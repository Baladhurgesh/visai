"""Apply / revert / status for candidate kernels (port of kernel-forge apply_generated_kernel.py).

- MLXModulePatcher: swaps mlx-lm module behavior in-process (RMSNorm, fused add+RMSNorm in the
  decoder block, SwiGLU). Used by in-process eval and by `python -m visai.serve`.
- VLLMSpanPatcher: kernel-forge's START/END span replacement with a .bak backup, for vLLM on CUDA.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Any, Callable

from visai.verify.kernel_gate import load_candidate

SUPPORTED_BLOCK_MODULES = ("qwen3", "qwen2", "llama", "mistral")


def _rmsnorm_call(kernel: Callable) -> Callable:
    def __call__(self, x):
        shape = x.shape
        y = kernel(x.reshape(-1, shape[-1]), self.weight, self.eps)
        return y.reshape(shape)

    return __call__


def _fused_block_call(kernel: Callable) -> Callable:
    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        shape = x.shape
        normed, h = kernel(
            x.reshape(-1, shape[-1]),
            r.reshape(-1, shape[-1]),
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        r = self.mlp(normed.reshape(shape))
        return h.reshape(shape) + r

    return __call__


class MLXModulePatcher:
    """patches: list of {"op": "rmsnorm"|"add_rmsnorm"|"swiglu", "candidate": path}."""

    def __init__(self, model_type: str, patches: list[dict[str, str]]):
        self.model_type = model_type
        self.patches = patches
        self._saved: list[tuple[Any, str, Any]] = []

    def _save_and_set(self, owner: Any, attr: str, value: Any) -> None:
        self._saved.append((owner, attr, getattr(owner, attr)))
        setattr(owner, attr, value)

    def apply(self) -> dict[str, Any]:
        import mlx.nn as nn

        applied = []
        mod = importlib.import_module(f"mlx_lm.models.{self.model_type}")
        for p in self.patches:
            kernel = load_candidate(Path(p["candidate"])).kernel
            op = p["op"]
            if op == "rmsnorm":
                self._save_and_set(nn.RMSNorm, "__call__", _rmsnorm_call(kernel))
            elif op == "add_rmsnorm":
                if self.model_type not in SUPPORTED_BLOCK_MODULES or not hasattr(mod, "TransformerBlock"):
                    raise RuntimeError(f"add_rmsnorm fusion not wired for model_type={self.model_type}")
                self._save_and_set(mod.TransformerBlock, "__call__", _fused_block_call(kernel))
            elif op == "swiglu":
                if not hasattr(mod, "swiglu"):
                    raise RuntimeError(f"{self.model_type} has no swiglu symbol")
                self._save_and_set(mod, "swiglu", kernel)
            else:
                raise RuntimeError(f"no model integration for op {op}")
            applied.append(op)
        return {"status": "applied", "ops": applied, "model_type": self.model_type}

    def revert(self) -> dict[str, Any]:
        while self._saved:
            owner, attr, value = self._saved.pop()
            setattr(owner, attr, value)
        return {"status": "reverted"}

    def status(self) -> dict[str, Any]:
        return {"status": "patched" if self._saved else "upstream", "n": len(self._saved)}


class VLLMSpanPatcher:
    """kernel-forge span patch: replace [start, end) in an installed vLLM file, keep a .bak."""

    def __init__(self, target: Path, generated: Path, start: str, end: str, generated_start: str | None = None):
        self.target = target
        self.backup = target.with_suffix(target.suffix + ".bak")
        self.generated = generated
        self.start = start
        self.end = end
        self.generated_start = generated_start or start

    def _generated_functions(self) -> str:
        text = self.generated.read_text()
        i = text.find(self.generated_start)
        if i < 0:
            raise RuntimeError(f"no '{self.generated_start[:40]}...' in {self.generated}")
        return text[i:].rstrip() + "\n\n\n"

    def apply(self) -> dict[str, Any]:
        if not self.backup.exists():
            shutil.copy2(self.target, self.backup)
        src = self.target.read_text()
        i0, i1 = src.find(self.start), src.find(self.end)
        if i0 < 0 or i1 < 0 or i1 <= i0:
            raise RuntimeError(f"could not find span in {self.target}")
        self.target.write_text(src[:i0] + self._generated_functions() + src[i1:])
        return {"status": "applied", "target": str(self.target), "backup": str(self.backup)}

    def revert(self) -> dict[str, Any]:
        if not self.backup.exists():
            raise RuntimeError(f"no backup at {self.backup}")
        shutil.copy2(self.backup, self.target)
        return {"status": "reverted", "target": str(self.target)}

    def status(self) -> dict[str, Any]:
        patched = self.backup.exists() and self.target.read_text() != self.backup.read_text()
        return {"status": "patched" if patched else "upstream"}
