"""KernelBench-style op suite for MLX.

Each OpSpec carries:
- reference: the trusted, naive implementation (computed in float32 for checking)
- runtime:   what mlx-lm actually runs today (the realistic baseline)
- visible shapes the writer is told about, and hidden shapes it never sees
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
TOL = {"float16": (1e-2, 1e-2), "bfloat16": (2e-2, 2e-2), "float32": (1e-4, 1e-4)}

# Input distributions (KernelBench-Verified style): scaled and negated variants catch shortcuts.
DISTRIBUTIONS = {"normal": 1.0, "x3": 3.0, "small": 0.01, "negated": -1.0}


@dataclass
class OpSpec:
    name: str
    family: str
    signature: str
    description: str
    reference: Callable[..., Any]
    runtime: Callable[..., Any]
    make_inputs: Callable[[dict, Any, int, float], tuple[list, dict]]
    visible_shapes: list[dict]
    hidden_shapes: list[dict]
    bench_shapes: list[dict]
    n_outputs: int = 1
    notes: list[str] = field(default_factory=list)

    def reference_source(self) -> str:
        return inspect.getsource(self.reference)

    def runtime_source(self) -> str:
        return inspect.getsource(self.runtime)


def _rand(key: int, shape: tuple, dtype, scale: float) -> mx.array:
    return (mx.random.normal(shape, key=mx.random.key(key)) * scale).astype(dtype)


# ------------------------------------------------------------------ rmsnorm


def rmsnorm_reference(x, w, eps: float = 1e-6):
    xf = x.astype(mx.float32)
    y = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (y * w.astype(mx.float32)).astype(x.dtype)


def rmsnorm_runtime(x, w, eps: float = 1e-6):
    return mx.fast.rms_norm(x, w, eps)


def rmsnorm_inputs(shape, dtype, seed, scale):
    rows, d = shape["rows"], shape["d"]
    x = _rand(seed, (rows, d), dtype, scale)
    w = (1.0 + 0.1 * mx.random.normal((d,), key=mx.random.key(seed + 1))).astype(dtype)
    return [x, w], {"eps": 1e-6}


# ------------------------------------------------------------------ add + rmsnorm (fusion)


def add_rmsnorm_reference(x, residual, w, eps: float = 1e-6):
    h = (x.astype(mx.float32) + residual.astype(mx.float32)).astype(x.dtype)
    hf = h.astype(mx.float32)
    y = hf * mx.rsqrt(mx.mean(hf * hf, axis=-1, keepdims=True) + eps)
    return (y * w.astype(mx.float32)).astype(x.dtype), h


def add_rmsnorm_runtime(x, residual, w, eps: float = 1e-6):
    h = x + residual
    return mx.fast.rms_norm(h, w, eps), h


def add_rmsnorm_inputs(shape, dtype, seed, scale):
    rows, d = shape["rows"], shape["d"]
    x = _rand(seed, (rows, d), dtype, scale)
    r = _rand(seed + 7, (rows, d), dtype, scale)
    w = (1.0 + 0.1 * mx.random.normal((d,), key=mx.random.key(seed + 1))).astype(dtype)
    return [x, r, w], {"eps": 1e-6}


# ------------------------------------------------------------------ layernorm


def layernorm_reference(x, w, b, eps: float = 1e-5):
    xf = x.astype(mx.float32)
    mu = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mu) ** 2, axis=-1, keepdims=True)
    y = (xf - mu) * mx.rsqrt(var + eps)
    return (y * w.astype(mx.float32) + b.astype(mx.float32)).astype(x.dtype)


def layernorm_runtime(x, w, b, eps: float = 1e-5):
    return mx.fast.layer_norm(x, w, b, eps)


def layernorm_inputs(shape, dtype, seed, scale):
    rows, d = shape["rows"], shape["d"]
    x = _rand(seed, (rows, d), dtype, scale) + 0.5 * scale
    w = (1.0 + 0.1 * mx.random.normal((d,), key=mx.random.key(seed + 1))).astype(dtype)
    b = (0.1 * mx.random.normal((d,), key=mx.random.key(seed + 2))).astype(dtype)
    return [x, w, b], {"eps": 1e-5}


# ------------------------------------------------------------------ softmax


def softmax_reference(x):
    xf = x.astype(mx.float32)
    m = mx.max(xf, axis=-1, keepdims=True)
    e = mx.exp(xf - m)
    return (e / mx.sum(e, axis=-1, keepdims=True)).astype(x.dtype)


def softmax_runtime(x):
    return mx.softmax(x, axis=-1, precise=True)


def softmax_inputs(shape, dtype, seed, scale):
    return [_rand(seed, (shape["rows"], shape["d"]), dtype, scale * 4)], {}


# ------------------------------------------------------------------ swiglu


def swiglu_reference(gate, up):
    g = gate.astype(mx.float32)
    return (g * mx.sigmoid(g) * up.astype(mx.float32)).astype(gate.dtype)


def swiglu_runtime(gate, up):
    from mlx_lm.models.activations import swiglu

    return swiglu(gate, up)


def swiglu_inputs(shape, dtype, seed, scale):
    rows, d = shape["rows"], shape["d"]
    return [_rand(seed, (rows, d), dtype, scale * 2), _rand(seed + 3, (rows, d), dtype, scale)], {}


# ------------------------------------------------------------------ gelu (tanh approx)


def gelu_reference(x):
    xf = x.astype(mx.float32)
    c = math.sqrt(2.0 / math.pi)
    return (0.5 * xf * (1.0 + mx.tanh(c * (xf + 0.044715 * xf**3)))).astype(x.dtype)


def gelu_runtime(x):
    return nn.gelu_approx(x)


def gelu_inputs(shape, dtype, seed, scale):
    return [_rand(seed, (shape["rows"], shape["d"]), dtype, scale * 2)], {}


# ------------------------------------------------------------------ matmul + bias + relu


def matmul_bias_relu_reference(x, w, b):
    y = x.astype(mx.float32) @ w.astype(mx.float32) + b.astype(mx.float32)
    return mx.maximum(y, 0.0).astype(x.dtype)


def matmul_bias_relu_runtime(x, w, b):
    return mx.maximum(mx.addmm(b, x, w), 0)


def matmul_bias_relu_inputs(shape, dtype, seed, scale):
    m, k, n = shape["m"], shape["k"], shape["n"]
    x = _rand(seed, (m, k), dtype, scale)
    w = _rand(seed + 5, (k, n), dtype, scale / math.sqrt(k))
    b = _rand(seed + 6, (n,), dtype, 0.1 * scale)
    return [x, w, b], {}


# ------------------------------------------------------------------ registry

_NORM_VISIBLE = [{"rows": 1, "d": 1024}, {"rows": 8, "d": 1024}, {"rows": 128, "d": 1024}]
_NORM_HIDDEN = [
    {"rows": 1, "d": 4096},
    {"rows": 3, "d": 2048},
    {"rows": 7, "d": 1000},
    {"rows": 33, "d": 1536},
    {"rows": 256, "d": 4096},
    {"rows": 5, "d": 4099},
]
_ELEM_VISIBLE = [{"rows": 1, "d": 3072}, {"rows": 16, "d": 3072}, {"rows": 128, "d": 3072}]
_ELEM_HIDDEN = [
    {"rows": 1, "d": 11008},
    {"rows": 9, "d": 1000},
    {"rows": 64, "d": 8192},
    {"rows": 3, "d": 4097},
]

OPS: dict[str, OpSpec] = {
    "rmsnorm": OpSpec(
        name="rmsnorm",
        family="rmsnorm",
        signature="kernel(x: mx.array[rows, d], w: mx.array[d], eps: float = 1e-6) -> mx.array[rows, d]",
        description="RMSNorm over the last axis. Accumulate in float32, return x.dtype.",
        reference=rmsnorm_reference,
        runtime=rmsnorm_runtime,
        make_inputs=rmsnorm_inputs,
        visible_shapes=_NORM_VISIBLE,
        hidden_shapes=_NORM_HIDDEN,
        bench_shapes=[{"rows": 1, "d": 1024}, {"rows": 1024, "d": 4096}],
        notes=["Runtime baseline is mx.fast.rms_norm, already a single fused kernel; hard to beat."],
    ),
    "add_rmsnorm": OpSpec(
        name="add_rmsnorm",
        family="rmsnorm",
        signature=(
            "kernel(x: mx.array[rows, d], residual: mx.array[rows, d], w: mx.array[d], eps: float = 1e-6)"
            " -> tuple[normed: mx.array[rows, d], h: mx.array[rows, d]]  where h = x + residual"
        ),
        description="Residual add followed by RMSNorm (Qwen3/Llama decoder block). Return (rmsnorm(h), h).",
        reference=add_rmsnorm_reference,
        runtime=add_rmsnorm_runtime,
        make_inputs=add_rmsnorm_inputs,
        visible_shapes=_NORM_VISIBLE,
        hidden_shapes=_NORM_HIDDEN,
        bench_shapes=[{"rows": 1, "d": 1024}, {"rows": 1024, "d": 4096}],
        n_outputs=2,
        notes=["mlx-lm launches add and rms_norm separately: fusing saves a full read of h and a launch."],
    ),
    "layernorm": OpSpec(
        name="layernorm",
        family="rmsnorm",
        signature="kernel(x: mx.array[rows, d], w: mx.array[d], b: mx.array[d], eps: float = 1e-5) -> mx.array",
        description="LayerNorm over the last axis with affine weight and bias.",
        reference=layernorm_reference,
        runtime=layernorm_runtime,
        make_inputs=layernorm_inputs,
        visible_shapes=_NORM_VISIBLE,
        hidden_shapes=_NORM_HIDDEN,
        bench_shapes=[{"rows": 8, "d": 1024}, {"rows": 1024, "d": 4096}],
    ),
    "softmax": OpSpec(
        name="softmax",
        family="softmax",
        signature="kernel(x: mx.array[rows, d]) -> mx.array[rows, d]",
        description="Numerically stable softmax over the last axis.",
        reference=softmax_reference,
        runtime=softmax_runtime,
        make_inputs=softmax_inputs,
        visible_shapes=_NORM_VISIBLE,
        hidden_shapes=_NORM_HIDDEN,
        bench_shapes=[{"rows": 32, "d": 1024}, {"rows": 1024, "d": 4096}],
    ),
    "swiglu": OpSpec(
        name="swiglu",
        family="swiglu",
        signature="kernel(gate: mx.array[rows, d], up: mx.array[rows, d]) -> mx.array[rows, d]",
        description="silu(gate) * up.",
        reference=swiglu_reference,
        runtime=swiglu_runtime,
        make_inputs=swiglu_inputs,
        visible_shapes=_ELEM_VISIBLE,
        hidden_shapes=_ELEM_HIDDEN,
        bench_shapes=[{"rows": 1, "d": 3072}, {"rows": 512, "d": 8192}],
        notes=["mlx-lm already mx.compile's swiglu into one elementwise kernel."],
    ),
    "gelu": OpSpec(
        name="gelu",
        family="swiglu",
        signature="kernel(x: mx.array[rows, d]) -> mx.array[rows, d]",
        description="GELU, tanh approximation.",
        reference=gelu_reference,
        runtime=gelu_runtime,
        make_inputs=gelu_inputs,
        visible_shapes=_ELEM_VISIBLE,
        hidden_shapes=_ELEM_HIDDEN,
        bench_shapes=[{"rows": 16, "d": 3072}, {"rows": 512, "d": 8192}],
    ),
    "matmul_bias_relu": OpSpec(
        name="matmul_bias_relu",
        family="gemm",
        signature="kernel(x: mx.array[m, k], w: mx.array[k, n], b: mx.array[n]) -> mx.array[m, n]",
        description="relu(x @ w + b).",
        reference=matmul_bias_relu_reference,
        runtime=matmul_bias_relu_runtime,
        make_inputs=matmul_bias_relu_inputs,
        visible_shapes=[{"m": 1, "k": 1024, "n": 1024}, {"m": 64, "k": 1024, "n": 1024}],
        hidden_shapes=[{"m": 3, "k": 1000, "n": 520}, {"m": 128, "k": 2048, "n": 1024}],
        bench_shapes=[{"m": 16, "k": 1024, "n": 1024}, {"m": 512, "k": 2048, "n": 2048}],
        notes=["GEMM: vendor matmul is usually optimal. Only the epilogue fusion is interesting."],
    ),
}


def get_op(name: str) -> OpSpec:
    if name not in OPS:
        raise KeyError(f"unknown op {name}; have {sorted(OPS)}")
    return OPS[name]
