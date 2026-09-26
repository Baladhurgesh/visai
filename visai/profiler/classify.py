"""Map kernel / op / module names to families (port of kernel-forge lib/classify.py + MLX names).

First regex match wins. Priority is what the agent should do, not runtime share.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from visai.schemas import Priority


@dataclass(frozen=True)
class FamilyRule:
    name: str
    pattern: re.Pattern[str]
    priority: Priority
    reason: str
    experiments: tuple[str, ...]
    visai_ops: tuple[str, ...] = ()


RULES: tuple[FamilyRule, ...] = (
    FamilyRule(
        name="gated_deltanet",
        pattern=re.compile(
            r"delta|gdn|gated_delta|chunk_gated|recurrent|causal_conv|conv1d|mamba|ssm|linear_attn|linear_attention",
            re.I,
        ),
        priority="rewrite",
        reason="Hybrid linear-attention layers; newer than FlashAttention; best custom-kernel bet.",
        experiments=(
            "fuse causal_conv1d with the GDN state update",
            "fewer CTAs / threadgroups per head (loop V tiles inside one program)",
            "vectorize state loads/stores",
        ),
    ),
    FamilyRule(
        name="full_attention",
        pattern=re.compile(
            r"flash|fmha|attention|attn|sdpa|scaled_dot_product|reshape_and_cache|paged_attn|flashinfer",
            re.I,
        ),
        priority="inspect",
        reason="Flash / SDPA paths are usually already optimal.",
        experiments=("only chase if still #1 after cheaper targets",),
    ),
    FamilyRule(
        name="gemm",
        pattern=re.compile(
            r"gemm|gemv|cutlass|cublas|xmma|wgmma|nvjet|matmul|addmm|qmm|qmv|quantized_matmul|"
            r"quantizedlinear|linear|steel",
            re.I,
        ),
        priority="skip",
        reason="Vendor GEMM/GEMV is already optimized. Change its precision (quantization) instead of rewriting it.",
        experiments=("try lower-bit weight quantization via the deployment search",),
        visai_ops=("matmul_bias_relu",),
    ),
    FamilyRule(
        name="rmsnorm",
        pattern=re.compile(r"rms.?norm|fused_add_rms|layer_?norm|layernorm", re.I),
        priority="fuse",
        reason="Small memory-bound launch; fuse with the residual add.",
        experiments=("fuse residual add + RMSNorm", "SIMD-group reduction", "vectorized loads"),
        visai_ops=("add_rmsnorm", "rmsnorm", "layernorm"),
    ),
    FamilyRule(
        name="swiglu",
        pattern=re.compile(r"silu|swiglu|act_and_mul|gelu|gated_silu|activation", re.I),
        priority="fuse",
        reason="Elementwise; fuse SiLU*mul (and ideally into the down-proj input).",
        experiments=("fuse silu and mul into one kernel", "vectorize loads"),
        visai_ops=("swiglu", "gelu"),
    ),
    FamilyRule(
        name="rope",
        pattern=re.compile(r"rotary|rope|mrope", re.I),
        priority="fuse",
        reason="Small kernel; fuse with Q/K prep.",
        experiments=("fuse RoPE into the q/k norm epilogue",),
    ),
    FamilyRule(
        name="softmax",
        pattern=re.compile(r"softmax", re.I),
        priority="fuse",
        reason="Row reduction; usually fused already.",
        experiments=("single-pass online softmax",),
        visai_ops=("softmax",),
    ),
    FamilyRule(
        name="quant",
        pattern=re.compile(r"quant|dequant|fp8|nvfp4|fp4|int8|scaled_mm", re.I),
        priority="inspect",
        reason="Only relevant if this checkpoint is quantized.",
        experiments=("fuse dequant into the consumer kernel",),
    ),
    FamilyRule(
        name="embedding",
        pattern=re.compile(r"embed|gather|take", re.I),
        priority="skip",
        reason="Gather-bound; rarely worth a custom kernel.",
        experiments=(),
    ),
    FamilyRule(
        name="memcpy_elementwise",
        pattern=re.compile(r"memcpy|memset|copy|elementwise|vectorized|add_kernel|mul_kernel|residual|binary", re.I),
        priority="fuse",
        reason="Launch overhead / fusion target on unified memory.",
        experiments=("fuse adjacent elementwise ops", "eliminate unnecessary copies"),
        visai_ops=("add_rmsnorm",),
    ),
)

OTHER = FamilyRule(
    name="other",
    pattern=re.compile(r"^$"),
    priority="inspect",
    reason="Unclassified. Inspect before writing a kernel.",
    experiments=(),
)

_BY_NAME = {r.name: r for r in RULES} | {OTHER.name: OTHER}


def classify(kernel_name: str) -> FamilyRule:
    for rule in RULES:
        if rule.pattern.search(kernel_name):
            return rule
    return OTHER


def rule_for(family: str) -> FamilyRule:
    return _BY_NAME.get(family, OTHER)
