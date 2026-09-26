"""Build a ProfileBrief for an MLX model or fall back to an architecture prior (kernel-forge prior_brief)."""

from __future__ import annotations

from typing import Any

from visai.profiler.hardware import probe
from visai.profiler.summarize import format_table, pick_targets, summarize_rows
from visai.schemas import FamilyStat, ProfileBrief, to_dict


def mlx_profile_brief(model, tokenizer, model_id: str, *, baseline: dict | None = None) -> dict[str, Any]:
    from visai.models.mlx_model import profile_families

    rows = profile_families(model, tokenizer)
    summary = summarize_rows(rows)
    brief = ProfileBrief(
        hardware=probe(),
        workload={"model": model_id, "batch": 1, "phase": "decode", "temperature": 0.0},
        baseline=baseline or {},
        families=summary["families"],
        targets=summary["targets"],
        artifacts={"source": "mlx per-op synchronized timing", "kernels": summary["kernels"]},
    )
    out = brief.to_json_obj()
    out["table"] = format_table(summary)
    return out


def prior_brief(model_id: str, note: str = "") -> dict[str, Any]:
    """Architecture prior for a dense decoder when profiling is unavailable."""
    fams = [
        FamilyStat("gemm", 70.0, 0.0, 0, "skip", False, "Vendor GEMV/GEMM; change precision instead.", "qmv"),
        FamilyStat("full_attention", 8.0, 0.0, 0, "inspect", False, "SDPA already fused.", "sdpa_vector"),
        FamilyStat("rmsnorm", 6.0, 0.0, 0, "fuse", True, "Fuse with the residual add.", "rms_norm"),
        FamilyStat("memcpy_elementwise", 6.0, 0.0, 0, "fuse", True, "Residual adds.", "binary_add"),
        FamilyStat("swiglu", 4.0, 0.0, 0, "fuse", True, "Compiled already.", "swiglu"),
        FamilyStat("rope", 3.0, 0.0, 0, "fuse", True, "Small.", "rope"),
    ]
    targets = pick_targets(fams, [])
    brief = ProfileBrief(
        status="prior",
        error=note or "profile unavailable; used dense-decoder architecture prior",
        hardware=probe(),
        workload={"model": model_id, "batch": 1},
        families=[to_dict(f) for f in fams],
        targets=[to_dict(t) for t in targets],
        artifacts={"source": "architecture prior"},
    )
    return brief.to_json_obj()
