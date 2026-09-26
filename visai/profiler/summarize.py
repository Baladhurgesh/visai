"""Turn per-kernel / per-op timings into FamilyStats and ranked AgentTargets (port of kernel-forge summarize.py)."""

from __future__ import annotations

import re
from collections import defaultdict

from visai.profiler.classify import classify, rule_for
from visai.schemas import AgentTarget, FamilyStat, to_dict


def summarize_rows(rows: list[dict], top_kernels: int = 25, min_pct: float = 1.0) -> dict:
    """rows: [{"name": str, "time_ns": float, "launches": int}] -> families, kernels, targets."""
    fam_time: dict[str, float] = defaultdict(float)
    fam_inst: dict[str, int] = defaultdict(int)
    fam_example: dict[str, str] = {}
    kernels: list[dict] = []
    total = 0.0
    for row in rows:
        t = float(row["time_ns"])
        name = row["name"]
        inst = int(row.get("launches") or 0)
        rule = classify(name)
        fam_time[rule.name] += t
        fam_inst[rule.name] += inst
        fam_example.setdefault(rule.name, name)
        kernels.append({"time_ns": t, "family": rule.name, "launches": inst, "name": name})
        total += t
    for k in kernels:
        k["gpu_pct"] = 100.0 * k["time_ns"] / total if total else 0.0
    kernels.sort(key=lambda x: -x["time_ns"])
    families: list[FamilyStat] = []
    for name, t in sorted(fam_time.items(), key=lambda x: -x[1]):
        rule = rule_for(name)
        families.append(
            FamilyStat(
                name=name,
                gpu_pct=round(100.0 * t / total if total else 0.0, 2),
                time_ns=t,
                launches=fam_inst[name],
                priority=rule.priority,
                triton_candidate=rule.priority in ("rewrite", "fuse"),
                reason=rule.reason,
                example_kernel=_short(fam_example[name]),
            )
        )
    targets = pick_targets(families, kernels, min_pct=min_pct)
    return {
        "total_time_ns": total,
        "families": [to_dict(f) for f in families],
        "kernels": kernels[:top_kernels],
        "targets": [to_dict(t) for t in targets],
    }


def pick_targets(families: list[FamilyStat], kernels: list[dict], limit: int = 3, min_pct: float = 1.0) -> list[AgentTarget]:
    """Prefer rewrite/fuse families. Never lead with GEMM. Skip tiny families (Amdahl)."""
    ranked = [f for f in families if f.priority != "skip"]
    ranked.sort(key=lambda f: (0 if f.priority == "rewrite" else 1 if f.priority == "fuse" else 2, -f.gpu_pct))
    targets: list[AgentTarget] = []
    for fam in ranked:
        if len(targets) >= limit:
            break
        if fam.gpu_pct < min_pct and fam.priority != "rewrite":
            continue
        rule = rule_for(fam.name)
        top_k = next((k for k in kernels if k["family"] == fam.name), None)
        per_launch_us = (top_k["time_ns"] / max(top_k["launches"], 1) / 1000.0) if top_k else 0.0
        amdahl_2x = fam.gpu_pct / 2.0
        targets.append(
            AgentTarget(
                family=fam.name,
                kernel=_short(top_k["name"] if top_k else fam.example_kernel),
                gpu_pct=fam.gpu_pct,
                launches=fam.launches,
                runtime_note=(
                    f"~{per_launch_us:.1f} us/launch, {fam.gpu_pct:.1f}% of time, {fam.launches} launches; "
                    f"2x on this family ~= +{amdahl_2x:.1f}% e2e (Amdahl)"
                ),
                suggested_experiments=list(rule.experiments) + [f"visai op: {o}" for o in rule.visai_ops],
            )
        )
    return targets


def _short(name: str, n: int = 120) -> str:
    return re.sub(r"\s+", " ", name)[:n]


def format_table(summary: dict) -> str:
    lines = ["family                 time%  launches  priority  kernel?", "-" * 64]
    for f in summary["families"]:
        cand = "yes" if f["triton_candidate"] else "no"
        lines.append(f"{f['name']:<22} {f['gpu_pct']:5.1f}% {f['launches']:9d}  {f['priority']:<8}  {cand}")
    return "\n".join(lines)
