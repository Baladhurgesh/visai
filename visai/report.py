"""Rich terminal reports (kernel-forge's final trial table + Visai optimization report)."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(stderr=True)


def trials_table(summary: dict[str, Any], title: str | None = None) -> Table:
    t = Table(title=title or f"{summary.get('target')}  ({summary.get('run_id')})")
    for col in ("slot", "label", "class", "gate1", "speedup", "confirmed", "outcome", "keep"):
        t.add_column(col)
    for r in summary.get("trials") or []:
        sp = r.get("speedup")
        t.add_row(
            str(r.get("slot")), str(r.get("label")), str(r.get("cls")), str(r.get("correctness")),
            f"{sp:.3f}x" if sp else "-", str(r.get("confirmed") or "-"), f"{r.get('outcome')} ({r.get('miss_cls')})",
            "KEEP" if r.get("keep") else "revert",
        )
    return t


def print_kernel_summary(summary: dict[str, Any]) -> None:
    console.print(trials_table(summary))
    w = summary.get("winner")
    stock = summary.get("stock") or {}
    lines = [
        f"Stock ({stock.get('label')}): {stock.get('latency_us', 0):.2f} us  "
        f"[reference {stock.get('reference_us', 0):.2f} us, runtime {stock.get('runtime_us', 0):.2f} us]",
    ]
    if w:
        lines.append(
            f"WINNER {w.get('label')}: {w.get('latency_us', 0):.2f} us  speedup {w.get('speedup', 0):.3f}x "
            f"(vs runtime {w.get('speedup_vs_runtime', 0):.3f}x, vs reference {w.get('speedup_vs_reference', 0):.3f}x)"
        )
    else:
        lines.append("No confirmed genuine win this session.")
    lines.append(
        f"Search: {summary.get('n_trials')} experiments, {len(summary.get('skills_retrieved') or [])} prior skills "
        f"retrieved, {summary.get('blocked_ideas', 0)} known-bad ideas blocked"
    )
    console.print(Panel("\n".join(lines), title="Visai kernel report"))


def print_pipeline_report(rep: dict[str, Any]) -> None:
    c = rep.get("comparison") or {}
    u, o = c.get("unoptimized") or {}, c.get("optimized") or {}
    q = (c.get("quality_metric") or "").lower()
    lines = [f"{rep.get('model')} on {rep.get('hardware')}   (quality budget: {rep.get('quality_budget')})", ""]
    lines.append("Layer profile (self time): " + ", ".join(f"{n} {p}%" for n, p, _ in rep.get("profile_top", [])[:6]))
    for lr in rep.get("layers") or []:
        b = lr.get("best")
        lines.append(f"  layer {lr.get('layer')}: " + (f"best {b['label']} {b['robust_speedup']:.3f}x" if b else "no improvement")
                     + f"  [{lr.get('iterations')} iters, stop: {lr.get('stop_reason')}]")
    for it in rep.get("integration") or []:
        lines.append(f"  integrate {it['layer']}: {'KEPT' if it['kept'] else 'reverted'}"
                     + (f" paired {it.get('paired_speedup_vs_current', 0):.3f}x vs current ({it.get('paired_wins')} rounds won)"
                        if it.get("paired_speedup_vs_current") else "") + f"  ({it.get('why')})")
    ml = rep.get("model_level") or {}
    lines.append(f"  model level: {ml.get('best_label')} {json.dumps(ml.get('best_config'))}  "
                 f"[{ml.get('candidates')} candidates, stop: {ml.get('stop_reason')}]")
    lines += ["", f"UNOPTIMIZED  {u.get('throughput', 0):.2f} {c.get('unit')}  latency {u.get('latency_s', 0):.4f}s  "
              f"{c.get('quality_metric')} {u.get(q)}  mem {u.get('peak_memory_gb', 0):.2f} GB",
              f"OPTIMIZED    {o.get('throughput', 0):.2f} {c.get('unit')}  latency {o.get('latency_s', 0):.4f}s  "
              f"{c.get('quality_metric')} {o.get(q)}  mem {o.get('peak_memory_gb', 0):.2f} GB",
              f"speedup {c.get('speedup', 0):.3f}x   latency -{c.get('latency_reduction_pct', 0):.1f}%   "
              f"quality delta {c.get('quality_delta', 0):+.5f}   ({rep.get('elapsed_min')} min)"]
    console.print(Panel("\n".join(lines), title="Visai pipeline: unoptimized vs optimized"))


def print_optimize_report(rep: dict[str, Any]) -> None:
    b, o = rep.get("baseline") or {}, rep.get("optimized") or {}
    lines = [
        f"{rep.get('model')} -> {rep.get('hardware')}",
        "",
        f"Baseline   {b.get('decode_tok_s', 0):.1f} tok/s   quality {b.get('quality_label')}",
        f"Optimized  {o.get('decode_tok_s', 0):.1f} tok/s   quality {o.get('quality_label')}",
        f"Improvement {rep.get('improvement_pct', 0):+.1f}%",
        f"Quality regression {rep.get('quality_regression_pct', 0):.2f}%  allowed {rep.get('quality_budget_pct', 0):.2f}%  "
        f"{'PASS' if rep.get('quality_pass') else 'FAIL'}",
        "",
        "Optimizations:",
    ]
    for opt in rep.get("optimizations") or []:
        lines.append(f"  {opt}")
    s = rep.get("search") or {}
    lines += [
        "",
        f"Search: {s.get('experiments', 0)} experiments, {s.get('skills_retrieved', 0)} prior skills retrieved, "
        f"{s.get('blocked', 0)} known-bad paths skipped, {s.get('skills_learned', 0)} skills learned",
    ]
    console.print(Panel("\n".join(lines), title="Visai Optimization Report"))
