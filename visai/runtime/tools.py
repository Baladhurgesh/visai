"""Strands tools for the kernel loop. Each returns kernel-forge style JSON.

Tools are closures over a LoopState so interventions and the outer loop see the same facts.
The same underlying functions back the Hermes tool_spec export (visai/integrations/hermes).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from visai.memory.store import GateConfig, faster, idea_allowed, log_event
from visai.runtime.state import LoopState, source_hash
from visai.tasks.ops import get_op


def baseline_us(bench: dict[str, Any], mode: str) -> float:
    return float(bench["reference_us"] if mode == "reference" else bench["runtime_us"])


def stock_from_baseline(bench: dict[str, Any], mode: str) -> dict[str, Any]:
    us = baseline_us(bench, mode)
    return {
        "label": f"stock:{mode}",
        "throughput": 1e6 / us,
        "unit": "calls/s (geomean over bench shapes)",
        "latency_us": us,
        "reference_us": bench.get("reference_us"),
        "runtime_us": bench.get("runtime_us"),
        "shapes": bench.get("shapes"),
    }


def _trim_gate(g: dict[str, Any]) -> dict[str, Any]:
    keep = ("status", "n", "n_pass", "n_fail", "n_errors", "n_hidden_fail", "hidden_failure_kinds",
            "visible_failures", "first_error", "problems", "error", "atol_rtol")
    return {k: g.get(k) for k in keep if g.get(k) not in (None, [], "")}


def make_kernel_tools(state: LoopState, gate: GateConfig) -> list:
    op = get_op(state.op)

    @tool
    def check_idea(idea: str) -> dict:
        """Check an idea against persistent do-not-repeat memory before spending a candidate on it.

        Args:
            idea: One sentence: label, idea class, and the concrete change.
        """
        return idea_allowed(idea, scope=state.target)

    @tool
    def run_correctness(label: str, idea_class: str, idea: str, source: str) -> dict:
        """Write a candidate and run Gate 1 (correctness vs the trusted reference on seeded AND hidden inputs).

        Hidden shapes/distributions are never revealed; you only learn which kinds failed.
        Re-submit the same label to repair; a new label is a new idea (checked against do-not-repeat).

        Args:
            label: snake_case file stem, unique per idea, e.g. add_rmsnorm_simd_v1.
            idea_class: idea class (fusion, fewer_launches, vectorize, reduction_strategy, tiling, precision, launch_config_tune, other).
            idea: one-paragraph description of the change and why it should be faster.
            source: full Python module defining kernel(...) with exactly the op signature.
        """
        slot = state.slot
        slot.labels.add(label)
        path = state.path_for(label)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        res = state.backend.evaluate(state.op, path, state.dtype, mode="gate")
        g = res.get("correctness") or {"status": res.get("status", "crash"), "error": res.get("error")}
        if g.get("status") == "pass":
            slot.passed[label] = source_hash(source)
        slot.last_correctness = {"label": label, "cls": idea_class, "idea": idea, **g}
        log_event(state.run_id, "gate1", target=state.target, label=label, status=g.get("status"),
                  n_pass=g.get("n_pass"), n=g.get("n"), attempt=slot.correctness_attempts)
        out = _trim_gate(g)
        out["attempts_left"] = state.max_attempts_per_slot - slot.correctness_attempts
        return out

    @tool
    def run_benchmark(label: str) -> dict:
        """Benchmark a candidate that passed Gate 1, then re-run to confirm (the gate requires two passes).

        Args:
            label: the label that passed run_correctness.
        """
        slot = state.slot
        path = state.path_for(label)
        runs = []
        for _ in range(2):
            res = state.backend.evaluate(state.op, path, state.dtype, mode="bench")
            b = res.get("bench")
            if not b:
                return {"status": "error", "error": res.get("error") or "benchmark failed"}
            runs.append(b)
        first, confirm = runs
        # Speedups are measured against the baseline timed in the SAME process as the candidate,
        # so machine-load drift between processes cannot fake (or hide) a win.
        sp_first = baseline_us(first, state.baseline_mode) / float(first["candidate_us"])
        sp_confirm = baseline_us(confirm, state.baseline_mode) / float(confirm["candidate_us"])
        one = {"throughput": 1.0}
        confirmed = faster(one, {"throughput": sp_first}, gate) and faster(one, {"throughput": sp_confirm}, gate)
        cand_thr = state.stock["throughput"] * sp_first
        confirm_thr = state.stock["throughput"] * sp_confirm
        per_shape = []
        for s in first.get("shapes") or []:
            per_shape.append(
                {
                    "shape": s["shape"],
                    "candidate_us": round(s["candidate_us"], 2),
                    "runtime_us": round(s["runtime_us"], 2),
                    "reference_us": round(s["reference_us"], 2),
                    "speedup_vs_runtime": round(s["runtime_us"] / s["candidate_us"], 3),
                    "speedup_vs_reference": round(s["reference_us"] / s["candidate_us"], 3),
                }
            )
        attempt = slot.last_correctness or {}
        result = {
            "label": label,
            "cls": attempt.get("cls") if attempt.get("label") == label else None,
            "idea": attempt.get("idea") if attempt.get("label") == label else None,
            "correctness": "pass",
            "n_errors": 0,
            "throughput": cand_thr,
            "confirm_throughput": confirm_thr,
            "confirmed": confirmed,
            "unit": state.stock.get("unit"),
            "latency_us": first["candidate_us"],
            "speedup": sp_first,
            "confirm_speedup": sp_confirm,
            "speedup_vs_reference": first["reference_us"] / first["candidate_us"],
            "speedup_vs_runtime": first["runtime_us"] / first["candidate_us"],
            "timed_output_ok": first.get("timed_output_ok") and confirm.get("timed_output_ok"),
            "per_shape": per_shape,
            "source": path.read_text(),
        }
        if not result["timed_output_ok"]:
            result["correctness"] = "fail"
            result["n_errors"] = 1
        slot.result = result
        log_event(state.run_id, "benchmark", target=state.target, label=label, speedup=round(result["speedup"], 4),
                  confirmed=confirmed)
        view = {k: v for k, v in result.items() if k != "source"}
        view["baseline_mode"] = state.baseline_mode
        view["gate"] = gate.describe()
        return view

    @tool
    def op_spec() -> dict:
        """Return the op contract: signature, reference and runtime sources, visible shapes, bench shapes."""
        return {
            "name": op.name,
            "signature": op.signature,
            "description": op.description,
            "reference_source": op.reference_source(),
            "runtime_baseline_source": op.runtime_source(),
            "visible_shapes": op.visible_shapes,
            "bench_shapes": op.bench_shapes,
            "dtype": state.dtype,
            "notes": op.notes,
        }

    return [op_spec, check_idea, run_correctness, run_benchmark]
