"""Outer optimization loop (kernel-forge launch_query.txt, generalized).

Plain Python owns the policy (budget, gates, confirmation, stop rule). Strands agents do the
creative work (write/repair a candidate, reflect on the result). Atlas holds all memory.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from visai.backends.mlx_metal import METAL_GUIDE, MLXMetalBackend
from visai.config import RUNS
from visai.jsonio import utc_stamp
from visai.memory import store as mem
from visai.memory.store import GateConfig
from visai.runtime.agents import build_reflector, build_writer
from visai.runtime.interventions import kernel_interventions
from visai.runtime.state import LoopState
from visai.runtime.tools import make_kernel_tools, stock_from_baseline
from visai.schemas import Reflection, WriterProposal
from visai.skills.update import retrieve_skills, upsert_skill_from_trial
from visai.tasks.ops import get_op


def target_name(op: str, backend: str, hardware: str, dtype: str) -> str:
    return f"{op}@{backend}:{hardware}:{dtype}"


def _failed_candidate(state: LoopState, proposal: WriterProposal | None, error: str = "") -> dict[str, Any]:
    last = state.slot.last_correctness or {}
    status = last.get("status") or ("no_candidate" if not error else "agent_error")
    return {
        "label": last.get("label") or (proposal.label if proposal else f"slot{state.slot.index}_none"),
        "cls": last.get("cls") or (proposal.cls if proposal else "other"),
        "idea": last.get("idea") or (proposal.hypothesis if proposal else ""),
        "correctness": "compile_error" if status in ("compile_error", "crash", "timeout") else (status if status != "pass" else "unbenchmarked"),
        "n_errors": int(last.get("n_fail") or 1),
        "throughput": None,
        "gate1": {k: last.get(k) for k in ("status", "n_pass", "n", "n_hidden_fail", "hidden_failure_kinds", "first_error", "problems")},
        "agent_error": error[:500] if error else None,
    }


def optimize_op(
    op_name: str,
    *,
    dtype: str = "float16",
    budget: int = 10,
    baseline_mode: str = "runtime",
    gate: GateConfig | None = None,
    run_id: str | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> dict[str, Any]:
    gate = gate or GateConfig()
    op = get_op(op_name)
    backend = MLXMetalBackend()
    target = target_name(op.name, backend.name, backend.hardware, dtype)
    run_id = run_id or f"{utc_stamp()}_{op.name}"
    cand_dir = RUNS / run_id / "candidates"
    say = on_event or (lambda step, data: None)

    mem.start_run(run_id, kind="kernel", target=target, op=op.name, backend=backend.name,
                  hardware=backend.hardware, dtype=dtype, budget=budget, baseline_mode=baseline_mode)
    say("baseline", {"target": target})
    base = backend.baseline(op.name, dtype)
    stock = stock_from_baseline(base, baseline_mode)
    mem.log_event(run_id, "baseline", target=target, stock={k: v for k, v in stock.items() if k != "shapes"})
    state = LoopState(run_id=run_id, target=target, op=op.name, dtype=dtype, backend=backend,
                      cand_dir=cand_dir, stock=stock, baseline_mode=baseline_mode)
    tools = make_kernel_tools(state, gate)
    guards = kernel_interventions(state)
    query = f"{op.name} {op.family} {op.description} {backend.hardware} {dtype} fusion reduction metal"
    skills_retrieved = retrieve_skills(query, operator=op.name, backend=backend.name, hardware=backend.hardware, k=5)
    mem.log_event(run_id, "skills_retrieved", target=target, skills=[s.get("name") for s in skills_retrieved])

    trials: list[dict[str, Any]] = []
    blocked_before = mem.get_db().events.count_documents({"run_id": run_id, "step": "blocked_idea"})
    winner = None
    for slot in range(1, budget + 1):
        state.new_slot(slot)
        refl = mem.latest_reflection(target)
        ctx = {
            "target": target, "backend_guide": METAL_GUIDE, "gate": gate.describe(), "baseline_mode": baseline_mode,
            "stock": {k: v for k, v in stock.items() if k != "shapes"} | {"per_shape": stock.get("shapes")},
            "op_spec": tools[0](),  # op_spec tool is a plain callable too
            "skills": skills_retrieved, "lessons": mem.refresh_lessons(target),
            "next_try": (refl or {}).get("next_try"), "slot": slot, "budget": budget,
        }
        say("slot", {"slot": slot})
        mem.log_event(run_id, "slot_start", target=target, slot=slot)
        proposal: WriterProposal | None = None
        err = ""
        t0 = time.time()
        try:
            writer = build_writer(run_id, target, backend.name, tools[1:], guards, slot)
            from visai.agent.prompts import writer_task

            res = writer(writer_task(ctx))
            proposal = getattr(res, "structured_output", None)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            mem.log_event(run_id, "writer_error", target=target, slot=slot, error=err[:500])
        cand = state.slot.result or _failed_candidate(state, proposal, err)
        if proposal is not None:
            cand.setdefault("cls", proposal.cls)
            cand["hypothesis"] = proposal.hypothesis
            cand["skills_used"] = proposal.skills_used
        if cand.get("source"):
            cand["source"] = cand["source"][:20000]
        trial = mem.record_trial(
            run_id, target, stock, cand, gate,
            meta={"slot": slot, "op": op.name, "backend": backend.name, "hardware": backend.hardware,
                  "dtype": dtype, "writer_seconds": round(time.time() - t0, 1),
                  "gate1_attempts": state.slot.correctness_attempts},
            reason=str(cand.get("idea") or ""),
        )
        trials.append(trial)
        say("trial", {"slot": slot, "label": cand.get("label"), "genuine": trial["genuine"],
                      "outcome": trial["miss"]["outcome"], "speedup": cand.get("speedup")})

        reflection: dict[str, Any] = {}
        try:
            from visai.agent.prompts import reflect_task

            r = build_reflector(run_id, target)(reflect_task(
                {"target": target, "gate": gate.describe(), "trial": {k: v for k, v in trial.items() if k != "_id"},
                 "recent": mem.recent_trials(target, limit=6), "lessons": mem.refresh_lessons(target)}
            ))
            out: Reflection | None = getattr(r, "structured_output", None)
            if out is not None:
                reflection = out.model_dump(by_alias=True)
        except Exception as exc:  # noqa: BLE001
            mem.log_event(run_id, "reflect_error", target=target, slot=slot, error=f"{type(exc).__name__}: {exc}"[:500])
        if reflection:
            mem.save_reflection(run_id, target, reflection)
        if cand.get("correctness") == "pass" or reflection.get("skill"):
            upsert_skill_from_trial(
                trial, operator=op.name, backend=backend.name, hardware=backend.hardware, dtype=dtype,
                shape=op.bench_shapes, reflection_skill=reflection.get("skill"), source=run_id,
            )
        mem.refresh_lessons(target)
        if trial["genuine"]:
            from visai.memory.kernels import register_kernel

            if cand.get("source"):
                register_kernel(cand["source"], status="kernel_winner", kind="op", op=op.name, hardware=backend.hardware,
                                dtype=dtype, label=cand.get("label"), speedup=cand.get("speedup"), run_id=run_id,
                                target=target)
            winner = trial
            say("win", {"label": cand.get("label"), "speedup": cand.get("speedup")})
            break

    blocked = mem.get_db().events.count_documents({"run_id": run_id, "step": "blocked_idea"}) - blocked_before
    summary = {
        "run_id": run_id,
        "target": target,
        "stock": {k: v for k, v in stock.items() if k != "shapes"},
        "n_trials": len(trials),
        "winner": ({k: v for k, v in winner["candidate"].items() if k != "source"} if winner else None),
        "skills_retrieved": [s.get("name") for s in skills_retrieved],
        "blocked_ideas": blocked,
        "trials": [
            {"slot": t.get("slot"), "label": t["candidate"].get("label"), "cls": t["candidate"].get("cls"),
             "correctness": t["candidate"].get("correctness"), "speedup": t["candidate"].get("speedup"),
             "confirmed": t["candidate"].get("confirmed"), "outcome": t["miss"]["outcome"],
             "miss_cls": t["miss"]["cls"], "keep": t["genuine"]}
            for t in trials
        ],
    }
    mem.finish_run(run_id, summary={k: v for k, v in summary.items() if k != "trials"}, n_trials=len(trials),
                   won=bool(winner))
    return summary
