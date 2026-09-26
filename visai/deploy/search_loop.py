"""Generic agent-driven config search (shared by LLM, ASR, and diarization model-level optimization).

Saturation mode: no fixed win threshold. A candidate is kept if it is faster than the CURRENT BEST on two
measurements (confirmation) and task quality stays within budget of the ORIGINAL stock. The loop stops when
the writer or the reflect agent declares saturation, after `patience` non-improving candidates in a row,
after `budget` candidates, or after `max_minutes` of wall-clock time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from strands import Agent, tool
from strands.session import RepositorySessionManager

from visai.agent.prompts import REFLECT_SYSTEM, reflect_task
from visai.config import RUNS
from visai.memory import store as mem
from visai.memory.store import GateConfig, idea_allowed, quality_ok
from visai.runtime.hooks import AtlasEventHooks
from visai.runtime.interventions import BudgetGuard, DoNotRepeatGuard, WriteScopeGuard
from visai.runtime.memory_store import AtlasMemoryStore
from visai.runtime.models import reflect_model, writer_model
from visai.runtime.session_repo import AtlasSessionRepository
from visai.runtime.state import LoopState
from visai.schemas import Reflection, WriterProposal
from visai.skills.update import retrieve_skills, upsert_skill_from_trial


@dataclass
class SearchSpec:
    target: str
    operator: str
    hardware: str
    system_prompt: str
    search_space_doc: str
    context: str
    quality_name: str
    evaluate: Callable[[dict], dict]
    validate: Callable[[dict], list[str]]
    speed_of: Callable[[dict], float]
    describe_metrics: Callable[[dict], dict] = field(default=lambda m: m)
    # Paired (interleaved) comparisons; when set they replace absolute throughput comparisons.
    paired: Callable[[dict, dict], tuple[bool, dict]] | None = None  # (best_cfg, cand_cfg) -> (faster, info)
    e2e_vs_stock: Callable[[dict], float] | None = None  # cand_cfg -> paired speedup vs unoptimized stock


SATURATION_NOTE = (
    "There is no fixed speedup threshold: any confirmed improvement over the CURRENT BEST is kept. Keep "
    "proposing genuinely different ideas while you believe gains remain. When you judge that remaining ideas "
    "are unlikely to beat the current best (diminishing returns), set saturated=true with a reason."
)


def run_config_search(
    run_id: str,
    spec: SearchSpec,
    stock: dict,
    gate: GateConfig,
    budget: int,
    *,
    start: dict | None = None,
    patience: int = 3,
    max_minutes: float | None = None,
    target_speedup: float | None = None,
    min_candidates: int = 3,
) -> dict[str, Any]:
    """stock: original metrics (quality reference). start: {"config", "metrics", "label"} current best to beat."""
    state = LoopState(run_id=run_id, target=spec.target, op=spec.operator, dtype="-", backend=None,
                      cand_dir=RUNS / run_id, stock=stock, max_attempts_per_slot=2)
    best = dict(start or {"config": {}, "metrics": stock, "label": "stock"})
    qkey = spec.quality_name.lower()
    t0 = time.time()

    @tool
    def check_idea(idea: str) -> dict:
        """Check an idea against persistent do-not-repeat memory.

        Args:
            idea: label, class and concrete config change in one sentence.
        """
        return idea_allowed(idea, scope=spec.target)

    @tool
    def evaluate_deploy_config(label: str, idea_class: str, idea: str, config: dict) -> dict:
        """Run the workload with this deployment config, measure throughput and quality, and compare to the current best.

        Args:
            label: snake_case unique label, e.g. q8_g64_mixed_head.
            idea_class: quantization | kv_cache | scheduling | precision | other.
            idea: why this config should be faster within the quality budget.
            config: deployment config dict (see search space).
        """
        slot = state.slot
        slot.labels.add(label)
        config = dict(config or {})
        errs = spec.validate(config)
        if errs:
            return {"status": "invalid", "errors": errs}
        mem.log_event(run_id, "deploy_eval", target=spec.target, label=label, config=config)
        try:
            m = spec.evaluate(config)
        except Exception as exc:  # noqa: BLE001
            slot.result = {"label": label, "cls": idea_class, "idea": idea, "config": config,
                           "correctness": "compile_error", "n_errors": 1, "throughput": None}
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}
        cand = {"label": label, "cls": idea_class, "idea": idea, "config": config, "correctness": "skipped", **m}
        q_ok = quality_ok(stock, cand, gate)
        cand["confirmed"] = False
        if spec.paired is not None:
            is_fast = False
            if q_ok:
                is_fast, info = spec.paired(best["config"], config)
                cand.update({"paired_vs_best": info, "confirmed": is_fast, "measured_throughput": cand["throughput"]})
                # express throughput relative to the current best so recording/gating use the paired result
                cand["throughput"] = best["metrics"]["throughput"] * info["ratio"]
            cand["speedup_vs_best"] = (cand.get("paired_vs_best") or {}).get("ratio", 0.0)
        else:
            is_fast = cand["throughput"] > best["metrics"]["throughput"]
            if q_ok and is_fast:
                s2, c2 = spec.speed_of(best["config"]), spec.speed_of(config)
                cand.update({"confirm_best_throughput": s2, "confirm_throughput": c2, "confirmed": c2 > s2})
            cand["speedup_vs_best"] = cand["throughput"] / best["metrics"]["throughput"]
        cand["speedup"] = cand["throughput"] / stock["throughput"]
        slot.result = cand
        mem.log_event(run_id, "benchmark", target=spec.target, label=label, speedup=round(cand["speedup"], 4),
                      quality=cand.get(qkey), confirmed=cand["confirmed"])
        return {"status": "ok", "speedup_vs_stock": round(cand["speedup"], 3),
                "speedup_vs_current_best": round(cand["speedup_vs_best"], 3), "quality_ok": q_ok,
                "faster_than_best": is_fast, "confirmed": cand["confirmed"], "quality_budget": gate.describe(),
                "candidate": spec.describe_metrics(m), "current_best": spec.describe_metrics(best["metrics"]),
                "stock": spec.describe_metrics(stock)}

    guards = [WriteScopeGuard(state), BudgetGuard(state), DoNotRepeatGuard(state)]
    skills = retrieve_skills(f"{spec.operator} {spec.target} {spec.context[:300]}", backend="mlx",
                             hardware=spec.hardware, operator=spec.operator, k=5)
    mem.log_event(run_id, "skills_retrieved", target=spec.target, skills=[s.get("name") for s in skills])
    trials: list[dict[str, Any]] = []
    streak, stop_reason = 0, f"budget of {budget} candidates"
    evaluated = 0
    for slot_i in range(1, budget + 1):
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            stop_reason = f"time cap {max_minutes} min"
            break
        state.new_slot(slot_i)
        refl = mem.latest_reflection(spec.target)
        prompt = (
            f"{spec.context}\nQUALITY BUDGET: {gate.describe()} vs ORIGINAL stock (quality = -{spec.quality_name})\n"
            f"{SATURATION_NOTE}\n"
            + (f"TARGET: {target_speedup:.2f}x end-to-end over ORIGINAL stock (search stops once reached); "
               f"current best is {best.get('e2e_vs_stock') or 1.0:.3f}x (paired measurement)\n" if target_speedup else "")
            + f"ORIGINAL STOCK: {json.dumps(spec.describe_metrics(stock), default=str)}\n"
            f"CURRENT BEST: {best['label']} {json.dumps(best['config'], default=str)} "
            f"{json.dumps(spec.describe_metrics(best['metrics']), default=str)}\n\n"
            f"SEARCH SPACE:\n{spec.search_space_doc}\n"
            f"RETRIEVED SKILLS:\n{json.dumps(skills, default=str)[:4000]}\n\n"
            f"LESSONS.md:\n{mem.refresh_lessons(spec.target)[-4000:]}\n\n"
            f"LAST REFLECTION next_try: {json.dumps((refl or {}).get('next_try'))}\n"
            f"CANDIDATE {slot_i} (hard cap {budget}; {streak} non-improving in a row, patience {patience})."
            + (f"\nSaturation is NOT allowed yet: only {evaluated} candidate(s) measured in this run, minimum {min_candidates}. "
               "Past lessons are priors, not proof on this machine/config - re-test the most promising known-good idea "
               "if unsure." if evaluated < min_candidates else "")
        )
        proposal = None
        try:
            agent = Agent(
                name="visai_deploy_writer", agent_id=f"deploy_writer_{slot_i}", model=writer_model(),
                system_prompt=spec.system_prompt, tools=[check_idea, evaluate_deploy_config], interventions=guards,
                hooks=[AtlasEventHooks(run_id, "writer", spec.target)],
                memory_manager={"stores": [AtlasMemoryStore(scope=spec.target, backend="mlx")], "injection": False},
                session_manager=RepositorySessionManager(session_id=f"{run_id}_{spec.operator}_{slot_i}",
                                                         session_repository=AtlasSessionRepository()),
                structured_output_model=WriterProposal, callback_handler=None,
            )
            proposal = getattr(agent(prompt), "structured_output", None)
        except Exception as exc:  # noqa: BLE001
            mem.log_event(run_id, "writer_error", target=spec.target, slot=slot_i, error=f"{type(exc).__name__}: {exc}"[:500])
        cand = state.slot.result
        if cand is None and proposal is not None and proposal.saturated:
            if evaluated >= min_candidates:
                stop_reason = f"writer declared saturation: {proposal.saturation_reason}"
                mem.log_event(run_id, "saturated", target=spec.target, by="writer", reason=proposal.saturation_reason)
                break
            mem.log_event(run_id, "saturation_rejected", target=spec.target, evaluated=evaluated, reason=proposal.saturation_reason)
            continue
        if cand is not None and cand.get("throughput") is not None:
            evaluated += 1
        cand = cand or {"label": (proposal.label if proposal else f"slot{slot_i}_none"),
                        "cls": (proposal.cls if proposal else "other"), "correctness": "no_candidate",
                        "n_errors": 1, "throughput": None}
        ref_gate = GateConfig(rel=0.0, quality_budget_abs=gate.quality_budget_abs,
                              quality_budget_rel=gate.quality_budget_rel, require_confirm=True)
        trial = mem.record_trial(run_id, spec.target, best["metrics"] | {"quality": stock.get("quality")}, cand, ref_gate,
                                 meta={"slot": slot_i, "op": spec.operator, "backend": "mlx", "hardware": spec.hardware},
                                 reason=str(cand.get("idea") or ""))
        trials.append(trial)
        improved = bool(trial["genuine"] and cand.get("throughput") and cand["throughput"] > best["metrics"]["throughput"])
        if improved:
            e2e_ratio = spec.e2e_vs_stock(cand["config"]) if spec.e2e_vs_stock else cand["throughput"] / stock["throughput"]
            best = {"config": cand["config"], "metrics": cand, "label": cand["label"], "e2e_vs_stock": e2e_ratio}
            mem.log_event(run_id, "model_level_best", target=spec.target, label=cand["label"], e2e_vs_stock=round(e2e_ratio, 4))
            streak = 0
        else:
            streak += 1
        reflection: dict[str, Any] = {}
        try:
            r = Agent(name="visai_reflect", model=reflect_model(), system_prompt=REFLECT_SYSTEM + "\n" + SATURATION_NOTE,
                      tools=[], hooks=[AtlasEventHooks(run_id, "reflect", spec.target)],
                      structured_output_model=Reflection, callback_handler=None)(
                reflect_task({"target": spec.target, "gate": gate.describe(),
                              "trial": {k: v for k, v in trial.items() if k != "_id"},
                              "recent": mem.recent_trials(spec.target, 6), "lessons": mem.refresh_lessons(spec.target)}))
            if getattr(r, "structured_output", None) is not None:
                reflection = r.structured_output.model_dump(by_alias=True)
                mem.save_reflection(run_id, spec.target, reflection)
        except Exception as exc:  # noqa: BLE001
            mem.log_event(run_id, "reflect_error", target=spec.target, error=f"{type(exc).__name__}: {exc}"[:300])
        if cand.get("throughput") is not None:
            upsert_skill_from_trial(trial, operator=spec.operator, backend="mlx", hardware=spec.hardware,
                                    dtype=str((cand.get("config") or {}).get("dtype", "-")),
                                    shape={"target": spec.target, "config": cand.get("config")},
                                    reflection_skill=reflection.get("skill"), source=run_id)
        e2e = best.get("e2e_vs_stock") or best["metrics"]["throughput"] / stock["throughput"]
        if target_speedup and e2e >= target_speedup:
            stop_reason = f"reached target {target_speedup:.2f}x end-to-end ({e2e:.3f}x)"
            mem.log_event(run_id, "target_reached", target=spec.target, speedup=e2e)
            break
        if ((proposal is not None and proposal.saturated) or reflection.get("saturated")) and evaluated >= min_candidates:
            why = (reflection.get("saturation_reason") or (proposal.saturation_reason if proposal else "")) or "agent judged saturation"
            stop_reason = f"agent declared saturation: {why}"
            mem.log_event(run_id, "saturated", target=spec.target, reason=why)
            break
        if streak >= patience:
            stop_reason = f"{patience} non-improving candidates in a row"
            mem.log_event(run_id, "saturated", target=spec.target, reason=stop_reason)
            break
    return {"best": best, "trials": trials, "skills_retrieved": len(skills), "stop_reason": stop_reason}
