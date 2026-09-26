"""End-to-end Visai pipeline for one model (Qwen3-0.6B, Parakeet, Nemotron-3 diarization) on this Mac.

1. BASELINE      stock model on fixed data points: e2e latency/throughput + task quality
2. PROFILE       exclusive time per layer class on a real forward pass; capture real activations
3. LAYER LOOPS   for each bottleneck layer: writer agent iterates layer kernels (Gate 1 = model's own layer
                 on real + hidden inputs; benchmark twice). No win threshold: keep the best confirmed
                 speedup; stop on agent-declared saturation, patience, iteration cap, or time cap.
4. INTEGRATE     put each winning layer kernel into the whole model; keep it only if e2e latency improves
                 (confirmed) and quality stays within budget of the original.
5. MODEL LEVEL   config search (quantization / precision / scheduling ...) on top of the kept layer kernels,
                 same saturation rules.
6. COMPARE       unoptimized vs optimized head-to-head on the same data points, same machine.
"""

from __future__ import annotations

import inspect
import json
import time
from typing import Any, Callable

from strands import Agent
from strands.session import RepositorySessionManager

from visai.agent.prompts import REFLECT_SYSTEM, reflect_task
from visai.backends.module_patch import resolve_class
from visai.config import RUNS
from visai.deploy.search_loop import SATURATION_NOTE, SearchSpec, run_config_search
from visai.jsonio import utc_now, utc_stamp
from visai.memory import store as mem
from visai.memory.db import get_db
from visai.memory.store import GateConfig
from visai.models.adapters import Adapter, free, get_adapter
from visai.profiler.hardware import probe
from visai.profiler.layers import format_table, profile_layers
from visai.runtime.hooks import AtlasEventHooks
from visai.runtime.interventions import kernel_interventions
from visai.runtime.layer_tools import LAYER_GUIDE, LayerBackend, make_layer_tools
from visai.runtime.memory_store import AtlasMemoryStore
from visai.runtime.models import reflect_model, writer_model
from visai.runtime.session_repo import AtlasSessionRepository
from visai.runtime.state import LoopState
from visai.schemas import Reflection, WriterProposal
from visai.skills.update import retrieve_skills, upsert_skill_from_trial

# Vendor GEMM / gather layers: change their precision at the model level instead of rewriting them.
SKIP_LAYERS = {"Linear", "QuantizedLinear", "Embedding", "QuantizedEmbedding"}

LAYER_SYSTEM = """You are Visai's kernel engineer for ONE bottleneck layer of a real model on Apple Silicon.
Goal: make this layer class faster while producing the same outputs as the model's own implementation.

Per candidate:
1. Read layer_spec (source, weights, captured input shapes, profile share) and the memory context.
2. Pick ONE idea of a different class than the last miss (check_idea if unsure).
3. run_correctness with the full source; repair the SAME label on failure (hidden cases are secret).
4. When Gate 1 passes, run_benchmark once. Speedup is vs the original layer on real shapes.
5. Return the structured proposal. """ + SATURATION_NOTE

DEFAULT_BUDGETS = {"qwen": {"quality_budget_rel": 0.01}, "parakeet": {"quality_budget_abs": 0.003},
                   "diar": {"quality_budget_abs": 0.005}}


def _layer_spec(root, row: dict, captures: list[str], adapter_key: str = "", model_id: str = "") -> dict[str, Any]:
    from mlx.utils import tree_flatten

    from visai.profiler.layers import load_capture

    cls = resolve_class(row["class"])
    inst = dict(root.named_modules()).get(row["example_path"])
    params = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tree_flatten(inst.parameters())} if inst else {}
    attrs = {k: v for k, v in vars(inst).items() if isinstance(v, (int, float, bool, str)) and not k.startswith("_")} if inst else {}
    children = {k: type(v).__name__ for k, v in (inst.children().items() if inst else [])}
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        src = "(source unavailable)"
    sigs = []
    for c in captures:
        _, _, spec = load_capture(c)
        sigs.append({"module_path": spec["module_path"], "signature": spec["signature"]})
    return {"class": row["class"], "name": row["name"], "adapter": adapter_key, "model": model_id,
            "profile": {k: row[k] for k in ("pct", "exclusive_ms", "calls", "us_per_call")},
            "source": src[:9000], "parameters": params, "attributes": attrs, "children": children,
            "captured_inputs": sigs, "instances_patched": "ALL instances of this class in the model"}


def _prior_layer_results(run_ids: str) -> dict[str, dict]:
    """Merge layer results across one or more previous runs (comma-separated ids)."""
    merged: dict[str, dict] = {}
    for rid in [r.strip() for r in run_ids.split(",") if r.strip()]:
        for name, rec in _prior_layer_results_one(rid).items():
            cur = merged.setdefault(name, {"best": None, "complete": False, "stop": rec["stop"]})
            if rec["best"] and (cur["best"] is None or rec["best"]["robust_speedup"] > cur["best"]["robust_speedup"]):
                cur["best"] = rec["best"]
            if rec["complete"]:
                cur["complete"], cur["stop"] = True, rec["stop"]
    return merged


def _prior_layer_results_one(run_id: str) -> dict[str, dict]:
    """Best confirmed layer candidate per layer from a previous run, and whether that layer's loop finished."""
    events = get_db().events.find({"run_id": run_id}, sort=[("ts", 1)])
    reached_integration = any(e["step"] in ("integration", "e2e_after_layers", "deploy_eval") for e in events)
    out: dict[str, dict] = {}
    for e in events:
        tgt = str(e.get("target") or "")
        if tgt.count(":") < 2 or "@" not in tgt or ":e2e@" in tgt:
            continue
        name = tgt.split(":")[1].split("@")[0]
        rec = out.setdefault(name, {"best": None, "complete": reached_integration, "stop": "integration reached"})
        if e["step"] == "benchmark" and str(e.get("improved")) == "True":
            sp = float(e.get("speedup") or 0)
            path = RUNS / run_id / "layers" / name / f"{e.get('label')}.py"
            if path.exists() and (rec["best"] is None or sp > rec["best"]["robust_speedup"]):
                rec["best"] = {"robust_speedup": sp, "label": e.get("label"), "path": str(path)}
        elif e["step"] in ("saturated", "target_reached"):
            rec["complete"] = True
            rec["stop"] = e.get("reason") or f"target reached {e.get('speedup')}"
    return out


def _layer_loop(run_id: str, adapter: Adapter, row: dict, captures: list[str], spec: dict, hardware: str,
                max_iters: int, patience: int, max_minutes: float, say: Callable,
                target_speedup: float | None = None, initial_best: dict | None = None) -> dict[str, Any]:
    target = f"{adapter.key}:{row['name']}@mlx:{hardware}"
    backend = LayerBackend(adapter.key, {}, row["class"], captures, model_id=adapter.model_id)
    base = backend.baseline()
    if not base:
        return {"target": target, "error": "baseline failed", "best": None, "trials": []}
    stock = {"label": "original", "throughput": 1e6 / base["runtime_us"], "unit": "layer calls/s (geomean over captured shapes)",
             "latency_us": base["runtime_us"]}
    mem.log_event(run_id, "layer_baseline", target=target, layer=row["name"], runtime_us=base["runtime_us"])
    state = LoopState(run_id=run_id, target=target, op=row["name"], dtype=adapter.dtype, backend=backend,
                      cand_dir=RUNS / run_id / "layers" / row["name"], stock=stock)
    best: dict[str, Any] = dict(initial_best or {"robust_speedup": 1.0, "label": None, "path": None})
    tools = make_layer_tools(state, spec, best)
    guards = kernel_interventions(state)
    skills = retrieve_skills(f"{row['name']} layer kernel {adapter.model_id} {hardware} fusion", backend="mlx",
                             hardware=hardware, operator=row["name"], k=5)
    mem.log_event(run_id, "skills_retrieved", target=target, skills=[s.get("name") for s in skills])
    trials, streak, t0 = [], 0, time.time()
    stop_reason = f"iteration cap {max_iters}"
    for it in range(1, max_iters + 1):
        if (time.time() - t0) / 60 > max_minutes:
            stop_reason = f"time cap {max_minutes} min"
            break
        state.new_slot(it)
        refl = mem.latest_reflection(target)
        prompt = (
            f"TARGET LAYER: {row['name']} ({row['class']}) in {adapter.model_id}, {row['pct']:.1f}% of profiled time, "
            f"{row['calls']} calls, {row['us_per_call']:.1f} us/call.\n{LAYER_GUIDE}\n"
            f"ORIGINAL LAYER: {base['runtime_us']:.2f} us (geomean over captured shapes)\n"
            f"CURRENT BEST: {best['label']} robust speedup {best['robust_speedup']:.3f}x"
            + (f"   TARGET: {target_speedup:.2f}x over the original layer (the loop stops once reached)\n" if target_speedup else "\n") +
            f"LAYER SPEC:\n{json.dumps(spec, indent=1, default=str)[:12000]}\n\n"
            f"RETRIEVED SKILLS:\n{json.dumps(skills, default=str)[:3000]}\n\n"
            f"LESSONS.md:\n{mem.refresh_lessons(target)[-3500:]}\n"
            f"LAST REFLECTION next_try: {json.dumps((refl or {}).get('next_try'))}\n"
            f"ITERATION {it} (hard cap {max_iters}; {streak} non-improving in a row, patience {patience})."
        )
        say("layer_iter", {"layer": row["name"], "iter": it})
        proposal = None
        try:
            agent = Agent(
                name="visai_layer_writer", agent_id=f"layer_{row['name']}_{it}", model=writer_model(),
                system_prompt=LAYER_SYSTEM, tools=tools, interventions=guards,
                hooks=[AtlasEventHooks(run_id, "writer", target)],
                memory_manager={"stores": [AtlasMemoryStore(scope=target, backend="mlx")], "injection": False},
                session_manager=RepositorySessionManager(session_id=f"{run_id}_{row['name']}_{it}",
                                                         session_repository=AtlasSessionRepository()),
                structured_output_model=WriterProposal, callback_handler=None,
            )
            proposal = getattr(agent(prompt), "structured_output", None)
        except Exception as exc:  # noqa: BLE001
            mem.log_event(run_id, "writer_error", target=target, error=f"{type(exc).__name__}: {exc}"[:500])
        cand = state.slot.result
        if cand is None and proposal is not None and proposal.saturated:
            stop_reason = f"writer declared saturation: {proposal.saturation_reason}"
            mem.log_event(run_id, "saturated", target=target, by="writer", reason=proposal.saturation_reason)
            break
        if cand is None:
            last = state.slot.last_correctness or {}
            cand = {"label": last.get("label") or (proposal.label if proposal else f"iter{it}_none"),
                    "cls": last.get("cls") or (proposal.cls if proposal else "other"),
                    "correctness": {"pass": "unbenchmarked", None: "no_candidate"}.get(last.get("status"), last.get("status") or "no_candidate"),
                    "n_errors": int(last.get("n_fail") or 1), "throughput": None,
                    "gate1": {k: last.get(k) for k in ("status", "n_pass", "n", "hidden_failure_kinds", "first_error", "problems")}}
        if cand.get("source"):
            cand["source"] = cand["source"][:20000]
        cmp_stock = dict(stock)
        if best["label"] and best["label"] != cand.get("label"):
            cmp_stock["throughput"] = stock["throughput"] * best["robust_speedup"]
        trial = mem.record_trial(run_id, target, cmp_stock, cand, GateConfig(rel=0.0),
                                 meta={"slot": it, "op": row["name"], "layer_class": row["class"], "backend": "mlx",
                                       "hardware": hardware, "model": adapter.model_id, "stage": "layer"},
                                 reason=str(cand.get("idea") or ""))
        trials.append(trial)
        streak = 0 if cand.get("improved_over_best") else streak + 1
        reflection: dict[str, Any] = {}
        try:
            r = Agent(name="visai_reflect", model=reflect_model(), system_prompt=REFLECT_SYSTEM + "\n" + SATURATION_NOTE,
                      tools=[], hooks=[AtlasEventHooks(run_id, "reflect", target)],
                      structured_output_model=Reflection, callback_handler=None)(
                reflect_task({"target": target, "gate": "no threshold; beat current best, confirmed twice",
                              "trial": {k: v for k, v in trial.items() if k not in ("_id",)} | {"candidate": {k: v for k, v in cand.items() if k != "source"}},
                              "recent": mem.recent_trials(target, 6), "lessons": mem.refresh_lessons(target)}))
            if getattr(r, "structured_output", None) is not None:
                reflection = r.structured_output.model_dump(by_alias=True)
                mem.save_reflection(run_id, target, reflection)
        except Exception as exc:  # noqa: BLE001
            mem.log_event(run_id, "reflect_error", target=target, error=f"{type(exc).__name__}: {exc}"[:300])
        if cand.get("correctness") == "pass" or reflection.get("skill"):
            upsert_skill_from_trial(trial, operator=row["name"], backend="mlx", hardware=hardware, dtype=adapter.dtype,
                                    shape=spec["captured_inputs"], reflection_skill=reflection.get("skill"), source=run_id)
        if target_speedup and best["robust_speedup"] >= target_speedup:
            stop_reason = f"reached target {target_speedup:.2f}x ({best['robust_speedup']:.3f}x)"
            mem.log_event(run_id, "target_reached", target=target, speedup=best["robust_speedup"])
            break
        if (proposal is not None and proposal.saturated) or reflection.get("saturated"):
            why = reflection.get("saturation_reason") or (proposal.saturation_reason if proposal else "") or "agent judged saturation"
            stop_reason = f"agent declared saturation: {why}"
            mem.log_event(run_id, "saturated", target=target, reason=why)
            break
        if streak >= patience:
            stop_reason = f"{patience} non-improving iterations in a row"
            mem.log_event(run_id, "saturated", target=target, reason=stop_reason)
            break
    say("layer_done", {"layer": row["name"], "best": best, "stop": stop_reason})
    return {"target": target, "layer": row["name"], "class": row["class"], "original_us": base["runtime_us"],
            "best": best if best["label"] else None, "iterations": len(trials), "stop_reason": stop_reason}


def optimize_pipeline(
    key: str,
    *,
    top_layers: int = 2,
    max_layer_iters: int = 6,
    layer_patience: int = 3,
    layer_minutes: float = 25.0,
    max_config_iters: int = 6,
    config_patience: int = 3,
    config_minutes: float = 30.0,
    quality_budget_rel: float | None = None,
    quality_budget_abs: float | None = None,
    min_layer_pct: float = 3.0,
    target_speedup: float | None = None,
    resume_from: str | None = None,
    model_id: str | None = None,
    hardware_label: str | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> dict[str, Any]:
    say = on_event or (lambda s, d: None)
    adapter = get_adapter(key, model_id=model_id)
    key = adapter.key
    hw = probe()
    hardware = hw.get("chip", "apple-silicon")
    run_id = f"{utc_stamp()}_pipeline_{key}"
    budgets = dict(DEFAULT_BUDGETS.get(key) or DEFAULT_BUDGETS["qwen"])
    if quality_budget_rel is not None:
        budgets = {"quality_budget_rel": quality_budget_rel}
    if quality_budget_abs is not None:
        budgets = {"quality_budget_abs": quality_budget_abs}
    qgate = GateConfig(rel=0.0, **budgets)
    model_target = f"{adapter.model_id}:e2e@mlx:{hardware}"
    mem.start_run(run_id, kind="pipeline", target=model_target, model=adapter.model_id, adapter=key, hardware=hardware,
                  quality_budget=qgate.describe(), target_speedup=target_speedup,
                  hardware_requested=hardware_label or f"this Mac ({hw.get('chip_name', hardware)}, MLX/Metal)",
                  task=adapter.task, unit=adapter.unit, quality_metric=adapter.quality_name)
    t_start = time.time()

    # 1) BASELINE
    say("baseline", {"model": adapter.model_id})
    stock = {"label": "stock", **adapter.evaluate({})}
    mem.log_event(run_id, "baseline", target=model_target, stock=adapter.describe(stock))

    # 2) PROFILE (or reuse the profile + captures of a previous run)
    say("profile", {"resume_from": resume_from})
    h = adapter.load({})
    prior: dict[str, dict] = {}
    if resume_from:
        first = resume_from.split(",")[0].strip()
        prev = get_db().profiles.find_one({"run_id": first, "kind": "layers"})
        if not prev:
            raise RuntimeError(f"no layer profile stored for {first}")
        caps_dir = RUNS / first / "captures"
        prof = {"layers": prev["layers"], "captures": {}}
        for r in prev["layers"]:
            files = sorted(str(p.with_suffix("")) for p in caps_dir.glob(f"{r['name']}_*.json"))
            if files:
                prof["captures"][r["class"]] = files
        prior = _prior_layer_results(resume_from)
    else:
        prof = profile_layers(adapter.root(h), lambda: adapter.profile_run(h, {}), capture_dir=RUNS / run_id / "captures")
    targets = [r for r in prof["layers"] if r["capturable"] and r["name"] not in SKIP_LAYERS and r["pct"] >= min_layer_pct
               and prof["captures"].get(r["class"])][:top_layers]
    specs = {r["class"]: _layer_spec(adapter.root(h), r, prof["captures"].get(r["class"], []), key, adapter.model_id)
             for r in targets}
    del h
    free()
    get_db().profiles.insert_one({"run_id": run_id, "target": model_target, "kind": "layers", "ts": utc_now(),
                                  "layers": prof["layers"][:25], "targets": [t["name"] for t in targets],
                                  "table": format_table(prof)})
    mem.log_event(run_id, "profile", target=model_target,
                  layers=[(r["name"], round(r["pct"], 1)) for r in prof["layers"][:8]], targets=[t["name"] for t in targets])
    say("profile_table", {"table": format_table(prof)})

    # 3) LAYER LOOPS
    layer_results = []
    for r in targets:
        p = prior.get(r["name"])
        if p and p["complete"]:
            say("layer_resumed", {"layer": r["name"], "best": p["best"], "stop": p["stop"]})
            layer_results.append({"target": f"{key}:{r['name']}@mlx:{hardware}", "layer": r["name"], "class": r["class"],
                                  "best": p["best"], "iterations": 0, "stop_reason": f"resumed from {resume_from}: {p['stop']}"})
            continue
        layer_results.append(_layer_loop(run_id, adapter, r, prof["captures"][r["class"]], specs[r["class"]], hardware,
                                         max_layer_iters, layer_patience, layer_minutes, say, target_speedup,
                                         initial_best=(p or {}).get("best")))

    # 4) INTEGRATE winning layer kernels into the whole model
    current = {"config": {}, "metrics": stock, "label": "stock"}
    patches: list[dict] = []
    integration = []
    for lr in layer_results:
        if not lr.get("best"):
            integration.append({"layer": lr["layer"], "kept": False, "why": "no layer candidate beat the original"})
            continue
        from visai.memory.kernels import kernel_id as _kid

        src_path = lr["best"]["path"]
        trial_patches = patches + [{"kind": "module", "class": lr["class"], "candidate": src_path,
                                    "kernel_id": lr["best"].get("kernel_id") or _kid(open(src_path).read())}]
        say("integrate", {"layer": lr["layer"]})
        try:
            m = adapter.evaluate({"patches": trial_patches})
        except Exception as exc:  # noqa: BLE001
            integration.append({"layer": lr["layer"], "kept": False, "why": f"e2e run failed: {type(exc).__name__}: {exc}"[:300]})
            continue
        from visai.memory.store import quality_ok

        q_ok = quality_ok(stock, m, qgate)
        confirmed, pinfo = False, {"ratio": 0.0, "wins": 0, "rounds": 0}
        if q_ok:
            # paired A/B/A/B vs the current config: both sides see the same machine load
            confirmed, pinfo = adapter.paired_faster({"patches": patches}, {"patches": trial_patches})
        cand = {"label": f"layer:{lr['layer']}:{lr['best']['label']}", "cls": "fusion", "correctness": "pass",
                "config": {"patches": trial_patches}, "confirmed": confirmed, **m,
                "measured_throughput": m["throughput"], "paired_vs_current": pinfo,
                "throughput": current["metrics"]["throughput"] * (pinfo["ratio"] or 0.0)}
        trial = mem.record_trial(run_id, model_target, current["metrics"] | {"quality": stock.get("quality")}, cand,
                                 GateConfig(rel=0.0, **budgets),
                                 meta={"stage": "integration", "op": "layer_integration", "layer": lr["layer"],
                                       "backend": "mlx", "hardware": hardware, "model": adapter.model_id})
        kept = trial["genuine"]
        from visai.memory.kernels import mark_integration

        mark_integration(src_path, kept=kept, e2e_paired=float(pinfo["ratio"] or 0), wins=int(pinfo["wins"]), run_id=run_id)
        if kept:
            patches = trial_patches
            current = {"config": {"patches": patches}, "metrics": cand, "label": cand["label"]}
        integration.append({"layer": lr["layer"], "kept": kept, "layer_speedup": lr["best"]["robust_speedup"],
                            "paired_speedup_vs_current": pinfo["ratio"], "paired_wins": f"{pinfo['wins']}/{pinfo['rounds']}",
                            "quality": m.get(adapter.quality_name.lower()), "why": trial["miss"]["why"]})
        mem.log_event(run_id, "integration", target=model_target, layer=lr["layer"], kept=kept,
                      paired_speedup=round(pinfo["ratio"], 4), wins=pinfo["wins"])

    e2e_now = adapter.paired_speed({}, {"patches": patches})["ratio"] if patches else 1.0
    mem.log_event(run_id, "e2e_after_layers", target=model_target, paired_speedup=round(e2e_now, 4))

    # 5) MODEL-LEVEL optimization on top of the kept layer kernels
    say("model_level", {})
    spec = SearchSpec(
        target=model_target, operator=f"{key}_model_level", hardware=hardware,
        system_prompt=("You are Visai's model-level optimizer for on-device inference (MLX on Apple Silicon). "
                       "Layer kernels already applied stay fixed; you choose the deployment config on top of them. "
                       "Per candidate: pick ONE config idea of a different class than the last miss, call "
                       "evaluate_deploy_config once (retry once only if invalid), return the structured proposal."),
        search_space_doc=adapter.search_space_doc, quality_name=adapter.quality_name,
        context=(f"MODEL: {adapter.model_id}  TASK: {adapter.task}\nHARDWARE: {json.dumps(hw)}\n"
                 f"LAYER PROFILE (self time %): {json.dumps([(r['name'], round(r['pct'], 1)) for r in prof['layers'][:10]])}\n"
                 f"KEPT LAYER KERNELS: {[p['class'] for p in patches]}"),
        evaluate=lambda cfg: adapter.evaluate({**cfg, "patches": patches}),
        validate=adapter.validate,
        speed_of=lambda cfg: adapter.speed_of({**cfg, "patches": patches}),
        describe_metrics=adapter.describe,
        paired=lambda a, b: adapter.paired_faster({**a, "patches": patches}, {**b, "patches": patches}),
        e2e_vs_stock=lambda c: adapter.paired_speed({}, {**c, "patches": patches})["ratio"],
    )
    ml = run_config_search(run_id, spec, stock, qgate, max_config_iters,
                           start={"config": {}, "metrics": current["metrics"], "label": current["label"],
                                  "e2e_vs_stock": e2e_now},
                           patience=config_patience, max_minutes=config_minutes, target_speedup=target_speedup)
    final_cfg = {**(ml["best"]["config"] or {}), "patches": patches}

    # 6) COMPARE unoptimized vs optimized head-to-head
    say("compare", {"final_config": final_cfg})
    comparison = compare(adapter, final_cfg, run_id=run_id)
    rep = {
        "run_id": run_id, "model": adapter.model_id, "adapter": key, "hardware": hardware,
        "quality_budget": qgate.describe(),
        "profile_top": [(r["name"], round(r["pct"], 1), r["calls"]) for r in prof["layers"][:10]],
        "layer_targets": [t["name"] for t in targets],
        "layers": layer_results, "integration": integration,
        "model_level": {"best_label": ml["best"]["label"], "best_config": ml["best"]["config"],
                        "stop_reason": ml["stop_reason"], "candidates": len(ml["trials"])},
        "final_config": final_cfg, "comparison": comparison,
        "elapsed_min": round((time.time() - t_start) / 60, 1),
    }
    mem.finish_run(run_id, report={k: v for k, v in rep.items() if k != "layers"} | {
        "layers": [{k: v for k, v in lr.items() if k != "trials"} for lr in layer_results]})
    return rep


def compare(adapter: Adapter, final_cfg: dict, *, run_id: str | None = None, runs: int = 3) -> dict[str, Any]:
    """Unoptimized vs optimized on the same data points, same machine.

    Speed is measured interleaved (A/B/A/B...) so both sides see the same machine load; quality and memory
    come from a separate full evaluation of each side.
    """
    base = adapter.evaluate({}, runs=1)
    opt = adapter.evaluate(final_cfg, runs=1)
    p = adapter.paired_speed({}, final_cfg, rounds=runs)
    for m, thr in ((base, p["a_median"]), (opt, p["b_median"])):
        m["latency_s"] = m["latency_s"] * m["throughput"] / thr
        m["throughput"] = thr
    q = adapter.quality_name.lower()
    out = {
        "ts": utc_now(), "model": adapter.model_id, "adapter": adapter.key, "hardware": probe().get("chip"),
        "unit": adapter.unit, "quality_metric": adapter.quality_name,
        "unoptimized": adapter.describe(base), "optimized": adapter.describe(opt),
        "speedup": p["ratio"], "paired": p,
        "latency_reduction_pct": 100 * (1 - 1 / p["ratio"]),
        "quality_delta": opt.get(q, 0) - base.get(q, 0),
        "memory_delta_gb": opt.get("peak_memory_gb", 0) - base.get("peak_memory_gb", 0),
        "final_config": final_cfg,
    }
    if run_id:
        out["run_id"] = run_id
    get_db().comparisons.insert_one(dict(out))
    return out
