"""`visai optimize`: model + hardware + workload + quality budget -> gated, faster deployment.

Stages:
1. Baseline: decode tok/s + perplexity (quick Gate 2) on the stock bf16 model.
2. Profile: ProfileBrief (per-op family split) stored in Atlas.
3. Deployment search (shared agent loop): quantization, mixed precision, KV-cache, prefill.
4. Kernels: promoted kernel winners are re-verified in the model dtype (Gate 1) and patched on top of the
   best config; kept only if the e2e gate still passes.
5. Optional full eval (GSM8K slice) on baseline and final.
"""

from __future__ import annotations

import json
import time
from typing import Any

from visai.backends.mlx_metal import MLXMetalBackend
from visai.backends.patchers import MLXModulePatcher
from visai.config import RUNS
from visai.deploy.mlx_search import SEARCH_SPACE_DOC, evaluate_config, free, gen_kwargs, remeasure_speed, validate
from visai.deploy.search_loop import SearchSpec, run_config_search
from visai.jsonio import utc_stamp
from visai.memory import store as mem
from visai.memory.db import get_db
from visai.memory.store import GateConfig, faster, quality_ok
from visai.models.mlx_model import load, measure_decode, model_type
from visai.profiler.hardware import probe
from visai.skills.update import list_skills
from visai.verify.perplexity import perplexity

DEPLOY_SYSTEM = """You are Visai's deployment optimizer for on-device inference (MLX on Apple Silicon).
Decode is memory-bandwidth bound, so bytes moved per token dominate. Find the deployment config with the
highest throughput whose quality stays within the budget.

Per candidate slot:
1. Read the context (baseline, profile, retrieved skills, lessons, do-not-repeat, last reflection).
2. Pick ONE config idea of a different class than the last miss. Use check_idea if unsure.
3. Call evaluate_deploy_config once (you may retry once if the config was invalid).
4. Return the structured proposal: label, cls (quantization | kv_cache | scheduling | precision | other),
   hypothesis, config, skills_used, experiments for next time.

Heuristics: fewer weight bits = fewer bytes = faster, but small models lose quality fast; mixed precision
(keep embeddings / lm_head / first+last layers high) recovers quality; KV-cache quant matters for long contexts.
"""


def _promoted_kernel(op: str, hardware: str) -> dict[str, Any] | None:
    rows = get_db().experiments.find({"op": op, "genuine": True, "hardware": hardware}, sort=[("ts", -1)], limit=1)
    return rows[0] if rows else None


def _llm_metrics(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
            if k in ("throughput", "perplexity", "prefill_tok_s", "peak_memory_gb", "unit")}


def optimize_model(
    model_id: str,
    *,
    workload: str = "decode",
    quality_budget_rel: float = 0.005,
    budget: int = 6,
    kernel_ops: list[str] | None = None,
    full_eval: bool = False,
    hardware_label: str = "",
) -> dict[str, Any]:
    hw = probe()
    hardware = hardware_label or hw.get("chip", "apple-silicon")
    run_id = f"{utc_stamp()}_optimize_{model_id.split('/')[-1]}"
    target = f"{model_id}:{workload}@mlx:{hardware}"
    gate = GateConfig(rel=0.03, quality_budget_rel=quality_budget_rel)
    mem.start_run(run_id, kind="model", target=target, model=model_id, workload=workload, hardware=hardware,
                  quality_budget_rel=quality_budget_rel, budget=budget)
    skills_before = {s["name"] for s in list_skills(500)}
    t_start = time.time()

    model, tok = load(model_id)
    mtype = model_type(model)
    speed = measure_decode(model, tok)
    q = perplexity(model, tok)
    stock = {"label": "stock:bf16", "throughput": speed["median_decode_tok_s"], "unit": "decode tok/s",
             "prefill_tok_s": speed["median_prefill_tok_s"], "peak_memory_gb": speed["peak_memory_gb"],
             "perplexity": q["perplexity"], "quality": q["quality"]}
    mem.log_event(run_id, "baseline", target=target, stock=stock)
    try:
        from visai.profiler.brief import mlx_profile_brief

        brief = mlx_profile_brief(model, tok, model_id, baseline=speed)
        brief.pop("table", None)
    except Exception as exc:  # noqa: BLE001
        from visai.profiler.brief import prior_brief

        brief = prior_brief(model_id, note=f"profile failed: {exc}")
    get_db().profiles.insert_one({"run_id": run_id, "target": target, **brief})
    mem.log_event(run_id, "profile", target=target, families=[(f["name"], f["gpu_pct"]) for f in brief["families"]])
    baseline_full = None
    if full_eval:
        from visai.verify.task_eval import run_eval_inprocess

        baseline_full = run_eval_inprocess(model, tok, n=10)["metrics"]
    del model, tok
    free()

    spec = SearchSpec(
        target=target, operator="model_deploy", hardware=hardware, system_prompt=DEPLOY_SYSTEM,
        search_space_doc=SEARCH_SPACE_DOC, quality_name="perplexity",
        context=(f"MODEL: {model_id} ({mtype})  HARDWARE: {json.dumps(hw)}\nWORKLOAD: {workload}\n"
                 f"PROFILE FAMILIES: {json.dumps([(f['name'], f['gpu_pct']) for f in brief['families']])}"),
        evaluate=lambda cfg: evaluate_config(model_id, {k: v for k, v in cfg.items() if k != "kernels"}),
        validate=validate,
        speed_of=lambda cfg: remeasure_speed(model_id, cfg),
        describe_metrics=_llm_metrics,
    )
    search = run_config_search(run_id, spec, stock, gate, budget)
    best, trials = search["best"], list(search["trials"])

    optimizations: list[str] = []
    if best["label"] != "stock":
        optimizations.append(
            f"deployment config {best['label']}: {json.dumps(best['config'])} -> "
            f"{best['metrics']['throughput'] / stock['throughput']:.2f}x decode, "
            f"ppl {100 * (best['metrics']['perplexity'] / stock['perplexity'] - 1):+.2f}%"
        )
    backend = MLXMetalBackend()
    promoted_paths: dict[str, str] = {}
    for op in kernel_ops or []:
        winner = _promoted_kernel(op, hardware)
        if not winner:
            mem.log_event(run_id, "kernel_skip", target=target, op=op, why="no promoted winner in memory")
            continue
        path = RUNS / run_id / "promoted" / f"{op}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(winner["candidate"]["source"])
        g = backend.evaluate(op, path, "bfloat16", mode="gate").get("correctness") or {}
        mem.log_event(run_id, "gate1", target=target, label=f"{op}:promoted", status=g.get("status"), dtype="bfloat16")
        if g.get("status") == "pass":
            promoted_paths[op] = str(path)
    if promoted_paths:
        def factory(m, ops):
            return MLXModulePatcher(model_type(m), [{"op": o, "candidate": promoted_paths[o]} for o in ops if o in promoted_paths])

        cfg = dict(best["config"]) | {"kernels": list(promoted_paths)}
        base_metrics = best["metrics"]
        kgate = GateConfig(rel=0.01, quality_budget_rel=quality_budget_rel)
        try:
            m = evaluate_config(model_id, cfg, patcher_factory=factory)
            cand = {"label": f"{best['label']}+kernels", "cls": "fusion", "config": cfg, "correctness": "pass", **m}
            cand["confirmed"] = False
            if faster(base_metrics, cand, kgate) and quality_ok(stock, cand, gate):
                again_base = remeasure_speed(model_id, best["config"])
                again_cand = remeasure_speed(model_id, cfg, patcher_factory=factory)
                cand["confirmed"] = again_cand > again_base * 1.01
            ktrial = mem.record_trial(run_id, target, base_metrics, cand, kgate,
                                      meta={"slot": budget + 1, "op": "model_kernels", "backend": "mlx",
                                            "hardware": hardware, "model": model_id})
            trials.append(ktrial)
            if ktrial["genuine"]:
                optimizations.append(
                    f"kernels {list(promoted_paths)}: {cand['throughput'] / base_metrics['throughput']:.3f}x on top of config")
                best = {"config": cfg, "metrics": cand, "label": cand["label"]}
            else:
                optimizations.append(f"kernels {list(promoted_paths)}: tried in-model, reverted ({ktrial['miss']['why']})")
        except Exception as exc:  # noqa: BLE001
            mem.log_event(run_id, "kernel_error", target=target, error=f"{type(exc).__name__}: {exc}"[:400])
            optimizations.append(f"kernels {list(promoted_paths)}: integration failed ({type(exc).__name__})")

    final_full = None
    if full_eval and best["label"] != "stock":
        from visai.verify.task_eval import run_eval_inprocess

        quant = {"bits": best["config"].get("bits"), "group_size": best["config"].get("group_size", 64),
                 "keep_high": best["config"].get("keep_high")}
        m2, t2 = load(model_id, quant if best["config"].get("bits") else None)
        final_full = run_eval_inprocess(m2, t2, n=10, gen_kwargs=gen_kwargs(best["config"]))["metrics"]
        del m2, t2
        free()

    bm = best["metrics"]
    reg_pct = 100 * (bm.get("perplexity", stock["perplexity"]) - stock["perplexity"]) / stock["perplexity"]
    skills_after = {s["name"] for s in list_skills(500)}
    rep = {
        "run_id": run_id, "model": model_id, "hardware": hardware, "workload": workload,
        "baseline": {"decode_tok_s": stock["throughput"], "perplexity": stock["perplexity"],
                     "quality_label": f"ppl {stock['perplexity']:.3f}", "gsm8k": baseline_full},
        "optimized": {"decode_tok_s": bm["throughput"], "perplexity": bm.get("perplexity"), "config": best["config"],
                      "label": best["label"], "quality_label": f"ppl {bm.get('perplexity', 0):.3f}", "gsm8k": final_full},
        "improvement_pct": 100 * (bm["throughput"] / stock["throughput"] - 1),
        "quality_regression_pct": max(reg_pct, 0.0),
        "quality_budget_pct": 100 * quality_budget_rel,
        "quality_pass": reg_pct <= 100 * quality_budget_rel + 1e-9,
        "optimizations": optimizations or ["no change beat the gate; stock kept"],
        "search": {"experiments": len(trials), "skills_retrieved": search["skills_retrieved"],
                   "blocked": get_db().events.count_documents({"run_id": run_id, "step": "blocked_idea"}),
                   "skills_learned": len(skills_after - skills_before)},
        "profile_families": [(f["name"], f["gpu_pct"]) for f in brief["families"]],
        "trials": [{"label": t["candidate"].get("label"), "config": t["candidate"].get("config"),
                    "tok_s": t["candidate"].get("throughput"), "ppl": t["candidate"].get("perplexity"),
                    "outcome": t["miss"]["outcome"], "keep": t["genuine"]} for t in trials],
        "elapsed_s": round(time.time() - t_start, 1),
    }
    mem.finish_run(run_id, report={k: v for k, v in rep.items() if k != "trials"})
    return rep
