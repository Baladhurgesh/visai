"""Harness-driven diarization sweep (no LLM): post-processing + precision, gated on DER and speed.

Used when the agent budget is unavailable. Every config is recorded in Atlas like an agent trial, so the
next agent run starts from these results (skills, lessons, do-not-repeat).
"""

from __future__ import annotations

import itertools
import json
from typing import Any

from visai.jsonio import utc_now, utc_stamp
from visai.memory import store as mem
from visai.memory.db import get_db
from visai.models.adapters import get_adapter
from visai.skills.update import upsert_skill_from_trial


def sweep(thresholds=(0.5, 0.4, 0.35, 0.3), merge_gaps=(0.0, 0.3, 0.6), dtypes=(None,), max_slowdown=0.03) -> dict[str, Any]:
    a = get_adapter("diar")
    run_id = f"{utc_stamp()}_sweep_diar"
    hw = "M3Pro"
    target = f"{a.model_id}:e2e@mlx:{hw}"
    mem.start_run(run_id, kind="pipeline", adapter="diar", model=a.model_id, target=target, hardware=hw,
                  note="harness sweep (agent budget exhausted): threshold x merge_gap x dtype")
    stock = {"label": "stock", **a.evaluate({})}
    mem.log_event(run_id, "baseline", target=target, stock=a.describe(stock))
    rows, best = [], {"label": "stock", "config": {}, "metrics": stock}
    for i, (th, mg, dt) in enumerate(itertools.product(thresholds, merge_gaps, dtypes), start=1):
        cfg = {"threshold": th, "merge_gap": mg} | ({"dtype": dt} if dt else {})
        if cfg == {"threshold": 0.5, "merge_gap": 0.0}:
            continue
        m = a.evaluate(cfg)
        label = f"diar_th{int(th * 100)}_gap{int(mg * 10)}" + (f"_{dt}" if dt else "")
        better_q = m["der"] < best["metrics"]["der"] - 1e-4
        speed_ok = m["throughput"] >= stock["throughput"] * (1 - max_slowdown)
        keep = better_q and speed_ok
        outcome = "win" if keep else ("slower" if not speed_ok else "no_quality_gain")
        why = f"DER {m['der']:.4f} vs best {best['metrics']['der']:.4f}; speed {m['throughput'] / stock['throughput']:.3f}x stock"
        trial = {
            "ts": utc_now(), "run_id": run_id, "target": target, "slot": i, "op": "diar_model_level", "stage": "model_level",
            "backend": "mlx", "hardware": hw, "genuine": keep, "keep": keep,
            "stock": a.describe(stock) | {"quality": stock["quality"]},
            "candidate": {"label": label, "cls": "postprocessing", "config": cfg, "correctness": "skipped",
                          "throughput": m["throughput"], "der": m["der"], "quality": m["quality"],
                          "speedup_vs_best": m["throughput"] / best["metrics"]["throughput"], "confirmed": keep},
            "miss": {"outcome": outcome, "cls": "quality" if outcome != "win" else "win", "why": why},
        }
        get_db().experiments.insert_one(trial)
        mem.log_event(run_id, "deploy_eval", target=target, label=label, config=cfg)
        mem.log_event(run_id, "benchmark", target=target, label=label, speedup=round(m["throughput"] / stock["throughput"], 4),
                      quality=m["der"], confirmed=keep)
        rows.append({"label": label, "config": cfg, "der": m["der"], "rtfx": m["throughput"], "keep": keep})
        if keep:
            best = {"label": label, "config": cfg, "metrics": m}
            mem.log_event(run_id, "model_level_best", target=target, label=label, e2e_vs_stock=m["throughput"] / stock["throughput"])
        upsert_skill_from_trial(trial, operator="diar_model_level", backend="mlx", hardware=hw, dtype=str(dt or "-"),
                                shape={"target": target, "config": cfg},
                                reflection_skill={"name": f"diar_postproc_threshold_{int(th * 100)}_gap_{int(mg * 10)}",
                                                  "observation": "quality_bound: DER dominated by missed speech",
                                                  "preconditions": {"model": a.model_id, "data": "AMI ihm, 10-min crops"},
                                                  "strategy": [f"threshold={th}", f"merge_gap={mg}s"]},
                                source=run_id)
    mem.add_lesson(
        f"Nemotron-3-Diarization on AMI: best post-processing {best['label']} {json.dumps(best['config'])} -> DER "
        f"{best['metrics']['der']:.4f} vs stock {stock['der']:.4f}; speed unchanged (post-processing only).",
        scope=target, source=run_id)
    return {"run_id": run_id, "stock": a.describe(stock), "best": {"label": best["label"], "config": best["config"],
            "metrics": a.describe(best["metrics"])}, "rows": rows}
