"""Case study 2: on-device meeting pipeline (Parakeet ASR + Nemotron-3 diarization) on Apple Silicon.

Each model gets its own gated deployment search (shared agent loop, same memory). The pipeline report
combines them: meeting RTFx = 1 / (1/asr_rtfx + 1/diar_rtfx) when the stages run back to back.
"""

from __future__ import annotations

import json
import time
from typing import Any

from visai.deploy.search_loop import SearchSpec, run_config_search
from visai.jsonio import utc_stamp
from visai.memory import store as mem
from visai.memory.store import GateConfig
from visai.profiler.hardware import probe
from visai.speech import asr, diar

SPEECH_SYSTEM = """You are Visai's deployment optimizer for on-device speech models (MLX on Apple Silicon).
Find the deployment config with the highest real-time factor whose task quality stays within the budget.

Per candidate slot:
1. Read the context (stock metrics, retrieved skills, lessons, do-not-repeat, last reflection).
2. Pick ONE config idea of a different class than the last miss. Use check_idea if unsure.
3. Call evaluate_deploy_config once (retry once only if the config was invalid).
4. Return the structured proposal: label, cls (quantization | precision | scheduling | other), hypothesis,
   config, skills_used, experiments for next time.
"""


def pipeline_rtfx(a: float, d: float) -> float:
    return 1.0 / (1.0 / a + 1.0 / d)


def _search(run_id: str, key: str, hw: dict, budget: int, budget_abs: float) -> dict[str, Any]:
    mod = asr if key == "asr" else diar
    model_id = asr.PARAKEET_ID if key == "asr" else diar.DIAR_ID
    qname = "WER" if key == "asr" else "DER"
    target = f"{model_id}:meeting@mlx:{hw.get('chip')}"
    stock = {"label": "stock", **mod.evaluate({})}
    mem.log_event(run_id, "baseline", target=target, stock=mod.describe(stock))
    spec = SearchSpec(
        target=target, operator=f"{key}_deploy", hardware=hw.get("chip", "apple-silicon"),
        system_prompt=SPEECH_SYSTEM, search_space_doc=mod.SEARCH_SPACE_DOC, quality_name=qname,
        context=f"MODEL: {model_id}\nHARDWARE: {json.dumps(hw)}\nWORKLOAD: on-device meeting transcription + diarization",
        evaluate=mod.evaluate, validate=mod.validate, speed_of=mod.speed_of, describe_metrics=mod.describe,
    )
    gate = GateConfig(rel=0.03, quality_budget_abs=budget_abs)
    res = run_config_search(run_id, spec, stock, gate, budget)
    return {"target": target, "stock": stock, **res}


def optimize_speech(targets: list[str], budget: int = 4, wer_budget_abs: float = 0.003, der_budget_abs: float = 0.005) -> dict:
    hw = probe()
    run_id = f"{utc_stamp()}_speech"
    mem.start_run(run_id, kind="model", target=f"meeting-pipeline@mlx:{hw.get('chip')}", model="parakeet+nemotron-diar",
                  hardware=hw.get("chip"), budget=budget)
    t0 = time.time()
    out: dict[str, Any] = {"run_id": run_id, "hardware": hw.get("chip")}
    for key in targets:
        r = _search(run_id, key, hw, budget, wer_budget_abs if key == "asr" else der_budget_abs)
        mod = asr if key == "asr" else diar
        best = r["best"]
        out[key] = {
            "target": r["target"],
            "stock": mod.describe(r["stock"]),
            "best_label": best["label"],
            "best_config": best["config"],
            "best": mod.describe(best["metrics"]),
            "speedup": best["metrics"]["throughput"] / r["stock"]["throughput"],
            "trials": [{"label": t["candidate"].get("label"), "config": t["candidate"].get("config"),
                        "rtfx": t["candidate"].get("throughput"), "quality": t["candidate"].get("wer" if key == "asr" else "der"),
                        "outcome": t["miss"]["outcome"], "keep": t["genuine"]} for t in r["trials"]],
            "skills_retrieved": r["skills_retrieved"],
        }
    if "asr" in out and "diar" in out:
        b = pipeline_rtfx(out["asr"]["stock"]["throughput"], out["diar"]["stock"]["throughput"])
        o = pipeline_rtfx(out["asr"]["best"]["throughput"], out["diar"]["best"]["throughput"])
        out["pipeline"] = {"stock_rtfx": b, "optimized_rtfx": o, "speedup": o / b,
                           "minutes_per_hour_of_meeting_stock": 60 / b, "minutes_per_hour_of_meeting_optimized": 60 / o}
    out["elapsed_s"] = round(time.time() - t0, 1)
    mem.finish_run(run_id, report={k: v for k, v in out.items() if k not in ("asr", "diar")} | {
        k: {kk: vv for kk, vv in out[k].items() if kk != "trials"} for k in ("asr", "diar") if k in out})
    return out
