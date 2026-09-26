"""Turn trials into reusable skills (the lesson, not the file).

A skill is keyed by (name, backend, hardware). Evidence accumulates across runs and
confidence is the Beta(1,1) posterior mean of "this strategy produced a gated speedup".
"""

from __future__ import annotations

import re
from typing import Any

from visai.jsonio import utc_now
from visai.llm.openrouter import embed_one
from visai.memory.db import get_db, strip_ids


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "unnamed"


def skill_text(s: dict[str, Any]) -> str:
    return (
        f"skill {s.get('name')} operator {s.get('operator')} backend {s.get('backend')} "
        f"hardware {s.get('hardware')} dtype {s.get('dtype')} observation {s.get('observation')} "
        f"strategy {' '.join(s.get('strategy') or [])} preconditions {s.get('preconditions')}"
    )


def upsert_skill_from_trial(
    trial: dict[str, Any],
    *,
    operator: str,
    backend: str,
    hardware: str,
    dtype: str,
    shape: Any = None,
    reflection_skill: dict[str, Any] | None = None,
    source: str = "",
) -> dict[str, Any]:
    cand = trial.get("candidate") or {}
    stock = trial.get("stock") or {}
    rs = reflection_skill or {}
    name = _slug(rs.get("name") or f"{operator}_{cand.get('cls') or 'other'}")
    speedup = None
    if stock.get("throughput") and cand.get("throughput"):
        speedup = float(cand["throughput"]) / float(stock["throughput"])
    passed = cand.get("correctness") in ("pass", "skipped")
    positive = bool(trial.get("genuine"))

    db = get_db()
    key = {"name": name, "backend": backend, "hardware": hardware}
    cur = db.skills.find_one(key) or {}
    ev = dict(cur.get("evidence") or {})
    trials = int(ev.get("trials", 0)) + 1
    passes = int(ev.get("correctness_passes", 0)) + (1 if passed else 0)
    pos = int(ev.get("positive_speedups", 0)) + (1 if positive else 0)
    sp_list = list(ev.get("speedups") or [])
    if speedup is not None and passed:
        sp_list.append(round(speedup, 4))
    mean_sp = sum(sp_list) / len(sp_list) if sp_list else 0.0
    neg = dict(cur.get("negative_evidence") or {})
    if not positive:
        miss = trial.get("miss") or {}
        neg_key = f"{cand.get('label')}"
        neg[neg_key] = {"effect": miss.get("outcome"), "why": miss.get("why"), "shape": shape}

    doc = {
        "operator": operator,
        "dtype": dtype,
        "observation": rs.get("observation") or cur.get("observation") or "",
        "preconditions": rs.get("preconditions") or cur.get("preconditions") or ({"shape": shape} if shape else {}),
        "strategy": rs.get("strategy") or cur.get("strategy") or [cand.get("cls") or "other"],
        "evidence": {
            "trials": trials,
            "correctness_passes": passes,
            "positive_speedups": pos,
            "mean_speedup": round(mean_sp, 4),
            "best_speedup": round(max(sp_list), 4) if sp_list else 0.0,
            "speedups": sp_list[-50:],
        },
        "negative_evidence": neg,
        "confidence": round((pos + 1) / (trials + 2), 4),
        "updated": utc_now(),
        "last_source": source,
    }
    if positive and cand.get("source"):
        doc["example_source"] = cand["source"][:12000]
        doc["example_label"] = cand.get("label")
    doc["embedding"] = embed_one(skill_text({**key, **doc}))
    db.skills.update_one(key, {"$set": doc, "$setOnInsert": {"created": utc_now()}}, upsert=True)
    return {**key, **{k: v for k, v in doc.items() if k != "embedding"}}


def retrieve_skills(
    query: str,
    *,
    operator: str | None = None,
    backend: str | None = None,
    hardware: str | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    flt: dict[str, Any] = {}
    if backend:
        flt["backend"] = backend
    rows = get_db().vector_search("skills", embed_one(query), k=k * 2, flt=flt or None)
    # rank: same operator first, then same hardware, then similarity * confidence
    def rank(r: dict[str, Any]) -> float:
        score = float(r.get("score", 0)) * (0.5 + float(r.get("confidence", 0.5)))
        if operator and r.get("operator") == operator:
            score += 1.0
        if hardware and r.get("hardware") == hardware:
            score += 0.25
        return score

    rows.sort(key=rank, reverse=True)
    return strip_ids(rows[:k])


def list_skills(limit: int = 100) -> list[dict[str, Any]]:
    return strip_ids(get_db().skills.find(None, sort=[("confidence", -1)], limit=limit))
