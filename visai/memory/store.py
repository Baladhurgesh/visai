"""Project memory for the optimization loop (port of kernel-forge lib/memory_store.py).

What changed vs kernel-forge:
- trials.jsonl / do_not_repeat.json / latest_helper.json live in Atlas collections
  (experiments, do_not_repeat, reflections) instead of agent/memory/*.
- idea_allowed() keeps the token-fingerprint rule and adds vector similarity.
- The genuine-win gate is generic: throughput can be tok/s (model) or 1/latency (kernel),
  and quality is any higher-is-better score with an absolute or relative budget.
LESSONS.md and the Hermes MEMORY.md snapshot are still rendered for agent compatibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visai.config import OUT
from visai.jsonio import utc_now
from visai.llm.openrouter import embed_one
from visai.memory.db import get_db, strip_ids

LESSONS_PATH = OUT / "LESSONS.md"
HERMES_MEMORY = Path.home() / ".hermes" / "memories" / "MEMORY.md"
HERMES_LIMIT = 2200
SIMILARITY_BLOCK = 0.9

STANDING = """# Visai optimization lessons

## Standing facts
- Profile first. Vendor GEMM / matmul is usually already optimal: skip it unless it is the only target.
- Apply Amdahl: a 2x win on a 5% op is ~2.5% end to end. Prefer fusion and fewer launches on memory-bound decode.
- Microbench wins often die end to end (launch tax, graph capture). Only the gated e2e number counts.
- Numerics must match the trusted reference on seeded AND hidden inputs before anything is timed.

## Path to a real win
- Launch-config-only retunes (block size / warps / threadgroup size / stages) are not a win path on their own.
- Fuse adjacent memory-bound ops, cut kernel launches, vectorize loads, use SIMD-group reductions.
- Confirm a suspected win with a second measurement. Keep only if both beat the gate.

## Gate
- Genuine: correctness pass, n_errors=0, throughput >= stock +3% (or +abs threshold), quality drop within budget,
  confirmed on a repeat run.
- Else revert, reason the miss, append do-not-repeat fingerprints, then try a *different class* of idea.
- Session budget: 10 candidates per target. Stop early only on a confirmed genuine win.
"""


@dataclass
class GateConfig:
    rel: float = 0.03
    abs: float | None = None
    quality_budget_abs: float | None = None
    quality_budget_rel: float | None = None
    require_confirm: bool = True

    def describe(self) -> str:
        speed = f">= +{self.rel:.0%}" + (f" or +{self.abs:g}" if self.abs else "")
        if self.quality_budget_abs is not None:
            q = f"quality drop <= {self.quality_budget_abs:g}"
        elif self.quality_budget_rel is not None:
            q = f"quality drop <= {self.quality_budget_rel:.2%}"
        else:
            q = "no task-quality gate"
        return f"{speed}, {q}" + (", confirmed twice" if self.require_confirm else "")


# ------------------------------------------------------------------ gate


def _f(d: dict | None, key: str) -> float | None:
    if not d or d.get(key) is None:
        return None
    try:
        return float(d[key])
    except (TypeError, ValueError):
        return None


def quality_ok(stock: dict | None, cand: dict | None, gate: GateConfig) -> bool:
    sq, cq = _f(stock, "quality"), _f(cand, "quality")
    if sq is None or cq is None:
        return True
    if gate.quality_budget_abs is not None:
        return cq + 1e-9 >= sq - gate.quality_budget_abs
    if gate.quality_budget_rel is not None:
        return cq + 1e-9 >= sq - abs(sq) * gate.quality_budget_rel
    return True


def faster(stock: dict | None, cand: dict | None, gate: GateConfig) -> bool:
    s, c = _f(stock, "throughput") or 0.0, _f(cand, "throughput") or 0.0
    if s <= 0:
        return False
    return c >= s * (1 + gate.rel) or (gate.abs is not None and c - s >= gate.abs)


def is_genuine(stock: dict | None, cand: dict | None, gate: GateConfig | None = None) -> bool:
    gate = gate or GateConfig()
    if not stock or not cand:
        return False
    if cand.get("n_errors"):
        return False
    if cand.get("correctness") not in (None, "pass", "skipped"):
        return False
    if gate.require_confirm and not cand.get("confirmed"):
        return False
    return faster(stock, cand, gate) and quality_ok(stock, cand, gate)


def genuine_note(stock: dict | None, cand: dict | None, gate: GateConfig | None = None) -> str:
    gate = gate or GateConfig()
    if not stock or not cand:
        return "missing stock/candidate metrics"
    s, c = _f(stock, "throughput") or 0.0, _f(cand, "throughput") or 0.0
    pct = ((c - s) / s * 100) if s else 0.0
    win = is_genuine(stock, cand, gate)
    unit = cand.get("unit") or stock.get("unit") or ""
    return (
        f"{'WIN' if win else 'NO-WIN'}: {c:.3f} vs stock {s:.3f} {unit} ({pct:+.2f}%) "
        f"quality {cand.get('quality')} vs {stock.get('quality')} (need {gate.describe()})"
    )


LAUNCH_CFG_RE = re.compile(
    r"\bbv\s*=?\s*\d+|\bwarps?\s*=?\s*\d+|\bstages?\s*=?\s*\d+|block_size\s*=?\s*\d+|"
    r"threadgroup\s*(size)?\s*=?\s*\d+|num_threads\s*=?\s*\d+|launch_config",
    re.I,
)


def analyze_miss(
    stock: dict | None,
    cand: dict | None,
    gate: GateConfig | None = None,
    *,
    reason: str = "",
    idea_cls: str = "",
) -> dict[str, Any]:
    """Deterministic why-it-missed. The agent must not retry the resulting fingerprints."""
    gate = gate or GateConfig()
    label = str((cand or {}).get("label") or "unknown")
    if is_genuine(stock, cand, gate):
        return {"outcome": "win", "cls": "win", "why": "hit the genuine-win gate", "do_not_repeat": [], "label": label}

    s, c = _f(stock, "throughput") or 0.0, _f(cand, "throughput") or 0.0
    pct = ((c - s) / s * 100) if s else 0.0
    n_err = int((cand or {}).get("n_errors") or 0)
    corr = (cand or {}).get("correctness")
    is_fast = faster(stock, cand, gate)
    q_ok = quality_ok(stock, cand, gate)

    if corr == "compile_error":
        outcome, cls, why = "compile_error", "broke", f"candidate failed to compile: {reason[:200]}"
    elif n_err or corr not in (None, "pass", "skipped"):
        outcome, cls, why = "broke", "errors", f"errors={n_err} correctness={corr}"
    elif not q_ok and not is_fast:
        outcome, cls, why = "quality_and_noise", "quality", f"quality dropped and speed {pct:+.2f}% is inside noise"
    elif not q_ok:
        outcome, cls, why = "quality_drop", "quality", "quality drop exceeds budget"
    elif pct < -1.0:
        outcome, cls, why = "slower", "regression", f"{c:.3f} < stock {s:.3f} ({pct:+.2f}%)"
    elif is_fast and not (cand or {}).get("confirmed"):
        outcome, cls, why = "unconfirmed", "noise", f"{pct:+.2f}% did not hold on the confirmation run"
    else:
        outcome, cls, why = (
            "noise",
            "launch_tax",
            f"{pct:+.2f}% is below the gate ({gate.describe()}); kernel-level gain did not survive",
        )

    dnr = [f"candidate matching {label}"]
    blob = f"{label} {reason} {idea_cls}".lower()
    if idea_cls == "launch_config_tune" or LAUNCH_CFG_RE.search(blob):
        cls = "launch_config_tune"
        dnr.append("only changing block size / warps / threadgroup size / stages as the applied kernel")
    if outcome.startswith("quality"):
        dnr.append(f"{label}: shipping a change that breaks task quality without a real speed win")
    return {
        "outcome": outcome,
        "cls": cls,
        "why": why,
        "do_not_repeat": dnr,
        "label": label,
        "delta_pct": round(pct, 3),
    }


# ------------------------------------------------------------------ do-not-repeat


def load_do_not_repeat(scope: str | None = None) -> list[str]:
    flt = {"$or": [{"scope": scope}, {"scope": "global"}]} if scope else None
    return [r["text"] for r in get_db().do_not_repeat.find(flt, sort=[("ts", 1)])]


def add_do_not_repeat(items: list[str], scope: str = "global", source: str = "") -> list[str]:
    db = get_db()
    have = set(load_do_not_repeat(scope))
    for raw in items:
        text = str(raw).strip().lstrip("- ").strip()
        if not text or text in have:
            continue
        db.do_not_repeat.update_one(
            {"text": text, "scope": scope},
            {"$setOnInsert": {"ts": utc_now(), "source": source, "embedding": embed_one(text)}},
            upsert=True,
        )
        have.add(text)
    return load_do_not_repeat(scope)


def idea_allowed(idea: str, scope: str | None = None) -> dict[str, Any]:
    """Reject a next-try that overlaps a persistent avoid fingerprint (token rule + vector similarity)."""
    blob = (idea or "").lower()
    hits: list[str] = []
    for x in load_do_not_repeat(scope):
        key = x.lower()
        if key in blob or (len(blob) > 12 and blob in key):
            hits.append(x)
            continue
        tokens = [t for t in re.split(r"[^a-z0-9]+", key) if len(t) >= 5]
        if tokens and all(t in blob for t in tokens[:5]):
            hits.append(x)
    similar: list[dict[str, Any]] = []
    if blob.strip():
        flt = {"scope": {"$in": [scope, "global"]}} if scope else None
        for r in get_db().vector_search("do_not_repeat", embed_one(idea), k=3, flt=flt):
            if r.get("score", 0) >= SIMILARITY_BLOCK:
                similar.append({"text": r["text"], "score": round(r["score"], 3)})
                hits.append(r["text"])
    return {"allowed": not hits, "blocked_by": list(dict.fromkeys(hits)), "similar": similar, "idea": idea}


# ------------------------------------------------------------------ events / runs


def log_event(run_id: str, step: str, **data: Any) -> None:
    get_db().events.insert_one({"run_id": run_id, "ts": utc_now(), "step": step, **data})


def start_run(run_id: str, **meta: Any) -> None:
    get_db().runs.update_one(
        {"run_id": run_id},
        {"$set": {"status": "running", **meta}, "$setOnInsert": {"started": utc_now()}},
        upsert=True,
    )


def finish_run(run_id: str, **meta: Any) -> None:
    get_db().runs.update_one({"run_id": run_id}, {"$set": {"status": "done", "finished": utc_now(), **meta}})


# ------------------------------------------------------------------ trials


def record_trial(
    run_id: str,
    target: str,
    stock: dict[str, Any],
    cand: dict[str, Any],
    gate: GateConfig,
    *,
    meta: dict[str, Any] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    meta = meta or {}
    miss = analyze_miss(stock, cand, gate, reason=reason, idea_cls=str(cand.get("cls") or ""))
    genuine = is_genuine(stock, cand, gate)
    if not genuine:
        add_do_not_repeat(miss.get("do_not_repeat") or [], scope=target, source=f"miss:{cand.get('label')}")
    row = {
        "ts": utc_now(),
        "run_id": run_id,
        "target": target,
        "stock": stock,
        "candidate": cand,
        "genuine": genuine,
        "keep": genuine,
        "gate": genuine_note(stock, cand, gate),
        "miss": miss,
        "reason": reason,
        **meta,
    }
    row["_id"] = get_db().experiments.insert_one(row)
    log_event(run_id, "trial", target=target, label=cand.get("label"), genuine=genuine, outcome=miss["outcome"])
    return row


def recent_trials(target: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
    flt = {"target": target} if target else None
    rows = get_db().experiments.find(flt, sort=[("ts", -1)], limit=limit)
    return strip_ids(reversed(rows))


# ------------------------------------------------------------------ reflections / lessons


def save_reflection(run_id: str, target: str, reflection: dict[str, Any]) -> None:
    db = get_db()
    db.reflections.insert_one({"run_id": run_id, "target": target, "ts": utc_now(), **reflection})
    add_do_not_repeat(list(reflection.get("do_not_repeat") or []), scope=target, source="reflection")
    for line in reflection.get("lesson_lines") or []:
        add_lesson(str(line), scope=target, source="reflection")


def latest_reflection(target: str | None = None) -> dict[str, Any] | None:
    flt = {"target": target} if target else None
    r = get_db().reflections.find_one(flt, sort=[("ts", -1)])
    return strip_ids([r])[0] if r else None


def add_lesson(text: str, scope: str = "global", source: str = "", kind: str = "line") -> None:
    text = text.strip().lstrip("- ").strip()
    if not text:
        return
    db = get_db()
    if db.lessons.find_one({"text": text, "scope": scope, "kind": kind}):
        return
    db.lessons.insert_one(
        {"text": text, "scope": scope, "kind": kind, "source": source, "ts": utc_now(), "embedding": embed_one(text)}
    )


def _tried_line(row: dict[str, Any]) -> str:
    c = row.get("candidate") or {}
    keep = "keep" if row.get("genuine") else "revert"
    thr = c.get("throughput")
    thr_s = f"{float(thr):.3f}" if thr is not None else "?"
    return f"- {row.get('ts')} | {row.get('target')} | {c.get('label')} | {thr_s} {c.get('unit') or ''} | {keep} | {row.get('gate')}"


def refresh_lessons(target: str | None = None) -> str:
    """Rebuild LESSONS.md from Atlas (experiments + do_not_repeat + reflections + lessons)."""
    db = get_db()
    flt = {"target": target} if target else None
    rows = db.experiments.find(flt, sort=[("ts", 1)])
    tried = [_tried_line(r) for r in rows[-20:]] or ["- (none yet)"]
    misses = [
        f"- {r.get('candidate', {}).get('label')}: {r.get('miss', {}).get('cls')} - {r.get('miss', {}).get('why')}"
        for r in rows
        if not r.get("genuine")
    ][-12:] or ["- (none yet)"]
    avoid = [f"- {x}" for x in load_do_not_repeat(target)] or ["- (none yet)"]
    lflt: dict[str, Any] = {"kind": "line"}
    if target:
        lflt["scope"] = {"$in": [target, "global"]}
    lessons = [f"- {r['text']}" for r in db.lessons.find(lflt, sort=[("ts", 1)])][-20:] or ["- (none yet)"]
    text = (
        STANDING
        + "\n## Tried\n"
        + "\n".join(tried)
        + "\n\n## Miss reasons (must change class next time)\n"
        + "\n".join(misses)
        + "\n\n## Avoid (do not retry - persistent)\n"
        + "\n".join(avoid)
        + "\n\n## Learned lessons\n"
        + "\n".join(lessons)
        + "\n"
    )
    LESSONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LESSONS_PATH.write_text(text)
    db.lessons.update_one(
        {"kind": "lessons_md", "scope": target or "global"},
        {"$set": {"text_md": text, "ts": utc_now()}},
        upsert=True,
    )
    return text


def hermes_snapshot() -> str:
    last = recent_trials(limit=4)
    lines = [
        "Visai loop: this repo. 10 candidates per target unless a confirmed genuine win. "
        "Real win = correctness pass on hidden inputs, >=3% faster twice, quality within budget.",
        "Read out/LESSONS.md Miss reasons + Avoid before writing. After a miss: record, reflect, "
        "then a different idea class. Never retry do_not_repeat.",
    ]
    if last:
        bits = [f"{(r.get('candidate') or {}).get('label')}: genuine={r.get('genuine')}" for r in last]
        lines.append("Recent: " + " | ".join(bits))
    digest = " ".join((LESSONS_PATH.read_text() if LESSONS_PATH.exists() else "").split())
    if digest:
        lines.append("Lessons: " + digest[:800])
    text = "\n§\n".join(lines)
    return text[: HERMES_LIMIT - 1]


def sync_hermes() -> dict[str, Any]:
    refresh_lessons()
    text = hermes_snapshot()
    HERMES_MEMORY.parent.mkdir(parents=True, exist_ok=True)
    HERMES_MEMORY.write_text(text + "\n")
    return {"path": str(HERMES_MEMORY), "chars": len(text), "limit": HERMES_LIMIT}


def show(target: str | None = None) -> dict[str, Any]:
    return {
        "backend": get_db().backend,
        "lessons": refresh_lessons(target),
        "n_trials": get_db().experiments.count_documents({"target": target} if target else None),
        "trials": recent_trials(target, limit=8),
        "latest_reflection": latest_reflection(target),
        "do_not_repeat": load_do_not_repeat(target),
    }
