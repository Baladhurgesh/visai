"""Kernel registry in Atlas: winning / integrated kernels, stored once by content hash.

status:
  layer_winner   - became the best candidate for its layer (Gate 1 pass, faster on two benchmark runs)
  integrated     - kept in the whole model (paired e2e A/B faster, quality within budget)
  reverted       - layer winner that made the whole model slower or broke quality; kept for the record
  kernel_winner  - KernelBench op win (genuine gate)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from visai.config import OUT
from visai.jsonio import utc_now
from visai.memory.db import get_db

MATERIALIZED = OUT / "kernels"


def kernel_id(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def register_kernel(source: str, *, status: str, **meta: Any) -> str:
    kid = kernel_id(source)
    db = get_db()
    existing = db.kernels.find_one({"kernel_id": kid})
    history = {"ts": utc_now(), "status": status, **{k: meta.get(k) for k in ("run_id", "speedup", "e2e_paired") if k in meta}}
    if existing:
        upd: dict[str, Any] = {"$push": {"history": history}, "$set": {"updated": utc_now()}}
        # integration outcome supersedes layer_winner; never downgrade integrated -> layer_winner
        if status in ("integrated", "reverted") or existing.get("status") not in ("integrated",):
            upd["$set"]["status"] = status
        for k in ("e2e_paired", "e2e_wins"):
            if k in meta:
                upd["$set"][k] = meta[k]
        if meta.get("speedup") and float(meta["speedup"]) > float(existing.get("speedup") or 0):
            upd["$set"]["speedup"] = meta["speedup"]
        db.kernels.update_one({"kernel_id": kid}, upd)
    else:
        db.kernels.insert_one({"_id": kid, "kernel_id": kid, "source": source, "status": status, "created": utc_now(),
                               "updated": utc_now(), "history": [history], **meta})
    return kid


def mark_integration(path: str, *, kept: bool, e2e_paired: float, wins: int, run_id: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return register_kernel(p.read_text(), status="integrated" if kept else "reverted", e2e_paired=e2e_paired,
                           e2e_wins=wins, run_id=run_id)


def materialize(kid: str) -> str:
    """Write a registry kernel to out/kernels/<id>.py (for configs whose original file is gone)."""
    doc = get_db().kernels.find_one({"kernel_id": kid})
    if not doc:
        raise KeyError(f"kernel {kid} not in registry")
    MATERIALIZED.mkdir(parents=True, exist_ok=True)
    path = MATERIALIZED / f"{kid}.py"
    path.write_text(doc["source"])
    return str(path)


def resolve_patch_path(patch: dict[str, Any]) -> str:
    path = patch.get("candidate")
    if path and Path(path).exists():
        return path
    if patch.get("kernel_id"):
        return materialize(patch["kernel_id"])
    raise FileNotFoundError(f"kernel file {path} missing and no kernel_id to restore it from Atlas")


def list_kernels(model: str | None = None) -> list[dict[str, Any]]:
    flt = {"model": model} if model else None
    rows = get_db().kernels.find(flt, sort=[("updated", -1)])
    return [{k: v for k, v in r.items() if k not in ("source", "_id")} for r in rows]


def backfill() -> dict[str, int]:
    """Register winners already recorded in events/experiments before the registry existed."""
    from visai.config import RUNS

    db = get_db()
    n_layer = n_kb = n_int = 0
    runs = {r["run_id"]: r for r in db.runs.find(None)}
    for e in db.events.find({"step": "benchmark"}, sort=[("ts", 1)]):
        tgt = str(e.get("target") or "")
        if str(e.get("improved")) != "True" and e.get("improved") is not True:
            continue
        if tgt.count(":") < 2 or "@" not in tgt:
            continue
        key, name = tgt.split(":")[0], tgt.split(":")[1].split("@")[0]
        path = RUNS / e["run_id"] / "layers" / name / f"{e.get('label')}.py"
        if not path.exists():
            continue
        run = runs.get(e["run_id"], {})
        register_kernel(path.read_text(), status="layer_winner", kind="layer", adapter=key, model=run.get("model"),
                        layer=name, hardware=tgt.split("@mlx:")[-1], label=e.get("label"), speedup=float(e.get("speedup") or 0),
                        run_id=e["run_id"], target=tgt)
        n_layer += 1
    from visai.tasks.ops import OPS

    for t in db.experiments.find({"genuine": True}):
        c = t.get("candidate") or {}
        if c.get("source") and t.get("op") in OPS and t.get("stage") != "layer":
            register_kernel(c["source"], status="kernel_winner", kind="op", op=t["op"], hardware=t.get("hardware"),
                            dtype=t.get("dtype"), label=c.get("label"), speedup=c.get("speedup"),
                            run_id=t["run_id"], target=t.get("target"))
            n_kb += 1
    for e in db.events.find({"step": "integration"}, sort=[("ts", 1)]):
        run = runs.get(e["run_id"], {})
        rep = (run.get("report") or {})
        for lr in rep.get("layers") or []:
            if lr.get("layer") == e.get("layer") and (lr.get("best") or {}).get("path"):
                kept = e.get("kept") is True or str(e.get("kept")) == "True"
                if mark_integration(lr["best"]["path"], kept=kept, e2e_paired=float(e.get("paired_speedup") or 0),
                                    wins=int(e.get("wins") or 0), run_id=e["run_id"]):
                    n_int += 1
    return {"layer_winners": n_layer, "kernelbench_winners": n_kb, "integration_updates": n_int}
