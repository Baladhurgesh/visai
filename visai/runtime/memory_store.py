"""AtlasMemoryStore: a Strands MemoryStore backed by Atlas Vector Search over skills + lessons."""

from __future__ import annotations

import asyncio
from typing import Any

from strands.memory import MemoryEntry

from visai.llm.openrouter import embed_one
from visai.memory.db import get_db
from visai.memory.store import add_lesson


def _skill_entry(r: dict[str, Any]) -> MemoryEntry:
    ev = r.get("evidence") or {}
    text = (
        f"[skill {r.get('name')}] op={r.get('operator')} backend={r.get('backend')} hw={r.get('hardware')} "
        f"strategy={r.get('strategy')} observation={r.get('observation')} "
        f"preconditions={r.get('preconditions')} evidence: trials={ev.get('trials')} "
        f"wins={ev.get('positive_speedups')} mean_speedup={ev.get('mean_speedup')} "
        f"best={ev.get('best_speedup')} confidence={r.get('confidence')} "
        f"negative={list((r.get('negative_evidence') or {}).items())[:3]}"
    )
    return MemoryEntry(content=text, metadata={"kind": "skill", "name": r.get("name"), "score": r.get("score")})


class AtlasMemoryStore:
    """Strands MemoryStore protocol: name/description/max_search_results/writable/extraction + search/add."""

    def __init__(self, scope: str = "global", backend: str | None = None, name: str = "visai_atlas"):
        self.name = name
        self.description = (
            "Visai optimization memory in MongoDB Atlas: learned skills (strategies with evidence and "
            "confidence) and lessons from past experiments, retrieved by vector search."
        )
        self.max_search_results = 6
        self.writable = True
        self.extraction = None
        self.scope = scope
        self.backend = backend

    def _search_sync(self, query: str, k: int) -> list[MemoryEntry]:
        db = get_db()
        vec = embed_one(query)
        flt = {"backend": self.backend} if self.backend else None
        skills = db.vector_search("skills", vec, k=k, flt=flt)
        lessons = db.vector_search("lessons", vec, k=k, flt={"scope": {"$in": [self.scope, "global"]}, "kind": "line"})
        entries = [_skill_entry(r) for r in skills]
        entries += [
            MemoryEntry(content=f"[lesson] {r['text']}", metadata={"kind": "lesson", "score": r.get("score")})
            for r in lessons
        ]
        entries.sort(key=lambda e: -float((e.metadata or {}).get("score") or 0))
        return entries[:k]

    async def search(self, query: str, options: dict | None = None) -> list[MemoryEntry]:
        k = int((options or {}).get("max_search_results") or self.max_search_results)
        return await asyncio.to_thread(self._search_sync, query, k)

    async def add(self, content: str, metadata: dict | None = None) -> Any:
        await asyncio.to_thread(add_lesson, content, self.scope, "agent:add_memory")
        return {"stored": True}
