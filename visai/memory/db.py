"""Document store: MongoDB Atlas when MONGODB_URI is set, local JSONL fallback otherwise.

Both expose the same small collection API (insert_one, find, find_one, update_one,
count_documents, delete_many) plus vector_search(), so the rest of Visai never
branches on the backend.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from visai.config import OUT, settings

VECTOR_COLLECTIONS: dict[str, list[str]] = {
    "skills": ["operator", "backend", "hardware", "dtype"],
    "do_not_repeat": ["scope"],
    "lessons": ["scope", "kind"],
}

COLLECTIONS = [
    "runs",
    "experiments",
    "events",
    "do_not_repeat",
    "reflections",
    "skills",
    "lessons",
    "profiles",
    "kernels",
    "sessions",
    "session_agents",
    "session_messages",
    "comparisons",
]


# ------------------------------------------------------------------ matching


def _get_path(doc: dict[str, Any], path: str) -> Any:
    cur: Any = doc
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _match_value(val: Any, cond: Any) -> bool:
    if isinstance(cond, dict) and any(k.startswith("$") for k in cond):
        for op, arg in cond.items():
            if op == "$in" and val not in arg:
                return False
            if op == "$nin" and val in arg:
                return False
            if op == "$ne" and val == arg:
                return False
            if op == "$gte" and not (val is not None and val >= arg):
                return False
            if op == "$gt" and not (val is not None and val > arg):
                return False
            if op == "$lte" and not (val is not None and val <= arg):
                return False
            if op == "$lt" and not (val is not None and val < arg):
                return False
            if op == "$exists" and (val is not None) != bool(arg):
                return False
        return True
    if isinstance(val, list) and not isinstance(cond, list):
        return cond in val
    return val == cond


def matches(doc: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$or":
            if not any(matches(doc, sub) for sub in cond):
                return False
            continue
        if key == "$and":
            if not all(matches(doc, sub) for sub in cond):
                return False
            continue
        if not _match_value(_get_path(doc, key), cond):
            return False
    return True


def _set_path(doc: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = doc
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def apply_update(doc: dict[str, Any], update: dict[str, Any], inserting: bool = False) -> None:
    for k, v in (update.get("$set") or {}).items():
        _set_path(doc, k, v)
    if inserting:
        for k, v in (update.get("$setOnInsert") or {}).items():
            _set_path(doc, k, v)
    for k, v in (update.get("$inc") or {}).items():
        _set_path(doc, k, (_get_path(doc, k) or 0) + v)
    for k, v in (update.get("$push") or {}).items():
        cur = _get_path(doc, k) or []
        cur.append(v)
        _set_path(doc, k, cur)
    for k, v in (update.get("$addToSet") or {}).items():
        cur = _get_path(doc, k) or []
        items = v["$each"] if isinstance(v, dict) and "$each" in v else [v]
        for it in items:
            if it not in cur:
                cur.append(it)
        _set_path(doc, k, cur)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# ------------------------------------------------------------------ local backend


class LocalCollection:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def _save(self, rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))

    def insert_one(self, doc: dict[str, Any]) -> str:
        doc = dict(doc)
        doc.setdefault("_id", uuid.uuid4().hex)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(doc, default=str) + "\n")
        return doc["_id"]

    def find(
        self,
        flt: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        limit: int = 0,
        projection: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        rows = [r for r in self._load() if matches(r, flt)]
        for key, direction in reversed(sort or []):
            rows.sort(key=lambda r: (_get_path(r, key) is None, _get_path(r, key)), reverse=direction < 0)
        if limit:
            rows = rows[:limit]
        if projection:
            drop = {k for k, v in projection.items() if not v}
            rows = [{k: v for k, v in r.items() if k not in drop} for r in rows]
        return rows

    def find_one(self, flt: dict[str, Any] | None = None, sort: list[tuple[str, int]] | None = None):
        rows = self.find(flt, sort=sort, limit=1)
        return rows[0] if rows else None

    def update_one(self, flt: dict[str, Any], update: dict[str, Any], upsert: bool = False) -> None:
        with self._lock:
            rows = self._load()
            for r in rows:
                if matches(r, flt):
                    apply_update(r, update)
                    self._save(rows)
                    return
            if upsert:
                doc = {k: v for k, v in flt.items() if not k.startswith("$") and not isinstance(v, dict)}
                doc["_id"] = uuid.uuid4().hex
                apply_update(doc, update, inserting=True)
                rows.append(doc)
                self._save(rows)

    def count_documents(self, flt: dict[str, Any] | None = None) -> int:
        return len(self.find(flt))

    def delete_many(self, flt: dict[str, Any] | None = None) -> int:
        with self._lock:
            rows = self._load()
            keep = [r for r in rows if not matches(r, flt)]
            self._save(keep)
            return len(rows) - len(keep)


# ------------------------------------------------------------------ atlas backend


class AtlasCollection:
    def __init__(self, coll: Any):
        self.coll = coll

    def insert_one(self, doc: dict[str, Any]) -> str:
        doc = dict(doc)
        doc.setdefault("_id", uuid.uuid4().hex)
        self.coll.insert_one(doc)
        return doc["_id"]

    def find(self, flt=None, sort=None, limit=0, projection=None) -> list[dict[str, Any]]:
        cur = self.coll.find(flt or {}, projection)
        if sort:
            cur = cur.sort(sort)
        if limit:
            cur = cur.limit(limit)
        return list(cur)

    def find_one(self, flt=None, sort=None):
        return self.coll.find_one(flt or {}, sort=sort)

    def update_one(self, flt, update, upsert=False) -> None:
        self.coll.update_one(flt, update, upsert=upsert)

    def count_documents(self, flt=None) -> int:
        return self.coll.count_documents(flt or {})

    def delete_many(self, flt=None) -> int:
        return self.coll.delete_many(flt or {}).deleted_count


# ------------------------------------------------------------------ facade


class DB:
    def __init__(self) -> None:
        s = settings()
        self.backend = "atlas" if s.has_atlas else "local"
        self.atlas_error: str | None = None
        self._colls: dict[str, Any] = {}
        self._root = OUT / "localdb"
        if self.backend == "atlas":
            from pymongo import MongoClient

            try:
                self._client = MongoClient(s.mongodb_uri, appname="visai", serverSelectionTimeoutMS=8000)
                self._client.admin.command("ping")
                self._db = self._client[s.db_name]
                self._db.command("dbStats")
            except Exception as exc:  # noqa: BLE001
                self.atlas_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                sys.stderr.write(f"[visai] Atlas unavailable ({self.atlas_error}); using local fallback {self._root}\n")
                self.backend = "local"

    def coll(self, name: str):
        if name not in self._colls:
            if self.backend == "atlas":
                self._colls[name] = AtlasCollection(self._db[name])
            else:
                self._colls[name] = LocalCollection(self._root / f"{name}.jsonl")
        return self._colls[name]

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self.coll(name)

    def ping(self) -> dict[str, Any]:
        if self.backend == "atlas":
            self._client.admin.command("ping")
            return {"backend": "atlas", "db": self._db.name}
        return {"backend": "local", "path": str(self._root), "atlas_error": self.atlas_error}

    def ensure_indexes(self) -> dict[str, Any]:
        """Create regular + Atlas Vector Search indexes (idempotent)."""
        if self.backend != "atlas":
            return {"backend": "local", "indexes": "n/a"}
        from pymongo.operations import SearchIndexModel

        dim = settings().embed_dim
        self._db.experiments.create_index([("run_id", 1), ("ts", 1)])
        self._db.events.create_index([("run_id", 1), ("ts", 1)])
        self._db.skills.create_index([("name", 1), ("backend", 1), ("hardware", 1)], unique=True)
        self._db.do_not_repeat.create_index([("text", 1), ("scope", 1)], unique=True)
        created = []
        existing_colls = set(self._db.list_collection_names())
        for coll, filters in VECTOR_COLLECTIONS.items():
            if coll not in existing_colls:
                self._db.create_collection(coll)
            name = f"{coll}_vec"
            have = {ix["name"] for ix in self._db[coll].list_search_indexes()}
            if name in have:
                continue
            fields = [{"type": "vector", "path": "embedding", "numDimensions": dim, "similarity": "cosine"}]
            fields += [{"type": "filter", "path": f} for f in filters]
            self._db[coll].create_search_index(
                SearchIndexModel(definition={"fields": fields}, name=name, type="vectorSearch")
            )
            created.append(name)
        return {"backend": "atlas", "created_search_indexes": created}

    def vector_search(
        self,
        coll: str,
        vector: list[float],
        k: int = 5,
        flt: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.backend == "atlas":
            stage: dict[str, Any] = {
                "index": f"{coll}_vec",
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": max(50, k * 10),
                "limit": k,
            }
            if flt:
                stage["filter"] = flt
            try:
                rows = list(
                    self._db[coll].aggregate(
                        [
                            {"$vectorSearch": stage},
                            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                            {"$project": {"embedding": 0}},
                        ]
                    )
                )
                return rows
            except Exception:  # noqa: BLE001  index still building -> client-side fallback
                pass
        rows = self.coll(coll).find(flt)
        scored = []
        for r in rows:
            emb = r.get("embedding")
            if not emb:
                continue
            r = {k2: v for k2, v in r.items() if k2 != "embedding"}
            r["score"] = cosine(vector, emb)
            scored.append(r)
        scored.sort(key=lambda r: -r["score"])
        return scored[:k]


@lru_cache(maxsize=1)
def get_db() -> DB:
    return DB()


def strip_ids(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: (str(v) if k == "_id" else v) for k, v in r.items() if k != "embedding"} for r in rows]
