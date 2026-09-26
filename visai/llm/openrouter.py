"""Thin OpenRouter helpers for non-agent calls (embeddings, one-off chat).

Agent calls go through Strands' OpenAIModel (see visai.runtime.models).
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Any

from openai import OpenAI

from visai.config import settings


@lru_cache(maxsize=1)
def client() -> OpenAI:
    s = settings()
    if not s.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env)")
    return OpenAI(
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base,
        default_headers={"HTTP-Referer": "https://localhost/visai", "X-Title": "visai"},
    )


def chat(messages: list[dict[str, Any]], model: str | None = None, **kwargs: Any) -> dict[str, Any]:
    s = settings()
    resp = client().chat.completions.create(model=model or s.model, messages=messages, **kwargs)
    choice = resp.choices[0]
    usage = resp.usage.model_dump() if resp.usage else {}
    return {"text": choice.message.content or "", "model": resp.model, "usage": usage}


def _hash_embed(text: str, dim: int) -> list[float]:
    """Deterministic offline fallback so memory still works without the embeddings API."""
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9_]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed(texts: list[str]) -> list[list[float]]:
    s = settings()
    texts = [t if t.strip() else "(empty)" for t in texts]
    try:
        resp = client().embeddings.create(model=s.embed_model, input=texts)
        out = [d.embedding for d in resp.data]
        if out and len(out[0]) == s.embed_dim:
            return out
    except Exception:  # noqa: BLE001
        pass
    return [_hash_embed(t, s.embed_dim) for t in texts]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
