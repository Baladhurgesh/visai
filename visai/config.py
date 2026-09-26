"""Settings read from .env. Model ids live only in .env, never in code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
RUNS = ROOT / "runs"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    db_name: str
    openrouter_api_key: str
    openrouter_base: str
    model: str
    reflect_model: str
    embed_model: str
    embed_dim: int
    server_base: str
    remote_executor_url: str

    @property
    def has_atlas(self) -> bool:
        return bool(self.mongodb_uri)


@lru_cache(maxsize=1)
def settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    model = _env("OPENROUTER_MODEL")
    return Settings(
        mongodb_uri=_env("MONGODB_URI"),
        db_name=_env("VISAI_DB", "visai"),
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        openrouter_base=_env("OPENROUTER_BASE", "https://openrouter.ai/api/v1"),
        model=model,
        reflect_model=_env("OPENROUTER_REFLECT_MODEL") or model,
        embed_model=_env("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small"),
        embed_dim=int(_env("VISAI_EMBED_DIM", "1536")),
        server_base=_env("VISAI_SERVER_BASE", "http://127.0.0.1:8080"),
        remote_executor_url=_env("VISAI_REMOTE_EXECUTOR_URL"),
    )


def require_model() -> str:
    s = settings()
    if not s.model:
        raise RuntimeError("OPENROUTER_MODEL is not set (put it in .env)")
    if not s.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env)")
    return s.model
