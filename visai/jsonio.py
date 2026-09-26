"""Tool output convention from kernel-forge: JSON on stdout, human table on stderr."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def dumps(payload: Any, indent: int | None = 2) -> str:
    return json.dumps(payload, indent=indent, default=_default, ensure_ascii=False)


def dump(payload: Any, out: Path | None = None, pretty_stderr: str | None = None) -> None:
    text = dumps(payload)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    if pretty_stderr:
        sys.stderr.write(pretty_stderr.rstrip() + "\n")
    sys.stdout.write(text + "\n")


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("response had no JSON object")
    return json.loads(text[start : end + 1])
