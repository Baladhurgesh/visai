"""Backend protocol: where candidates compile, get verified, and get timed."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Backend(Protocol):
    name: str
    hardware: str
    candidate_language: str

    def evaluate(self, op: str, candidate: Path, dtype: str, mode: str = "all") -> dict[str, Any]:
        """Run Gate 1 (correctness) and/or the microbenchmark. Returns kernel-forge style JSON."""
        ...

    def baseline(self, op: str, dtype: str) -> dict[str, Any]:
        """Time reference + runtime implementations without a candidate."""
        ...


class Patcher(Protocol):
    def apply(self) -> dict[str, Any]: ...

    def revert(self) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...
