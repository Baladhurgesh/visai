"""Per-run state shared by tools and interventions (what the gates check against)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class SlotState:
    """One candidate slot (kernel-forge: one of the 10 candidates). Repairs stay inside the slot."""

    index: int
    labels: set[str] = field(default_factory=set)
    correctness_attempts: int = 0
    benchmark_calls: int = 0
    tool_calls: int = 0
    guides: int = 0
    passed: dict[str, str] = field(default_factory=dict)  # label -> source hash that passed Gate 1
    last_correctness: dict[str, Any] | None = None
    result: dict[str, Any] | None = None  # final benchmarked candidate


@dataclass
class LoopState:
    run_id: str
    target: str
    op: str
    dtype: str
    backend: Any
    cand_dir: Path
    stock: dict[str, Any]
    baseline_mode: str = "runtime"
    max_attempts_per_slot: int = 4
    max_benchmarks_per_slot: int = 2
    max_guides_per_slot: int = 3
    max_tool_calls_per_slot: int = 12
    slot: SlotState = field(default_factory=lambda: SlotState(index=0))

    def new_slot(self, index: int) -> SlotState:
        self.slot = SlotState(index=index)
        return self.slot

    def path_for(self, label: str) -> Path:
        return self.cand_dir / f"{label}.py"
