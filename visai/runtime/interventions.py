"""Strands interventions: kernel-forge's rules enforced in code instead of in the prompt.

- WriteScopeGuard:   candidate labels must be safe file stems (writes stay in the run's candidate dir)
- BudgetGuard:       bounded repairs / benchmarks per candidate slot
- DoNotRepeatGuard:  a new idea that overlaps a do-not-repeat fingerprint is sent back (Guide) with the reason
- GateGuard:         no benchmark / apply unless Gate 1 passed for the exact source being benchmarked
"""

from __future__ import annotations

import re
from typing import Any

from strands.hooks import BeforeToolCallEvent
from strands.interventions import Deny, Guide, InterventionHandler, Proceed

from visai.memory.store import idea_allowed, log_event
from visai.runtime.state import LoopState, source_hash

LABEL_RE = re.compile(r"^[a-z0-9_]{3,64}$")
IDEA_TOOLS = {"run_correctness", "evaluate_deploy_config"}


def _inp(event: BeforeToolCallEvent) -> dict[str, Any]:
    return dict(event.tool_use.get("input") or {})


class WriteScopeGuard(InterventionHandler):
    name = "write-scope"

    def __init__(self, state: LoopState):
        self.state = state

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any):
        if event.tool_use.get("name") not in IDEA_TOOLS | {"run_benchmark", "apply_candidate"}:
            return Proceed()
        label = str(_inp(event).get("label") or "")
        if not LABEL_RE.match(label):
            return Deny(reason=f"label {label!r} must match {LABEL_RE.pattern} (it becomes a file name)")
        return Proceed()


class OneIdeaGuard(InterventionHandler):
    """One idea per candidate slot (kernel-forge rule): repairs reuse the label; a new label waits for the next slot."""

    name = "one-idea"

    def __init__(self, state: LoopState):
        self.state = state

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any):
        if event.tool_use.get("name") not in IDEA_TOOLS:
            return Proceed()
        label = str(_inp(event).get("label") or "")
        slot = self.state.slot
        if slot.labels and label not in slot.labels and (slot.result is not None or slot.passed):
            return Deny(reason=(
                f"This iteration already evaluated '{next(iter(slot.labels))}'. One idea per iteration: return your "
                "structured proposal now; propose the new idea next iteration."))
        return Proceed()


class BudgetGuard(InterventionHandler):
    name = "budget"

    def __init__(self, state: LoopState):
        self.state = state

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any):
        name = event.tool_use.get("name")
        slot = self.state.slot
        slot.tool_calls += 1
        if slot.tool_calls > self.state.max_tool_calls_per_slot:
            return Deny(reason="tool-call budget for this candidate is exhausted. Return your structured proposal now.")
        if name in IDEA_TOOLS:
            if slot.correctness_attempts >= self.state.max_attempts_per_slot:
                return Deny(
                    reason=f"repair budget exhausted ({self.state.max_attempts_per_slot} runs for this "
                    "candidate). Stop and return your structured proposal now."
                )
            slot.correctness_attempts += 1
        if name == "run_benchmark":
            if slot.benchmark_calls >= self.state.max_benchmarks_per_slot:
                return Deny(reason="benchmark budget exhausted for this candidate. Return your structured proposal now.")
            slot.benchmark_calls += 1
        return Proceed()


class DoNotRepeatGuard(InterventionHandler):
    name = "do-not-repeat"

    def __init__(self, state: LoopState):
        self.state = state

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any):
        if event.tool_use.get("name") not in IDEA_TOOLS:
            return Proceed()
        inp = _inp(event)
        label = str(inp.get("label") or "")
        slot = self.state.slot
        if label in slot.labels:  # a repair of an idea already admitted in this slot
            return Proceed()
        idea = f"{label} {inp.get('idea_class', '')} {inp.get('idea', '')}"
        verdict = idea_allowed(idea, scope=self.state.target)
        if verdict["allowed"]:
            return Proceed()
        slot.guides += 1
        log_event(self.state.run_id, "blocked_idea", target=self.state.target, label=label, blocked_by=verdict["blocked_by"])
        if slot.guides > self.state.max_guides_per_slot:
            return Deny(reason="too many blocked ideas in a row; return your structured proposal now.")
        return Guide(
            feedback=(
                f"Idea '{label}' is blocked by persistent memory: {verdict['blocked_by']}. "
                "It repeats a known miss. Pick a DIFFERENT class of idea and a new label."
            )
        )


class GateGuard(InterventionHandler):
    name = "gate"

    def __init__(self, state: LoopState):
        self.state = state

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any):
        if event.tool_use.get("name") not in ("run_benchmark", "apply_candidate"):
            return Proceed()
        label = str(_inp(event).get("label") or "")
        path = self.state.path_for(label)
        passed = self.state.slot.passed.get(label)
        if not path.exists() or passed is None or passed != source_hash(path.read_text()):
            return Deny(
                reason=(
                    f"Gate 1 has not passed for the current source of '{label}'. "
                    "Run run_correctness until status=pass before benchmarking. A failing candidate is never timed or applied."
                )
            )
        return Proceed()


def kernel_interventions(state: LoopState) -> list[InterventionHandler]:
    return [WriteScopeGuard(state), OneIdeaGuard(state), BudgetGuard(state), DoNotRepeatGuard(state), GateGuard(state)]
