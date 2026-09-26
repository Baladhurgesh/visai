"""Strands hooks -> Atlas `events` collection (feeds the live dashboard)."""

from __future__ import annotations

import json
from typing import Any

from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)

from visai.memory.store import log_event


def _short(obj: Any, n: int = 600) -> str:
    try:
        text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    except Exception:  # noqa: BLE001
        text = str(obj)
    return text[:n]


class AtlasEventHooks(HookProvider):
    def __init__(self, run_id: str, agent_role: str, target: str):
        self.run_id, self.role, self.target = run_id, agent_role, target

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self._before_invocation)
        registry.add_callback(AfterInvocationEvent, self._after_invocation)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)
        registry.add_callback(AfterModelCallEvent, self._after_model)

    def _log(self, step: str, **data: Any) -> None:
        log_event(self.run_id, step, agent=self.role, target=self.target, **data)

    def _before_invocation(self, event: BeforeInvocationEvent) -> None:
        self._log("agent_start")

    def _after_invocation(self, event: AfterInvocationEvent) -> None:
        usage = {}
        try:
            usage = dict(event.agent.event_loop_metrics.accumulated_usage)
        except Exception:  # noqa: BLE001
            pass
        self._log("agent_end", usage=usage)

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        tu = event.tool_use
        inp = dict(tu.get("input") or {})
        inp.pop("source", None)
        self._log("tool_call", tool=tu.get("name"), input=_short(inp, 400))

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        res = event.result
        status = res.get("status") if isinstance(res, dict) else "error"
        text = ""
        if isinstance(res, dict):
            for block in res.get("content") or []:
                if isinstance(block, dict) and "text" in block:
                    text += block["text"]
                elif isinstance(block, dict) and "json" in block:
                    text += _short(block["json"])
        self._log("tool_result", tool=event.tool_use.get("name"), status=status, result=_short(text, 700))

    def _after_model(self, event: AfterModelCallEvent) -> None:
        if event.exception is not None:
            self._log("model_error", error=_short(str(event.exception), 400))
