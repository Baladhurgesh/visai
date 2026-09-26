"""Strands agents: writer (tools + interventions + Atlas memory) and reflect (structured output only)."""

from __future__ import annotations

from strands import Agent
from strands.memory import MemoryManagerConfig
from strands.session import RepositorySessionManager

from visai.agent.prompts import REFLECT_SYSTEM, WRITER_SYSTEM
from visai.runtime.hooks import AtlasEventHooks
from visai.runtime.memory_store import AtlasMemoryStore
from visai.runtime.models import reflect_model, writer_model
from visai.runtime.session_repo import AtlasSessionRepository
from visai.schemas import Reflection, WriterProposal


def build_writer(run_id: str, target: str, backend: str, tools: list, interventions: list, slot: int) -> Agent:
    return Agent(
        name="visai_writer",
        agent_id=f"writer_{slot}",
        model=writer_model(),
        system_prompt=WRITER_SYSTEM,
        tools=tools,
        interventions=interventions,
        hooks=[AtlasEventHooks(run_id, "writer", target)],
        memory_manager=MemoryManagerConfig(
            stores=[AtlasMemoryStore(scope=target, backend=backend)],
            add_tool_config=True,
            injection=False,
        ),
        session_manager=RepositorySessionManager(
            session_id=f"{run_id}_writer_{slot}", session_repository=AtlasSessionRepository()
        ),
        structured_output_model=WriterProposal,
        callback_handler=None,
    )


def build_reflector(run_id: str, target: str) -> Agent:
    return Agent(
        name="visai_reflect",
        model=reflect_model(),
        system_prompt=REFLECT_SYSTEM,
        tools=[],
        hooks=[AtlasEventHooks(run_id, "reflect", target)],
        structured_output_model=Reflection,
        callback_handler=None,
    )
