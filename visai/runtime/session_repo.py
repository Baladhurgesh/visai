"""AtlasSessionRepository: Strands SessionRepository persisted in Atlas (resumable long runs)."""

from __future__ import annotations

from typing import Any

from strands.session import SessionRepository
from strands.types.session import Session, SessionAgent, SessionMessage

from visai.memory.db import get_db


class AtlasSessionRepository(SessionRepository):
    def __init__(self) -> None:
        self.db = get_db()

    def create_session(self, session: Session, **kwargs: Any) -> Session:
        self.db.sessions.update_one({"session_id": session.session_id}, {"$set": session.to_dict()}, upsert=True)
        return session

    def read_session(self, session_id: str, **kwargs: Any) -> Session | None:
        row = self.db.sessions.find_one({"session_id": session_id})
        if not row:
            return None
        row.pop("_id", None)
        return Session.from_dict(row)

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        self.db.session_agents.update_one(
            {"session_id": session_id, "agent_id": session_agent.agent_id},
            {"$set": {"session_id": session_id, **session_agent.to_dict()}},
            upsert=True,
        )

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> SessionAgent | None:
        row = self.db.session_agents.find_one({"session_id": session_id, "agent_id": agent_id})
        if not row:
            return None
        row.pop("_id", None)
        row.pop("session_id", None)
        return SessionAgent.from_dict(row)

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        self.create_agent(session_id, session_agent)

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        self.db.session_messages.update_one(
            {"session_id": session_id, "agent_id": agent_id, "message_id": session_message.message_id},
            {"$set": {"session_id": session_id, "agent_id": agent_id, **session_message.to_dict()}},
            upsert=True,
        )

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> SessionMessage | None:
        row = self.db.session_messages.find_one(
            {"session_id": session_id, "agent_id": agent_id, "message_id": message_id}
        )
        if not row:
            return None
        for k in ("_id", "session_id", "agent_id"):
            row.pop(k, None)
        return SessionMessage.from_dict(row)

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        self.create_message(session_id, agent_id, session_message)

    def list_messages(
        self, session_id: str, agent_id: str, limit: int | None = None, offset: int = 0, **kwargs: Any
    ) -> list[SessionMessage]:
        rows = self.db.session_messages.find(
            {"session_id": session_id, "agent_id": agent_id}, sort=[("message_id", 1)]
        )
        rows = rows[offset : (offset + limit) if limit else None]
        out = []
        for r in rows:
            for k in ("_id", "session_id", "agent_id"):
                r.pop(k, None)
            out.append(SessionMessage.from_dict(r))
        return out

    def create_multi_agent(self, session_id: str, multi_agent: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Visai does not use Strands multi-agent orchestration")

    def read_multi_agent(self, session_id: str, multi_agent_id: str, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def update_multi_agent(self, session_id: str, multi_agent: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Visai does not use Strands multi-agent orchestration")
