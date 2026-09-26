"""Strands model providers: OpenAIModel pointed at OpenRouter."""

from __future__ import annotations

from strands.models.openai import OpenAIModel

from visai.config import require_model, settings


def openrouter_model(model_id: str | None = None, *, max_tokens: int = 16000, reasoning: str | None = "high") -> OpenAIModel:
    require_model()
    s = settings()
    params: dict = {"max_tokens": max_tokens}
    extra: dict = {}
    if reasoning:
        extra["reasoning"] = {"effort": reasoning}
    if extra:
        params["extra_body"] = extra
    return OpenAIModel(
        client_args={
            "api_key": s.openrouter_api_key,
            "base_url": s.openrouter_base,
            "default_headers": {"HTTP-Referer": "https://localhost/visai", "X-Title": "visai"},
        },
        model_id=model_id or s.model,
        params=params,
    )


def writer_model() -> OpenAIModel:
    return openrouter_model(settings().model)


def reflect_model() -> OpenAIModel:
    return openrouter_model(settings().reflect_model, max_tokens=8000)
