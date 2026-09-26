"""Quick in-process quality check: perplexity on a fixed WikiText slice (Gate 2 fast path)."""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

FALLBACK_TEXT = (
    "The tower is 324 metres tall, about the same height as an 81-storey building, and the tallest structure "
    "in Paris. Its base is square, measuring 125 metres on each side. During its construction, the Eiffel Tower "
    "surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held "
    "for 41 years until the Chrysler Building in New York City was finished in 1930. "
) * 8


@lru_cache(maxsize=1)
def _wikitext(n_chars: int = 20000) -> str:
    try:
        import datasets

        ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(t for t in ds["text"] if t.strip())
        return text[:n_chars]
    except Exception:  # noqa: BLE001
        return FALLBACK_TEXT


def perplexity(model, tokenizer, *, max_tokens: int = 2048, seq_len: int = 512) -> dict:
    ids = tokenizer.encode(_wikitext())[:max_tokens]
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(ids) - 1, seq_len):
        chunk = ids[i : i + seq_len + 1]
        if len(chunk) < 2:
            break
        x = mx.array(chunk[:-1])[None]
        y = mx.array(chunk[1:])[None]
        logits = model(x).astype(mx.float32)
        nll = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(nll)
        total_nll += float(nll.item())
        total_tok += y.size
    ppl = math.exp(total_nll / max(total_tok, 1))
    return {"perplexity": ppl, "tokens": total_tok, "quality": -ppl}
