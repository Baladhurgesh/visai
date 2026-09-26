"""Gate 2: task quality on a frozen eval slice (port of kernel-forge eval_gsm8k.py).

Two drivers with the same EvalBrief output:
- run_eval_inprocess: an already-loaded (and possibly patched) mlx-lm model. Used inside the loop.
- run_eval_http: any OpenAI-compatible server (mlx_lm.server locally, vLLM on CUDA), streaming for real TTFT.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

from visai.profiler.hardware import probe
from visai.schemas import EvalBrief
from visai.verify.datasets import build_messages, grade, load_task


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ok = sum(1 for r in rows if r.get("correct"))
    errs = sum(1 for r in rows if r.get("error"))
    dec = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
    ttft = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
    toks = sum(r.get("completion_tokens") or 0 for r in rows)
    dwall = sum(max((r.get("total_s") or 0) - (r.get("ttft_s") or 0), 0) for r in rows)
    return {
        "n": n,
        "n_correct": ok,
        "n_errors": errs,
        "accuracy": ok / n if n else None,
        "median_decode_tok_s": statistics.median(dec) if dec else None,
        "median_ttft_s": statistics.median(ttft) if ttft else None,
        "overall_decode_tok_s": toks / dwall if dwall else None,
        "total_completion_tokens": toks,
    }


def format_table(m: dict[str, Any], name: str = "eval") -> str:
    acc = m.get("accuracy")
    acc_s = f"{100 * acc:.1f}%" if acc is not None else "n/a"
    return f"{name}  n={m.get('n')}  acc={acc_s} ({m.get('n_correct')}/{m.get('n')})  median decode={m.get('median_decode_tok_s')} tok/s"


def _prompt_text(tokenizer, messages: list[dict[str, str]], no_thinking: bool) -> str:
    kwargs = {"enable_thinking": False} if no_thinking else {}
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_eval_inprocess(
    model,
    tokenizer,
    *,
    dataset: str = "gsm8k",
    n: int = 10,
    seed: int = 42,
    shots: int = 4,
    max_tokens: int = 384,
    no_thinking: bool = True,
    gen_kwargs: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    task = task or load_task(dataset, n=n, seed=seed, shots=shots)
    sampler = make_sampler(temp=0.0)
    rows = []
    for idx, ex in zip(task["indices"], task["examples"]):
        prompt = _prompt_text(tokenizer, build_messages(ex["question"], task["shots"], task["instruction"]), no_thinking)
        text, last, ttft = "", None, None
        t0 = time.perf_counter()
        try:
            for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler, **(gen_kwargs or {})):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text += resp.text
                last = resp
            pred, ok = grade(text, ex["answer"], task["score"])
            rows.append(
                {
                    "index": idx, "gold": ex["answer"], "pred": pred, "correct": ok, "ttft_s": ttft,
                    "total_s": time.perf_counter() - t0,
                    "completion_tokens": last.generation_tokens if last else 0,
                    "decode_tok_s": last.generation_tps if last else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"index": idx, "gold": ex["answer"], "correct": False, "error": f"{type(exc).__name__}: {exc}"})
    metrics = summarize(rows)
    brief = EvalBrief(
        hardware=probe(),
        config={"dataset": task["name"], "n": len(rows), "seed": seed, "shots": len(task["shots"]),
                "score": task["score"], "indices": task["indices"], "no_thinking": no_thinking},
        metrics=metrics,
        failures=[r for r in rows if not r.get("correct")][:15],
    )
    out = brief.__dict__.copy()
    out["predictions"] = rows
    return out


def run_eval_http(
    base: str,
    model: str,
    *,
    dataset: str = "gsm8k",
    n: int = 10,
    seed: int = 42,
    shots: int = 4,
    max_tokens: int = 384,
    no_thinking: bool = True,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(base_url=base.rstrip("/") + "/v1", api_key="local")
    task = load_task(dataset, n=n, seed=seed, shots=shots)
    rows = []
    for idx, ex in zip(task["indices"], task["examples"]):
        msgs = build_messages(ex["question"], task["shots"], task["instruction"])
        t0, ttft, text, n_tok = time.perf_counter(), None, "", 0
        try:
            extra = {"chat_template_kwargs": {"enable_thinking": False}} if no_thinking else {}
            stream = client.chat.completions.create(
                model=model, messages=msgs, max_tokens=max_tokens, temperature=0.0, stream=True, extra_body=extra
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text += delta
                    n_tok += 1
            total = time.perf_counter() - t0
            pred, ok = grade(text, ex["answer"], task["score"])
            dec = n_tok / (total - ttft) if ttft is not None and total > ttft else None
            rows.append({"index": idx, "gold": ex["answer"], "pred": pred, "correct": ok, "ttft_s": ttft,
                         "total_s": total, "completion_tokens": n_tok, "decode_tok_s": dec})
        except Exception as exc:  # noqa: BLE001
            rows.append({"index": idx, "gold": ex["answer"], "correct": False, "error": f"{type(exc).__name__}: {exc}"})
    brief = EvalBrief(
        hardware=probe(),
        config={"base": base, "model": model, "dataset": task["name"], "n": len(rows), "seed": seed},
        metrics=summarize(rows),
        failures=[r for r in rows if not r.get("correct")][:15],
    )
    out = brief.__dict__.copy()
    out["predictions"] = rows
    return out
