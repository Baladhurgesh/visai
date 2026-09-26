"""Eval task loading and grading (port of kernel-forge lib/eval_dataset.py + gsm8k_score.py).

EVAL_DATASET:
  gsm8k                    openai/gsm8k main (test split, train for shots)
  /path/to/file.jsonl      local rows
  org/name[:config[:split]] Hugging Face Hub
Prompt/gold fields are auto-detected; override with EVAL_PROMPT_FIELD / EVAL_GOLD_FIELD / EVAL_CHOICES_FIELD.
Scores: gsm8k | exact | lastline | choice.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

PROMPT_KEYS = ("question", "prompt", "input", "problem", "query", "instruction", "text", "ctx", "context")
GOLD_KEYS = ("answer", "gold", "target", "output", "answerKey", "correct_answer", "label", "answers", "answer_text")
CHOICES_KEYS = ("choices", "options", "endings", "choice")
SPLIT_PREF = ("test", "validation", "valid", "val", "dev", "train")
ANSWER_HINT = re.compile(r"(?:answer|choice|final)\s*[:\-is]+\s*([A-Za-z])", re.I)
LETTER_RE = re.compile(r"\b([A-D])\b")

TAG_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
NUM_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
GSM8K_INSTRUCTION = (
    "Solve the grade-school math problem. Show the steps. "
    "Put the final numeric answer on its own line in the form: #### <number>"
)


def env_or(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


# ------------------------------------------------------------------ gsm8k scoring


def normalize_number(text: str) -> str:
    s = text.strip().replace(",", "").replace("$", "")
    if s.endswith("."):
        s = s[:-1]
    try:
        val = float(s)
        return str(int(val)) if val.is_integer() else format(val, "g")
    except ValueError:
        return s


def extract_tagged(text: str) -> str | None:
    hits = TAG_RE.findall(text or "")
    return normalize_number(hits[-1]) if hits else None


def extract_last_number(text: str) -> str | None:
    hits = NUM_RE.findall(text or "")
    return normalize_number(hits[-1]) if hits else None


def gsm8k_gold(answer_field: str) -> str:
    tagged = extract_tagged(answer_field)
    if tagged is not None:
        return tagged
    last = extract_last_number(answer_field)
    if last is None:
        raise ValueError(f"no numeric gold in: {answer_field!r}")
    return last


def gsm8k_pred(completion: str) -> str | None:
    return extract_tagged(completion) or extract_last_number(completion)


def grade(completion: str, gold: str, mode: str) -> tuple[str | None, bool]:
    mode = (mode or "exact").lower()
    if mode == "gsm8k":
        pred = gsm8k_pred(completion)
        try:
            g = gsm8k_gold(gold)
        except ValueError:
            g = gold.strip()
        return pred, pred is not None and pred == g
    if mode == "choice":
        text = completion or ""
        hints = ANSWER_HINT.findall(text)
        pred = hints[-1].upper() if hints else None
        if pred is None:
            letters = LETTER_RE.findall(text)
            pred = letters[-1] if letters else None
        return pred, pred is not None and pred.upper() == gold.strip().upper()
    text = (completion or "").strip()
    pred = ([ln.strip() for ln in text.splitlines() if ln.strip()] or [""])[-1] if mode == "lastline" else text
    return pred, pred.casefold() == gold.strip().casefold()


# ------------------------------------------------------------------ rows


def stringify(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="replace")
    if isinstance(val, dict):
        for k in ("text", "label"):
            if k in val:
                return stringify(val[k])
        return None
    if isinstance(val, (list, tuple)):
        return stringify(val[0]) if val else None
    text = str(val).strip()
    return text or None


def _first(obj: dict[str, Any], keys: tuple[str, ...], explicit: str | None = None) -> Any:
    if explicit and obj.get(explicit) not in (None, ""):
        return obj[explicit]
    for key in keys:
        if obj.get(key) not in (None, ""):
            return obj[key]
    return None


def _choice_pairs(choices: Any) -> list[tuple[str, str]] | None:
    if choices is None:
        return None
    if isinstance(choices, dict):
        texts = choices.get("text") or choices.get("texts")
        labels = choices.get("label") or choices.get("labels")
        if texts:
            labs = list(labels) if labels else [chr(65 + i) for i in range(len(texts))]
            return [(str(lab), str(t)) for lab, t in zip(labs, texts)]
        return [(str(k), str(v)) for k, v in choices.items()]
    if isinstance(choices, (list, tuple)):
        return [(chr(65 + i), str(c)) for i, c in enumerate(choices)]
    return None


def normalize_row(obj: dict[str, Any], prompt_field=None, gold_field=None, choices_field=None) -> dict[str, str]:
    prompt = stringify(_first(obj, PROMPT_KEYS, prompt_field))
    gold = _first(obj, GOLD_KEYS, gold_field)
    pairs = _choice_pairs(_first(obj, CHOICES_KEYS, choices_field))
    if pairs:
        letter_of = {}
        lines = [prompt or ""]
        for lab, text in pairs:
            lines.append(f"{lab}. {text}")
            letter_of[text.strip().casefold()] = lab
            letter_of[lab.casefold()] = lab
        g = stringify(gold) or ""
        if g.casefold() in letter_of:
            g = letter_of[g.casefold()]
        elif g.isdigit() and int(g) < len(pairs):
            g = pairs[int(g)][0]
        return {"question": "\n".join(lines), "answer": g, "multiple_choice": "1"}
    gold_s = stringify(gold)
    if prompt is None or gold_s is None:
        raise ValueError(f"row needs prompt + gold; set EVAL_PROMPT_FIELD / EVAL_GOLD_FIELD. keys={sorted(obj)}")
    return {"question": prompt, "answer": gold_s, "multiple_choice": ""}


def sample_indices(n_total: int, n: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(range(n_total), min(n, n_total)))


def _hf_split(repo: str, config: str | None, split: str | None):
    import datasets

    obj = datasets.load_dataset(repo, name=config) if config else datasets.load_dataset(repo)
    wanted = split or env_or("EVAL_SPLIT") or next((s for s in SPLIT_PREF if s in obj), next(iter(obj.keys())))
    return obj[wanted], wanted


def load_task(
    dataset: str | None = None,
    n: int = 10,
    seed: int = 42,
    shots: int = 0,
    indices: list[int] | None = None,
    score: str | None = None,
) -> dict[str, Any]:
    name = (dataset or env_or("EVAL_DATASET") or "gsm8k").strip()
    pf, gf, cf = env_or("EVAL_PROMPT_FIELD") or None, env_or("EVAL_GOLD_FIELD") or None, env_or("EVAL_CHOICES_FIELD") or None
    path = Path(name).expanduser()
    if path.is_file():
        rows = [normalize_row(json.loads(l), pf, gf, cf) for l in path.read_text().splitlines() if l.strip()]
        idxs = indices if indices is not None else sample_indices(len(rows), n, seed)
        few = [r for i, r in enumerate(rows) if i not in set(idxs)][:shots]
        mc = any(rows[i].get("multiple_choice") for i in idxs)
        return {"name": path.stem, "examples": [rows[i] for i in idxs], "indices": idxs, "shots": few,
                "score": (score or env_or("EVAL_SCORE") or ("choice" if mc else "exact")).lower(), "instruction": ""}
    if name.lower() in {"gsm8k", "openai-gsm8k"}:
        repo, config, split = "openai/gsm8k", "main", "test"
    else:
        parts = name.removeprefix("hf:").split(":")
        repo, config, split = parts[0], (parts[1] if len(parts) > 1 else None), (parts[2] if len(parts) > 2 else None)
    ds, split_name = _hf_split(repo, config, split)
    idxs = indices if indices is not None else sample_indices(len(ds), n, seed)
    examples = [normalize_row(dict(ds[int(i)]), pf, gf, cf) for i in idxs]
    few: list[dict[str, str]] = []
    if shots:
        try:
            shot_ds, _ = _hf_split(repo, config, "train")
            few = [normalize_row(dict(shot_ds[i]), pf, gf, cf) for i in range(min(shots, len(shot_ds)))]
        except Exception:  # noqa: BLE001
            left = [i for i in range(len(ds)) if i not in set(idxs)][:shots]
            few = [normalize_row(dict(ds[i]), pf, gf, cf) for i in left]
    is_gsm = "gsm8k" in repo.lower()
    mc = any(e.get("multiple_choice") for e in examples)
    return {
        "name": repo + (f":{config}" if config else ""),
        "split": split_name,
        "examples": examples,
        "indices": idxs,
        "shots": few,
        "score": (score or env_or("EVAL_SCORE") or ("gsm8k" if is_gsm else "choice" if mc else "exact")).lower(),
        "instruction": GSM8K_INSTRUCTION if is_gsm else env_or("EVAL_INSTRUCTION"),
    }


def build_messages(question: str, shots: list[dict], instruction: str) -> list[dict[str, str]]:
    parts = [instruction] if instruction else []
    for ex in shots:
        parts += [f"Question: {ex['question'].strip()}", ex["answer"].strip()]
    parts.append(f"Question: {question.strip()}")
    return [{"role": "user", "content": "\n\n".join(parts)}]
