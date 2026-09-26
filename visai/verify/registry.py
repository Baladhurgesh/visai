"""VerifierRegistry: task -> public benchmark + quality metric, with staged quick/full checks.

quick(): seconds, used on every candidate during search.
full():  larger frozen slice, used before promoting a winner.
Every verifier returns {"quality": <higher is better>, ...} so the generic gate can compare.
"""

from __future__ import annotations

from typing import Any


class LLMVerifier:
    task = "causal-lm"
    metric = "perplexity (quick) / GSM8K accuracy (full)"

    def __init__(self, dataset: str = "gsm8k", n_full: int = 10, shots: int = 4):
        self.dataset, self.n_full, self.shots = dataset, n_full, shots

    def quick(self, model, tokenizer, gen_kwargs: dict | None = None) -> dict[str, Any]:
        from visai.verify.perplexity import perplexity

        return {"kind": "perplexity", **perplexity(model, tokenizer)}

    def full(self, model, tokenizer, gen_kwargs: dict | None = None) -> dict[str, Any]:
        from visai.verify.task_eval import run_eval_inprocess

        brief = run_eval_inprocess(
            model, tokenizer, dataset=self.dataset, n=self.n_full, shots=self.shots, gen_kwargs=gen_kwargs
        )
        m = brief["metrics"]
        return {"kind": "task_eval", "quality": m.get("accuracy"), "metrics": m, "config": brief["config"]}


class ASRVerifier:
    task = "automatic-speech-recognition"
    metric = "WER on LibriSpeech test-clean (quality = -WER)"

    def __init__(self, n_quick: int = 20, n_full: int = 200):
        self.n_quick, self.n_full = n_quick, n_full

    def _wer(self, transcribe, n: int) -> dict[str, Any]:
        import datasets
        import jiwer

        ds = datasets.load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
        refs, hyps = [], []
        for i, row in enumerate(ds):
            if i >= n:
                break
            refs.append(row["text"].lower())
            hyps.append(transcribe(row["audio"]["array"], row["audio"]["sampling_rate"]).lower())
        wer = jiwer.wer(refs, hyps)
        return {"kind": "wer", "wer": wer, "quality": -wer, "n": len(refs)}

    def quick(self, transcribe, *_: Any, **__: Any) -> dict[str, Any]:
        return self._wer(transcribe, self.n_quick)

    def full(self, transcribe, *_: Any, **__: Any) -> dict[str, Any]:
        return self._wer(transcribe, self.n_full)


class ClassificationVerifier:
    task = "image-classification"
    metric = "top-1 on an ImageNet validation slice"

    def quick(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise NotImplementedError("image classification verifier is not wired yet")

    full = quick


VerifierRegistry: dict[str, type] = {
    "causal-lm": LLMVerifier,
    "automatic-speech-recognition": ASRVerifier,
    "image-classification": ClassificationVerifier,
}


def verifier_for(task: str, **kwargs: Any):
    if task not in VerifierRegistry:
        raise KeyError(f"no verifier for task {task}; have {sorted(VerifierRegistry)}")
    return VerifierRegistry[task](**kwargs)
