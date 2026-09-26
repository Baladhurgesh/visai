---
name: visai-opt
description: Profile, write, verify, benchmark, and learn hardware-specific kernel and deployment optimizations.
version: 0.1.0
author: Baladhurgesh
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [MLX, Metal, Triton, vLLM, eval, MongoDB, self-improving]
---

# Visai optimization loop

Use this when the user wants a faster kernel or deployment for a model on specific hardware, verified
against a trusted reference and a task benchmark, with memory that carries across sessions.

Visai's own agent loop (`visai kernelbench`, `visai optimize`) runs this skill end to end with Strands
agents on OpenRouter. As a Hermes/NemoClaw skill you are the writer; the tools below are the same ones.

## Working directory

```bash
export VISAI_ROOT="${VISAI_ROOT:-$(pwd)}"
cd "$VISAI_ROOT"
```

Required env (in `.env`, never hardcoded): `MONGODB_URI`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
(optional `OPENROUTER_REFLECT_MODEL`).

## Memory first

```bash
uv run visai memory show --target "<op>@mlx:<chip>:float16"
```

Read Miss reasons + Avoid in the rendered LESSONS.md. Before a new idea:

```bash
uv run visai memory allowed "<label> <class> <what changes>" --target "<target>"
```

If `allowed` is false, pick a different class of idea.

## Candidate contract (MLX / Metal)

A Python module defining `kernel(...)` with exactly the op signature (see `uv run visai bench <op>` and
`visai/tasks/ops.py`). Build kernels once at import with `mx.fast.metal_kernel`. No file/OS access, no
global state, no output caching.

## Gate 1: correctness (required before any timing)

```bash
uv run visai gate <op> candidates/<label>.py
```

Seeded visible cases plus hidden shapes and scaled / tiny / negated distributions. Accept only `status=pass`.

## Benchmark (twice)

```bash
uv run visai bench <op> --candidate candidates/<label>.py
```

## Genuine win

- Gate 1 pass on hidden inputs, n_errors = 0
- geomean speedup over the realistic runtime baseline >= 3%, confirmed by a second run
- for model-level changes: task quality within `--quality-budget`

Launch-config-only retunes are not a win path. Prefer fusion and fewer launches. Do not rewrite vendor GEMM.

## After each candidate

Record it through the loop (`visai kernelbench` records automatically) or reflect manually, then refresh:

```bash
uv run visai memory refresh
uv run visai memory sync-hermes
```

Run **10** candidates per target unless a confirmed genuine win lands earlier.
