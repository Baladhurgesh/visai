# Visai

When optimizing kernels or deployments, load the `visai-opt` skill (`integrations/hermes/SKILL.md`) and follow it.
Memory lives in MongoDB Atlas (`MONGODB_URI`); without it Visai falls back to `out/localdb/`.
Run `uv run visai gate <op> <candidate>` and require `status=pass` before any benchmark.
Read `uv run visai memory show` first. After each candidate, check `uv run visai memory allowed "<next idea>"`.
Real win = Gate 1 pass on hidden inputs, >=3% over the realistic runtime baseline confirmed twice, quality within budget.
No launch-config-only bets. Do not rewrite vendor GEMM.
Keep going for 10 candidates per target unless a genuine win lands earlier.
