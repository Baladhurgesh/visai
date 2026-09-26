"""Generate Hermes/NemoClaw tool specs from the same functions that back Visai's Strands tools."""

from __future__ import annotations

import json
from pathlib import Path

from visai.config import ROOT

HERMES = ROOT / "integrations" / "hermes"

CLI_TOOLS = [
    ("visai_gate", "Gate 1: correctness of a candidate kernel vs the trusted reference on seeded + hidden inputs.",
     "uv run visai gate <op> <candidate>", {"op": "string", "candidate": "string", "dtype": "string"}),
    ("visai_bench", "Microbenchmark reference, realistic runtime baseline, and a candidate across bench shapes.",
     "uv run visai bench <op> --candidate <candidate>", {"op": "string", "candidate": "string", "dtype": "string"}),
    ("visai_memory_show", "Rendered LESSONS.md, recent trials, latest reflection, and do-not-repeat list from Atlas.",
     "uv run visai memory show --target <target>", {"target": "string"}),
    ("visai_memory_allowed", "Check a next idea against persistent do-not-repeat memory (token rule + vector similarity).",
     "uv run visai memory allowed <idea> --target <target>", {"idea": "string", "target": "string"}),
    ("visai_skills", "List learned optimization skills with evidence and confidence.", "uv run visai skills list", {}),
    ("visai_profile", "ProfileBrief (op-family hotspots + targets) for an MLX model decode step.",
     "uv run visai profile --model <model>", {"model": "string"}),
    ("visai_eval", "Gate 2 frozen task eval (GSM8K / jsonl / HF dataset) with decode tok/s.",
     "uv run visai eval --model <model> --dataset gsm8k --n 10", {"model": "string", "dataset": "string", "n": "integer"}),
    ("visai_kernelbench", "Run Visai's own self-improving kernel loop (Strands writer + reflect agents).",
     "uv run visai kernelbench --ops <ops> --budget 10", {"ops": "string", "budget": "integer"}),
    ("visai_optimize", "End-to-end model optimization: deployment search + promoted kernels, gated on quality budget.",
     "uv run visai optimize --model <model> --quality-budget 0.5%", {"model": "string", "quality_budget": "string"}),
]


def export() -> dict:
    from pathlib import Path as _P

    from visai.memory.store import GateConfig
    from visai.runtime.state import LoopState
    from visai.runtime.tools import make_kernel_tools

    HERMES.mkdir(parents=True, exist_ok=True)
    state = LoopState(run_id="spec", target="spec", op="add_rmsnorm", dtype="float16", backend=None,
                      cand_dir=_P("/tmp"), stock={"throughput": 1.0})
    tools = []
    for t in make_kernel_tools(state, GateConfig()):
        spec = t.tool_spec
        name = f"visai_loop_{spec['name']}"
        doc = {"name": name, "description": spec["description"], "parameters": spec["inputSchema"]["json"],
               "entry": "in-loop Strands tool (visai kernelbench)"}
        (HERMES / f"tool_spec_{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
        tools.append({"name": name, "spec": f"tool_spec_{name}.json", "entry": "visai kernelbench"})
    for name, desc, entry, params in CLI_TOOLS:
        doc = {
            "name": name, "description": desc, "entry": entry,
            "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in params.items()}},
            "stdout": "JSON", "stderr": "human-readable table",
        }
        (HERMES / f"tool_spec_{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
        tools.append({"name": name, "spec": f"tool_spec_{name}.json", "entry": entry})
    tools.append({"name": "hermes_visai_agent", "spec": "SKILL.md", "entry": "hermes chat -s visai-opt"})
    (HERMES / "tools.json").write_text(json.dumps({"tools": tools}, indent=2) + "\n")
    return {"status": "ok", "dir": str(HERMES), "tools": len(tools)}


if __name__ == "__main__":
    print(json.dumps(export(), indent=2))
