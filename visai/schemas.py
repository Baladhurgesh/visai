"""Stable JSON contracts.

The dataclasses (WorkloadSpec ... ProfileBrief, EvalBrief) are ported from kernel-forge
and keep their field names so Hermes/NemoClaw tools stay compatible. The pydantic models
are Visai additions and double as Strands structured-output schemas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Priority = Literal["rewrite", "fuse", "skip", "inspect"]


@dataclass
class WorkloadSpec:
    model: str = ""
    batch: int = 1
    input_tokens: int = 512
    output_tokens: int = 256
    temperature: float = 0.0
    precision: str = "fixed"
    ignore_eos: bool = True


@dataclass
class RunSample:
    total_s: float
    ttft_s: float | None
    prompt_tokens: int
    completion_tokens: int
    prefill_tok_s: float | None
    decode_tok_s: float | None


@dataclass
class Baseline:
    warmup: int = 0
    timed_runs: int = 0
    median_total_s: float | None = None
    median_ttft_s: float | None = None
    median_prefill_tok_s: float | None = None
    median_decode_tok_s: float | None = None
    runs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FamilyStat:
    name: str
    gpu_pct: float
    time_ns: float
    launches: int
    priority: Priority
    triton_candidate: bool
    reason: str
    example_kernel: str


@dataclass
class AgentTarget:
    family: str
    kernel: str
    gpu_pct: float
    launches: int
    runtime_note: str
    goal: str = "Improve latency without changing numerical output."
    constraints: list[str] = field(
        default_factory=lambda: [
            "Do not change numerics beyond atol/rtol of the current kernel.",
            "Do not rewrite vendor GEMM first.",
        ]
    )
    suggested_experiments: list[str] = field(default_factory=list)


@dataclass
class ProfileBrief:
    schema_version: str = SCHEMA_VERSION
    tool: str = "profile_inference_hotspots"
    status: str = "ok"
    error: str | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    workload: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    families: list[dict[str, Any]] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_json_obj(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalBrief:
    schema_version: str = SCHEMA_VERSION
    tool: str = "eval_task_baseline"
    status: str = "ok"
    hardware: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)


# ---------------------------------------------------------------- Visai models

IdeaClass = Literal[
    "fusion",
    "fewer_launches",
    "vectorize",
    "reduction_strategy",
    "tiling",
    "precision",
    "launch_config_tune",
    "quantization",
    "kv_cache",
    "scheduling",
    "other",
]


class CandidateFile(BaseModel):
    path: str = Field(description="Relative path, e.g. candidates/rmsnorm_v3.py")
    replaces: str = Field(default="", description="Module/function this candidate replaces")
    content: str = Field(description="Full, runnable source of the candidate")


class WriterProposal(BaseModel):
    """Writer contract (kernel-forge kernel_prompt.py) as a Strands structured output."""

    label: str = Field(description="Short unique label, e.g. 'rmsnorm_simd_reduce_v1'")
    cls: IdeaClass = Field(description="Idea class; must differ from the last miss class")
    hypothesis: str
    targets: list[str] = Field(default_factory=list)
    files: list[CandidateFile] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict, description="Deployment config for non-kernel ideas")
    experiments: list[str] = Field(default_factory=list)
    correctness: str = ""
    expected_speedup: str = ""
    skills_used: list[str] = Field(default_factory=list)
    saturated: bool = Field(
        default=False,
        description="True if you believe no remaining idea is likely to beat the current best for this target",
    )
    saturation_reason: str = ""


class NextTry(BaseModel):
    title: str = ""
    why: str = ""
    how: str = ""
    cls: str = "other"


class Reflection(BaseModel):
    """Reflect contract (kernel-forge judge_kernel.py) as a Strands structured output."""

    model_config = {"populate_by_name": True}

    continue_: bool = Field(default=True, alias="continue")
    genuine_win: bool = False
    explain_miss: str = ""
    opinion: str = ""
    next_try: NextTry = Field(default_factory=NextTry)
    next_ideas: list[NextTry] = Field(default_factory=list)
    do_not_repeat: list[str] = Field(default_factory=list)
    lesson_lines: list[str] = Field(default_factory=list)
    skill: dict[str, Any] | None = Field(
        default=None,
        description="Optional reusable lesson: {name, operator, observation, preconditions, strategy}",
    )
    saturated: bool = Field(
        default=False,
        description="True if the search for this target has hit a saturation point (diminishing returns)",
    )
    saturation_reason: str = ""


class MissAnalysis(BaseModel):
    outcome: str
    cls: str
    why: str
    do_not_repeat: list[str] = Field(default_factory=list)
    label: str = ""
    delta_pct: float = 0.0


class SkillEvidence(BaseModel):
    trials: int = 0
    correctness_passes: int = 0
    positive_speedups: int = 0
    mean_speedup: float = 0.0
    best_speedup: float = 0.0


class Skill(BaseModel):
    name: str
    operator: str
    backend: str
    hardware: str
    dtype: str = "float16"
    observation: str = ""
    preconditions: dict[str, Any] = Field(default_factory=dict)
    strategy: list[str] = Field(default_factory=list)
    evidence: SkillEvidence = Field(default_factory=SkillEvidence)
    negative_evidence: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    example_source: str = ""
