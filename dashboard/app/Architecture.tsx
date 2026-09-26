"use client";

import { useEffect, useState } from "react";

type Doc = Record<string, any>;

function Box({ x, y, w, h, title, sub, tone = "#1b2430", stroke = "#30363d" }: Doc) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} fill={tone} stroke={stroke} />
      <text x={x + w / 2} y={y + 22} fill="#e6edf3" fontSize="13" fontWeight={600} textAnchor="middle">{title}</text>
      {(sub || []).map((s: string, i: number) => (
        <text key={i} x={x + w / 2} y={y + 40 + i * 15} fill="#8b98a5" fontSize="11" textAnchor="middle">{s}</text>
      ))}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, label, dashed, lx, ly, anchor = "middle" }: Doc) {
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#58a6ff" strokeWidth={1.6} markerEnd="url(#arrow)" strokeDasharray={dashed ? "4 3" : undefined} />
      {label ? (
        <text x={lx ?? (x1 + x2) / 2} y={ly ?? (y1 + y2) / 2 - 5} fill="#8b98a5" fontSize="10" textAnchor={anchor}>{label}</text>
      ) : null}
    </g>
  );
}

function Diagram() {
  const W = 1180, H = 470;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W }}>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#58a6ff" />
        </marker>
      </defs>
      {/* top: pipeline stages owned by plain Python */}
      <text x={20} y={22} fill="#8b98a5" fontSize="11">OUTER LOOP (plain Python: budgets, gates, confirmation, stop rules)</text>
      <Box x={20} y={34} w={150} h={70} title="1 · Baseline" sub={["stock model on fixed", "data points"]} />
      <Box x={200} y={34} w={160} h={70} title="2 · Layer profile" sub={["self time per layer class", "+ capture real activations"]} />
      <Box x={390} y={34} w={200} h={70} title="3 · Layer kernel loops" sub={["per bottleneck layer, until", "target / saturation / caps"]} tone="#162133" stroke="#1f6feb" />
      <Box x={620} y={34} w={170} h={70} title="4 · Integrate" sub={["whole model, paired A/B", "keep only if e2e faster"]} />
      <Box x={820} y={34} w={170} h={70} title="5 · Model level" sub={["quantization / precision /", "scheduling configs"]} tone="#162133" stroke="#1f6feb" />
      <Box x={1020} y={34} w={140} h={70} title="6 · Compare" sub={["unoptimized vs", "optimized (paired)"]} />
      <Arrow x1={170} y1={69} x2={198} y2={69} />
      <Arrow x1={360} y1={69} x2={388} y2={69} />
      <Arrow x1={590} y1={69} x2={618} y2={69} />
      <Arrow x1={790} y1={69} x2={818} y2={69} />
      <Arrow x1={990} y1={69} x2={1018} y2={69} />

      {/* middle: one candidate iteration */}
      <text x={20} y={146} fill="#8b98a5" fontSize="11">ONE ITERATION (layer loop shown; model level uses evaluate_deploy_config instead of Gate 1 + benchmark)</text>
      <Box x={20} y={160} w={190} h={92} title="Writer agent" sub={["Strands + OpenRouter", "proposes ONE idea,", "writes / repairs kernel"]} tone="#1d1a2e" stroke="#8957e5" />
      <Box x={250} y={160} w={190} h={92} title="Interventions" sub={["write-scope · one-idea", "budget · do-not-repeat", "gate (no timing w/o Gate 1)"]} tone="#2a1d12" stroke="#d29922" />
      <Box x={480} y={160} w={200} h={92} title="Gate 1 (subprocess)" sub={["candidate vs model's own layer", "real captures + hidden", "x3 · x0.01 · negated · noise"]} tone="#122218" stroke="#3fb950" />
      <Box x={720} y={160} w={170} h={92} title="Benchmark ×2" sub={["same process as original", "robust = min of 2 runs", "new best? register kernel"]} tone="#122218" stroke="#3fb950" />
      <Box x={930} y={160} w={230} h={92} title="Record + Reflect agent" sub={["deterministic miss analysis", "reflect: why, next_try, lessons,", "skill, saturated?"]} tone="#1d1a2e" stroke="#8957e5" />
      <Arrow x1={210} y1={206} x2={248} y2={206} label="tool call" />
      <Arrow x1={440} y1={206} x2={478} y2={206} label="allowed" />
      <Arrow x1={680} y1={206} x2={718} y2={206} label="pass" />
      <Arrow x1={890} y1={206} x2={928} y2={206} />
      <path d="M 560 252 C 560 282, 170 282, 170 254" fill="none" stroke="#f85149" strokeDasharray="4 3" markerEnd="url(#arrow)" />
      <text x={440} y={296} fill="#f85149" fontSize="10" textAnchor="middle">Gate 1 fail → failure kinds only (hidden shapes never shown) → repair the same label</text>

      {/* bottom: memory */}
      <Box x={20} y={330} w={1140} h={120} title="MongoDB Atlas memory (persists across runs and machines)" sub={[]} tone="#101a14" stroke="#238636" />
      {[
        ["skills", "vector search · evidence · confidence"],
        ["do_not_repeat", "vector + token fingerprints"],
        ["lessons / reflections", "rendered into LESSONS.md"],
        ["experiments · events", "every trial · live feed"],
        ["kernels", "winning / integrated kernels"],
        ["sessions", "Strands session repository"],
      ].map(([t, s], i) => (
        <g key={t}>
          <rect x={40 + i * 186} y={368} width={170} height={62} rx={8} fill="#13241a" stroke="#2ea043" />
          <text x={125 + i * 186} y={392} fill="#e6edf3" fontSize="12" fontWeight={600} textAnchor="middle">{t}</text>
          <text x={125 + i * 186} y={410} fill="#8b98a5" fontSize="10" textAnchor="middle">{s}</text>
        </g>
      ))}
      <Arrow x1={60} y1={328} x2={60} y2={254} label="retrieve skills / lessons" dashed lx={66} ly={318} anchor="start" />
      <Arrow x1={1045} y1={254} x2={1045} y2={328} label="write trial, lessons, skill" dashed lx={1039} ly={318} anchor="end" />
      <Arrow x1={290} y1={328} x2={290} y2={254} label="do-not-repeat" dashed lx={296} ly={318} anchor="start" />
    </svg>
  );
}

const TOOLS = [
  ["layer_spec", "layer writer", "Layer source, parameters/attributes, children, captured input shapes, profile share."],
  ["op_spec", "kernel writer", "KernelBench op contract: signature, reference + runtime source, visible/bench shapes."],
  ["check_idea", "all writers", "Asks memory whether an idea repeats a known miss (token fingerprints + Atlas vector similarity)."],
  ["run_correctness", "layer / kernel writer", "Writes the candidate and runs Gate 1 in a subprocess against the trusted reference; returns pass/fail and failure kinds only."],
  ["run_benchmark", "layer / kernel writer", "Times candidate vs original twice in one process; robust speedup = min; registers the kernel if it is a new best."],
  ["evaluate_deploy_config", "model-level writer", "Loads the model with a config (+ kept layer kernels), measures quality, then paired A/B speed vs the current best."],
  ["search_memory / add_memory", "all writers", "Strands MemoryManager tools backed by AtlasMemoryStore ($vectorSearch over skills + lessons)."],
  ["WriterProposal / Reflection", "writer / reflect", "Pydantic structured outputs enforced by Strands (label, class, hypothesis, config, saturated, next_try, skill)."],
];

const INTERVENTIONS = [
  ["WriteScopeGuard", "Labels must be safe file stems; candidates can only be written inside the run's folder."],
  ["OneIdeaGuard", "One idea per iteration: repairs reuse the label, a new idea waits for the next iteration."],
  ["BudgetGuard", "Caps correctness runs, benchmarks, and total tool calls per iteration."],
  ["DoNotRepeatGuard", "A new idea that matches a do-not-repeat fingerprint is sent back (Guide) with the reason."],
  ["GateGuard", "Benchmark / apply is denied unless Gate 1 passed for the exact source hash being timed."],
];

const STOPS = [
  ["Target", "Layer or model reaches the target speedup (e.g. 2.0x), confirmed."],
  ["Saturation", "Writer or reflect agent declares diminishing returns (model level: only after ≥3 measured candidates)."],
  ["Patience", "N non-improving iterations in a row (layer 4, model level 3 by default)."],
  ["Hard caps", "Max iterations and wall-clock minutes per layer / per model-level stage."],
];

export default function Architecture() {
  const [a, setA] = useState<Doc | null>(null);
  useEffect(() => {
    fetch("/api/architecture").then((r) => r.json()).then(setA);
  }, []);
  return (
    <>
      <div className="panel">
        <h2>How Visai works</h2>
        <p style={{ margin: "4px 0 12px", maxWidth: 980 }}>
          Visai takes a model, this Mac, a workload and a quality budget. It measures the stock model, profiles which
          layers cost the most time, and has an agent write faster kernels for those layers. Each kernel must match the
          model's own layer on real and hidden inputs before it is timed, and must make the <i>whole model</i> faster before
          it is kept. It then searches model-level settings, compares unoptimized vs optimized head to head, and stores
          every lesson in MongoDB Atlas so the next run starts smarter.
        </p>
        <Diagram />
      </div>

      <div className="grid">
        <div className="panel">
          <h2>Agents (Strands Agents SDK on OpenRouter)</h2>
          <table>
            <tbody>
              <tr><td><b>Writer</b></td><td className="mono">{a?.writer_model || "…"}</td><td>Proposes one idea per iteration, writes and repairs the kernel or config through tools. Output: WriterProposal.</td></tr>
              <tr><td><b>Reflect</b></td><td className="mono">{a?.reflect_model || "…"}</td><td>After every trial: explains the miss mechanism, next idea of a different class, do-not-repeat fingerprints, lessons, reusable skill, saturation call. Output: Reflection.</td></tr>
              <tr><td><b>Embeddings</b></td><td className="mono">{a?.embed_model || "…"}</td><td>Vectors for skills, lessons and do-not-repeat entries (Atlas Vector Search).</td></tr>
              <tr><td><b>Harness</b></td><td className="mono">plain Python</td><td>Owns budgets, gates, confirmation runs, paired measurements and stop rules. Agents cannot skip a gate.</td></tr>
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h2>Gates</h2>
          <table>
            <tbody>
              <tr><td><b>Gate 1 · correctness</b></td><td>Candidate vs the model's own layer (or trusted op reference) on captured real activations plus hidden x3 / x0.01 / negated / noise inputs, in an isolated subprocess with a hidden seed. Static anti-cheat checks (no file/OS access, no global state, no calling the layer itself).</td></tr>
              <tr><td><b>Gate 2 · quality</b></td><td>Task metric within budget of the original: perplexity (Qwen), WER on LibriSpeech (Parakeet), DER on AMI (diarization).</td></tr>
              <tr><td><b>Speed</b></td><td>Layer: candidate vs original twice in the same process. Whole model: interleaved A/B/A/B rounds, median ratio, must win most rounds.</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid">
        <div className="panel">
          <h2>Tools the agents can call</h2>
          <table>
            <thead><tr><th>tool</th><th>used by</th><th>what it does</th></tr></thead>
            <tbody>{TOOLS.map(([t, u, d]) => <tr key={t}><td className="mono">{t}</td><td>{u}</td><td>{d}</td></tr>)}</tbody>
          </table>
        </div>
        <div className="panel">
          <h2>Interventions (rules enforced in code)</h2>
          <table>
            <tbody>{INTERVENTIONS.map(([t, d]) => <tr key={t}><td className="mono">{t}</td><td>{d}</td></tr>)}</tbody>
          </table>
          <h2 style={{ marginTop: 16 }}>Stop rules (never runs endlessly)</h2>
          <table>
            <tbody>{STOPS.map(([t, d]) => <tr key={t}><td><b>{t}</b></td><td>{d}</td></tr>)}</tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <h2>Memory in MongoDB Atlas {a ? <span className="pill" style={{ marginLeft: 8 }}>{a.backend === "atlas" ? "connected" : "local fallback"}</span> : null}</h2>
        <table>
          <thead><tr><th>collection</th><th>documents</th><th>what it holds / how it is used</th></tr></thead>
          <tbody>
            {[
              ["skills", "Reusable strategies with preconditions, evidence (trials, wins, mean/best speedup), Beta-posterior confidence and example source. Retrieved by $vectorSearch at the start of every loop."],
              ["do_not_repeat", "Fingerprints of failed ideas (from miss analysis and reflections). Checked by token rule + vector similarity before any new idea runs."],
              ["lessons", "Short hardware-aware lessons; rendered with trials into LESSONS.md for the writer prompt and Hermes."],
              ["reflections", "Every reflect-agent output (explain_miss, next_try, saturation)."],
              ["experiments", "Every trial: stock vs candidate, gate results, miss analysis, source for benchmarked candidates."],
              ["events", "Step-by-step agent activity (tool calls, Gate 1, benchmarks, integration) — powers the live feed."],
              ["kernels", "Winning and integrated kernels by content hash; final configs reference these ids so optimized models can be rebuilt anywhere."],
              ["profiles", "Layer self-time profiles and chosen targets per run."],
              ["comparisons", "Unoptimized vs optimized head-to-head results."],
              ["runs", "Run metadata, status, final reports."],
              ["sessions", "Strands AtlasSessionRepository: agent conversations are resumable."],
            ].map(([c, d]) => (
              <tr key={c}><td className="mono">{c}</td><td>{a?.counts?.[c] ?? "…"}</td><td>{d}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h2>Code map</h2>
        <table>
          <tbody>
            {[
              ["visai/pipeline.py", "Outer loop: baseline → profile → layer loops → integrate → model level → compare"],
              ["visai/profiler/layers.py", "Exclusive-time layer profiler + activation capture"],
              ["visai/verify/run_module.py", "Gate 1 + benchmark subprocess for layer kernels"],
              ["visai/runtime/*.py", "Strands models, agents, tools, interventions, Atlas memory store, session repo, hooks"],
              ["visai/deploy/search_loop.py", "Model-level config search with paired A/B and saturation rules"],
              ["visai/models/adapters.py", "Qwen / Parakeet / Nemotron adapters: load, workload, quality, paired speed"],
              ["visai/memory/*.py", "Atlas store: trials, do-not-repeat, lessons, skills, kernel registry"],
              ["integrations/hermes/", "SKILL.md + tool specs so Hermes / NemoClaw can drive the same tools"],
            ].map(([f, d]) => <tr key={f}><td className="mono">{f}</td><td>{d}</td></tr>)}
          </tbody>
        </table>
      </div>
    </>
  );
}
