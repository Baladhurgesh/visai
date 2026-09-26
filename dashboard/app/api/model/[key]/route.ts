import { NextResponse } from "next/server";
import { find, findOne } from "@/lib/db";

export const dynamic = "force-dynamic";

const MODELS: Record<string, { id: string; label: string; quality: string; unit: string }> = {
  qwen: { id: "Qwen/Qwen3-0.6B", label: "Qwen3-0.6B", quality: "perplexity", unit: "decode tok/s" },
  parakeet: { id: "mlx-community/parakeet-tdt-0.6b-v3", label: "Parakeet-TDT 0.6B v3", quality: "wer", unit: "x real-time" },
  diar: { id: "mlx-community/Nemotron-3-Diarization", label: "Nemotron-3 Diarization", quality: "der", unit: "x real-time" },
};

type Doc = Record<string, any>;

function layerOf(target: string, key: string): string | null {
  // layer targets look like "parakeet:LSTM@mlx:M3Pro"
  if (!target?.startsWith(`${key}:`) || !target.includes("@")) return null;
  const name = target.slice(key.length + 1).split("@")[0];
  return name && name !== "e2e" ? name : null;
}

export async function GET(_req: Request, ctx: { params: Promise<{ key: string }> }) {
  const { key } = await ctx.params;
  const m = MODELS[key];
  if (!m) return NextResponse.json({ error: "unknown model" }, { status: 404 });

  const allRuns = await find("runs", {}, { sort: { started: 1 } });
  const runs = allRuns.filter(
    (r) => r.adapter === key || r.model === m.id || String(r.target || "").startsWith(m.id) ||
      (key !== "qwen" && r.model === "parakeet+nemotron-diar"),
  );
  const runIds = new Set(runs.map((r) => r.run_id));
  const pipelines = runs.filter((r) => r.kind === "pipeline");
  const latestDone = [...pipelines].reverse().find((r) => r.status === "done" && r.report?.comparison);

  const ids = [...runIds];
  const steps = ["layer_baseline", "skills_retrieved", "blocked_idea", "gate1", "benchmark", "saturated", "saturation_rejected",
    "target_reached", "integration", "model_level_best"];
  const [ev, ex, comparison, profiles] = await Promise.all([
    find("events", { run_id: { $in: ids }, step: { $in: steps } }, { sort: { ts: 1 } }),
    find("experiments", { run_id: { $in: ids } }, { sort: { ts: 1 }, project: { "candidate.source": 0 } }),
    findOne("comparisons", { adapter: key }, { ts: -1 }),
    find("profiles", { kind: "layers", run_id: { $in: ids } }, { sort: { ts: -1 } }),
  ]);
  const profile = profiles.find((p) => runIds.has(p.run_id) && p.layers?.length) || null;

  // ---------------- layers: every attempt the agent made, per layer, per run
  const layers: Record<string, any> = {};
  const L = (name: string) =>
    (layers[name] ||= { name, runs: {} as Record<string, any>, attempts: [] as Doc[], skills: new Set<string>(), blocked: [] as Doc[] });
  const R = (lay: any, run: string) =>
    (lay.runs[run] ||= { run_id: run, original_us: null, best: 1, first: null, attempts: 0, bestAt: null, stop: null, skills: [], blocked: 0, resumed: false });

  for (const e of ev) {
    const name = layerOf(e.target, key);
    if (!name) continue;
    const lay = L(name);
    const run = R(lay, e.run_id);
    if (e.step === "layer_baseline") run.original_us = Number(e.runtime_us);
    if (e.step === "skills_retrieved") {
      run.skills = e.skills || [];
      (e.skills || []).forEach((s: string) => lay.skills.add(s));
    }
    if (e.step === "blocked_idea") {
      run.blocked += 1;
      lay.blocked.push({ run_id: e.run_id, ts: e.ts, label: e.label, blocked_by: e.blocked_by });
    }
    if (e.step === "gate1") {
      let a = lay.attempts.find((x: Doc) => x.run_id === e.run_id && x.label === e.label);
      if (!a) {
        a = { run_id: e.run_id, label: e.label, ts: e.ts, gate1: [], speedup: null, improved: false };
        lay.attempts.push(a);
      }
      a.gate1.push({ status: e.status, n_pass: e.n_pass, n: e.n });
    }
    if (e.step === "benchmark" && e.improved !== undefined) {
      const a = lay.attempts.find((x: Doc) => x.run_id === e.run_id && x.label === e.label);
      const sp = Number(e.speedup);
      if (a) {
        a.speedup = sp;
        a.improved = String(e.improved) === "True" || e.improved === true;
      }
      run.attempts += 1;
      if (run.first === null) run.first = sp;
      if ((String(e.improved) === "True" || e.improved === true) && sp > run.best) {
        run.best = sp;
        run.bestAt = run.attempts;
      }
    }
    if (e.step === "saturated" || e.step === "target_reached") run.stop = e.reason || `target reached ${Number(e.speedup).toFixed(3)}x`;
  }
  // idea class / hypothesis / miss reason from recorded trials
  for (const t of ex) {
    const name = layerOf(t.target, key);
    if (!name || !layers[name]) continue;
    const c = t.candidate || {};
    const a = layers[name].attempts.find((x: Doc) => x.run_id === t.run_id && x.label === c.label);
    if (a) Object.assign(a, { cls: c.cls, idea: c.idea || c.hypothesis, outcome: t.miss?.outcome, why: t.miss?.why });
  }
  const layerList = Object.values(layers).map((lay: any) => {
    const runsArr = Object.values(lay.runs).filter((r: any) => r.attempts > 0 || r.original_us);
    const best = Math.max(1, ...runsArr.map((r: any) => r.best));
    const prof = profile?.layers?.find((p: Doc) => p.name === lay.name);
    return { ...lay, skills: [...lay.skills], runs: runsArr, best, pct: prof?.pct ?? null, calls: prof?.calls ?? null };
  });
  layerList.sort((a: any, b: any) => (b.pct ?? 0) - (a.pct ?? 0));

  // ---------------- integration + model level (latest completed pipeline, plus history)
  const integration = ev.filter((e) => e.step === "integration").map((e) => ({ ...e }));
  const modelLevel = ex
    .filter((t) => !layerOf(t.target, key) && t.stage !== "integration" && (t.op || "").match(/model_level|_deploy|model_kernels/))
    .map((t) => ({
      run_id: t.run_id, ts: t.ts, slot: t.slot, label: t.candidate?.label, cls: t.candidate?.cls, config: t.candidate?.config,
      speedup_vs_best: t.candidate?.speedup_vs_best, paired: t.candidate?.paired_vs_best?.ratio,
      quality: t.candidate?.[m.quality], confirmed: t.candidate?.confirmed, outcome: t.miss?.outcome, why: t.miss?.why, keep: t.genuine,
    }));
  const e2eBest = ev.filter((e) => e.step === "model_level_best").map((e) => ({ run_id: e.run_id, ts: e.ts, label: e.label, e2e: Number(e.e2e_vs_stock) }));
  const modelBlocked = ev.filter((e) => e.step === "blocked_idea" && !layerOf(e.target, key));
  const saturations = ev.filter((e) => ["saturated", "saturation_rejected", "target_reached"].includes(e.step) && !layerOf(e.target, key));

  return NextResponse.json({
    key, model: m, comparison, latest: latestDone || null,
    runs: runs.map((r) => ({ run_id: r.run_id, kind: r.kind, status: r.status, started: r.started, target_speedup: r.target_speedup, note: r.note })),
    profile: profile ? { run_id: profile.run_id, layers: profile.layers.slice(0, 12), targets: profile.targets } : null,
    layers: layerList, integration, modelLevel, e2eBest, modelBlocked, saturations,
  });
}
