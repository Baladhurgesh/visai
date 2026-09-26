import { NextResponse } from "next/server";
import { count, find } from "@/lib/db";

export const dynamic = "force-dynamic";

type Doc = Record<string, any>;

const LABELS: Record<string, string> = {
  qwen: "Qwen3-0.6B",
  parakeet: "Parakeet-TDT 0.6B v3",
  diar: "Nemotron-3 Diarization",
};

export async function GET() {
  const [comparisons, validations, kernels, runs, e2eEvents, lstmEvents] = await Promise.all([
    find("comparisons", {}, { sort: { ts: -1 } }),
    find("validations", {}, { sort: { ts: -1 } }),
    find("kernels", {}, { project: { source: 0 } }),
    find("runs", { kind: "pipeline" }, { sort: { started: 1 } }),
    find("events", { step: "e2e_after_layers" }, { sort: { ts: -1 } }),
    find("events", { step: "benchmark", target: "parakeet:LSTM@mlx:M3Pro" }, { sort: { ts: 1 } }),
  ]);

  // latest comparison per model = the result shown
  const latest: Record<string, Doc> = {};
  for (const c of comparisons) if (!latest[c.adapter]) latest[c.adapter] = c;
  const models = Object.keys(latest).map((k) => ({ key: k, label: LABELS[k] || latest[k].model, ...latest[k] }));
  models.sort((a, b) => ["qwen", "parakeet", "diar"].indexOf(a.key) - ["qwen", "parakeet", "diar"].indexOf(b.key));

  // layer kernels: best isolated speedup vs what happened inside the whole model
  const byLayer: Record<string, Doc> = {};
  for (const k of kernels) {
    if (k.kind !== "layer") continue;
    const id = `${k.adapter}:${k.layer}`;
    const cur = (byLayer[id] ||= { adapter: k.adapter, model: LABELS[k.adapter] || k.adapter, layer: k.layer, best: 0, e2e: null, status: "layer_winner" });
    cur.best = Math.max(cur.best, Number(k.speedup) || 0);
    if (k.e2e_paired != null) {
      cur.e2e = Number(k.e2e_paired);
      cur.status = k.status;
    }
  }
  const runAdapter: Record<string, string> = Object.fromEntries(runs.map((r) => [r.run_id, r.adapter]));
  const allKept: Record<string, number> = {};
  for (const e of e2eEvents) {
    const a = runAdapter[e.run_id];
    if (a && allKept[a] === undefined && Number(e.paired_speedup) !== 1) allKept[a] = Number(e.paired_speedup);
  }

  // memory compounding on Parakeet's LSTM: first candidate and best per run
  const perRun: Record<string, Doc> = {};
  for (const e of lstmEvents) {
    const r = (perRun[e.run_id] ||= { run_id: e.run_id, first: null, best: 1, n: 0, bestAt: null });
    const sp = Number(e.speedup);
    r.n += 1;
    if (r.first === null) r.first = sp;
    if ((e.improved === true || String(e.improved) === "True") && sp > r.best) {
      r.best = sp;
      r.bestAt = r.n;
    }
  }

  const [experiments, skills, dnr, nk] = await Promise.all([count("experiments"), count("skills"), count("do_not_repeat"), count("kernels")]);
  return NextResponse.json({
    models, validations, layers: Object.values(byLayer), allKept,
    compounding: Object.values(perRun).filter((r) => r.n > 0),
    counts: { experiments, skills, do_not_repeat: dnr, kernels: nk, pipeline_runs: runs.length },
  });
}
