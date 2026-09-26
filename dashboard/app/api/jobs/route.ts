import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { find, insertOne, REPO_ROOT, updateOne } from "@/lib/db";

export const dynamic = "force-dynamic";

const BUILTIN = ["qwen", "parakeet", "diar"];
const HF_ID = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/;

function alive(pid?: number): boolean {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function tail(file: string, n = 30): string[] {
  if (!fs.existsSync(file)) return [];
  const lines = fs.readFileSync(file, "utf8").replace(/\r/g, "\n").split("\n");
  return lines.filter((l) => l.trim() && !/MallocStackLogging|Fetching \d+ files|it\/s\]/.test(l)).slice(-n);
}

function llmKey(modelId: string): string {
  const name = modelId.split("/").pop()!.toLowerCase();
  return "llm-" + name.replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function num(v: any, lo: number, hi: number, dflt: number): number {
  const x = Number(v);
  return Number.isFinite(x) ? Math.min(hi, Math.max(lo, x)) : dflt;
}

export async function GET() {
  const jobs = await find("jobs", {}, { sort: { created: -1 }, limit: 30 });
  for (const j of jobs) {
    const lines = tail(j.log);
    j.tail = lines;
    if (j.status === "running" && !alive(j.pid)) {
      const exit = lines.map((l) => l.match(/JOB_EXIT (\d+)/)).find(Boolean);
      j.status = exit && exit[1] === "0" ? "done" : "failed";
      await updateOne("jobs", { job_id: j.job_id }, { status: j.status, finished: new Date().toISOString() });
    }
  }
  return NextResponse.json({ jobs });
}

export async function POST(req: Request) {
  const b = await req.json();
  const model = String(b.model || "");
  const modelId = String(b.model_id || "").trim();
  if (b.hardware !== "local") return NextResponse.json({ error: "Only this Mac (MLX/Metal) is configured. The Modal CUDA backend is not set up." }, { status: 400 });
  if (!BUILTIN.includes(model) && model !== "llm") return NextResponse.json({ error: "unknown model" }, { status: 400 });
  if (model === "llm" && !HF_ID.test(modelId)) return NextResponse.json({ error: "Enter a Hugging Face model id like Qwen/Qwen3-1.7B" }, { status: 400 });

  const running = (await find("jobs", { status: "running" })).filter((j) => alive(j.pid));
  if (running.length && !b.force) {
    return NextResponse.json({ error: `A job is already running (${running[0].label}). Parallel runs share the GPU and make timings noisy.`, running: true }, { status: 409 });
  }

  const args = [
    "run", "--model", model,
    "--top-layers", String(num(b.top_layers, 1, 8, 3)),
    "--max-layer-iters", String(num(b.max_layer_iters, 1, 20, 6)),
    "--layer-patience", String(num(b.layer_patience, 1, 10, 3)),
    "--layer-minutes", String(num(b.layer_minutes, 1, 120, 20)),
    "--max-config-iters", String(num(b.max_config_iters, 1, 20, 6)),
    "--config-patience", String(num(b.config_patience, 1, 10, 3)),
    "--config-minutes", String(num(b.config_minutes, 1, 120, 20)),
    "--hardware", "local",
  ];
  if (b.target_speedup) args.push("--target-speedup", String(num(b.target_speedup, 1.01, 10, 2)));
  if (model === "llm") args.push("--model-id", modelId);
  if (b.quality_budget_rel !== undefined && b.quality_budget_rel !== "") args.push("--quality-budget-rel", String(num(b.quality_budget_rel, 0, 0.2, 0.01)));
  if (b.quality_budget_abs !== undefined && b.quality_budget_abs !== "") args.push("--quality-budget-abs", String(num(b.quality_budget_abs, 0, 0.2, 0.005)));

  const jobId = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14) + "_" + (model === "llm" ? llmKey(modelId) : model);
  const logDir = path.join(REPO_ROOT, "out", "jobs");
  fs.mkdirSync(logDir, { recursive: true });
  const log = path.join(logDir, `${jobId}.log`);
  const visai = path.join(REPO_ROOT, ".venv", "bin", "visai");
  const quoted = [visai, ...args].map((a) => `'${a.replace(/'/g, "'\\''")}'`).join(" ");
  const env = { ...process.env };
  delete env.VISAI_DASHBOARD_LOCAL;
  const child = spawn("/bin/sh", ["-c", `${quoted} > '${log}' 2>&1; echo "JOB_EXIT $?" >> '${log}'`], {
    cwd: REPO_ROOT, detached: true, stdio: "ignore", env,
  });
  child.unref();

  const doc = {
    job_id: jobId, created: new Date().toISOString(), status: "running", pid: child.pid, log,
    model, model_id: model === "llm" ? modelId : null, adapter_key: model === "llm" ? llmKey(modelId) : model,
    label: model === "llm" ? modelId : model, hardware: "this Mac (MLX/Metal)", command: `visai ${args.join(" ")}`,
  };
  await insertOne("jobs", doc);
  return NextResponse.json(doc);
}

export async function DELETE(req: Request) {
  const id = new URL(req.url).searchParams.get("job_id");
  const job = (await find("jobs", { job_id: id }))[0];
  if (!job) return NextResponse.json({ error: "no such job" }, { status: 404 });
  try {
    process.kill(-job.pid, "SIGTERM");
  } catch {
    try { process.kill(job.pid, "SIGTERM"); } catch { /* already gone */ }
  }
  await updateOne("jobs", { job_id: id }, { status: "stopped", finished: new Date().toISOString() });
  return NextResponse.json({ ok: true });
}
