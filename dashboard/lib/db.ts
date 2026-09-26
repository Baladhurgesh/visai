import fs from "node:fs";
import path from "node:path";
import { MongoClient, type Db } from "mongodb";

type Doc = Record<string, any>;

const ROOT = path.resolve(process.cwd(), "..");

function readDotEnv(): Record<string, string> {
  const out: Record<string, string> = {};
  const file = path.join(ROOT, ".env");
  if (!fs.existsSync(file)) return out;
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m) out[m[1]] = m[2].replace(/^["']|["']$/g, "").trim();
  }
  return out;
}

const env = { ...readDotEnv(), ...Object.fromEntries(Object.entries(process.env).filter(([, v]) => v !== undefined)) } as Record<string, string>;
const URI = (process.env.VISAI_DASHBOARD_LOCAL ? "" : env.MONGODB_URI || "").trim();
const DB_NAME = env.VISAI_DB || "visai";

let clientPromise: Promise<MongoClient> | null = null;

async function atlas(): Promise<Db> {
  if (!clientPromise) clientPromise = new MongoClient(URI, { appName: "visai-dashboard" }).connect();
  return (await clientPromise).db(DB_NAME);
}

function readLocal(coll: string): Doc[] {
  const file = path.join(ROOT, "out", "localdb", `${coll}.jsonl`);
  if (!fs.existsSync(file)) return [];
  return fs
    .readFileSync(file, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
}

export const backend = URI ? "atlas" : "local";

export function envValue(name: string): string {
  return (env[name] || "").trim();
}

export async function count(coll: string): Promise<number> {
  if (URI) return (await atlas()).collection(coll).estimatedDocumentCount();
  return readLocal(coll).length;
}

function matches(doc: Doc, filter: Doc): boolean {
  return Object.entries(filter).every(([k, v]) => {
    const val = k.split(".").reduce((o: any, p) => (o == null ? undefined : o[p]), doc);
    if (v && typeof v === "object" && "$gt" in v) return val > v.$gt;
    if (v && typeof v === "object" && "$in" in v) return (v.$in as any[]).includes(val);
    return val === v;
  });
}

export async function find(
  coll: string,
  filter: Doc = {},
  opts: { sort?: Record<string, 1 | -1>; limit?: number; project?: Doc } = {},
): Promise<Doc[]> {
  if (URI) {
    let cur = (await atlas()).collection(coll).find(filter, { projection: { embedding: 0, ...(opts.project || {}) } });
    if (opts.sort) cur = cur.sort(opts.sort);
    if (opts.limit) cur = cur.limit(opts.limit);
    return (await cur.toArray()).map((d) => ({ ...d, _id: String(d._id) }));
  }
  let rows = readLocal(coll).filter((d) => matches(d, filter));
  if (opts.sort) {
    const [[key, dir]] = Object.entries(opts.sort);
    rows.sort((a, b) => ((a[key] ?? "") > (b[key] ?? "") ? dir : (a[key] ?? "") < (b[key] ?? "") ? -dir : 0));
  }
  if (opts.limit) rows = rows.slice(0, opts.limit);
  return rows.map(({ embedding, ...rest }) => {
    if (opts.project) for (const [k, v] of Object.entries(opts.project)) if (!v) delete (rest as Doc)[k];
    return rest;
  });
}

export async function findOne(coll: string, filter: Doc = {}, sort?: Record<string, 1 | -1>): Promise<Doc | null> {
  const rows = await find(coll, filter, { sort, limit: 1 });
  return rows[0] || null;
}
