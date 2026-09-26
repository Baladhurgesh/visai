import { find } from "@/lib/db";

export const dynamic = "force-dynamic";

// Server-sent events: pushes new `events` docs for a run (or all runs) as the agent works.
export async function GET(req: Request) {
  const url = new URL(req.url);
  const runId = url.searchParams.get("run_id");
  let since = url.searchParams.get("since") || "";
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      let open = true;
      req.signal.addEventListener("abort", () => {
        open = false;
      });
      while (open) {
        try {
          const filter: Record<string, any> = runId ? { run_id: runId } : {};
          if (since) filter.ts = { $gt: since };
          const rows = await find("events", filter, { sort: { ts: 1 }, limit: 200 });
          for (const r of rows) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(r)}\n\n`));
            if (r.ts > since) since = r.ts;
          }
          controller.enqueue(encoder.encode(`: ping\n\n`));
        } catch (e) {
          controller.enqueue(encoder.encode(`event: error\ndata: ${JSON.stringify(String(e))}\n\n`));
        }
        await new Promise((r) => setTimeout(r, 1500));
      }
      controller.close();
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache, no-transform", Connection: "keep-alive" },
  });
}
