"""Dev dashboard: watch the router route tasks and count tokens.

    python -m scripts.dashboard          # then open http://127.0.0.1:8765

Routes prompts through the real pipeline (classify -> gate -> remote contract).
"Live" mode actually calls Fireworks (needs .env.local); off by default it
shows what *would* be sent and the estimated token cost instead.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent.classify import classify
from agent.config import AgentConfig
from agent.contracts import build_contracts
from agent.gate import solve as gate_solve

ROOT = Path(__file__).resolve().parent.parent
CONFIG = AgentConfig.from_path(None)
CONTRACTS = build_contracts(CONFIG.max_tokens)
LOCK = threading.Lock()  # ponytail: one global lock; this is a single-user dev tool
_remote_client = None


def _load_env_local() -> None:
    path = ROOT / ".env.local"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _get_remote_client():
    global _remote_client
    if _remote_client is None:
        remote = importlib.import_module("agent.remote")
        _remote_client = remote.RemoteClient(
            timeout=float(CONFIG.remote.get("timeout_seconds", 25)),
            retries=int(CONFIG.remote.get("retries", 2)),
            temperature=float(CONFIG.remote.get("temperature", 0)),
            usd_per_mtok=float(CONFIG.remote.get("usd_per_mtok", 0) or 0),
        )
    return _remote_client


def _ledger() -> dict:
    if _remote_client is None:
        return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "requests": 0, "estimated_usd": 0.0, "entries": []}
    return _remote_client.ledger.as_dict()


def route_prompt(task_id: str, prompt: str, live: bool) -> dict:
    classification = asyncio.run(classify(prompt, None))
    category, confidence = classification.category, classification.confidence
    result = {"task_id": task_id, "prompt": prompt, "category": category,
              "confidence": round(confidence, 2), "reason": classification.reason,
              "tokens": 0}
    answer = gate_solve(category, prompt) if confidence >= 0.6 else None
    if answer is not None:
        result.update(source="gate", answer=answer)
        return result
    contract = CONTRACTS[category]
    remote_prompt = contract.remote_prompt(prompt)
    estimated = max(1, len(remote_prompt.split())) + contract.max_tokens
    result.update(remote_prompt=remote_prompt, max_tokens=contract.max_tokens,
                  estimated_tokens=estimated)
    if not live:
        result.update(source="deferred", answer="")
        return result
    remote = importlib.import_module("agent.remote")
    client = _get_remote_client()
    call = remote.RemoteCall(task_id=task_id, category=category,
                             prompt=remote_prompt, max_tokens=contract.max_tokens)
    try:
        payload = asyncio.run(client.complete(call))
    except Exception as exc:
        result.update(source="error", answer="", error=str(exc))
        return result
    entry = client.ledger.entries[-1]
    result.update(source="remote", answer=contract.assemble(prompt, payload),
                  tokens=entry["total_tokens"], model=entry["model"])
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/dataset":
            try:
                data = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                data = []
            self._json({"tasks": [
                {"id": t.get("id"), "category": t.get("category"),
                 "prompt": t.get("prompt"), "expected": t.get("expected_answer")}
                for t in data if isinstance(t, dict)
            ]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/route":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        prompt = str(body.get("prompt", "")).strip()
        if not prompt:
            self._json({"error": "empty prompt"}, 400)
            return
        live = bool(body.get("live", False))
        if live and not os.environ.get("FIREWORKS_BASE_URL"):
            self._json({"error": "live mode needs FIREWORKS_BASE_URL (put creds in .env.local)"}, 400)
            return
        with LOCK:
            result = route_prompt(str(body.get("task_id") or "adhoc"), prompt, live)
            result["ledger"] = _ledger()
        self._json(result)


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Floor-CL Router Dashboard</title>
<style>
:root {
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --gate: #2a78d6; --local: #1baf7a; --remote: #eda100; --deferred: #898781;
  --error: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --gate: #3987e5; --local: #199e70; --remote: #c98500; --error: #d03b3b;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 16px 24px 0; display: flex; align-items: baseline; gap: 12px; }
h1 { font-size: 17px; margin: 0; }
header .sub { color: var(--ink2); font-size: 13px; }
main { padding: 16px 24px 40px; display: grid; gap: 16px;
  grid-template-columns: minmax(280px, 360px) 1fr; align-items: start; }
.card { background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; }
.card h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
  color: var(--muted); margin: 0 0 10px; font-weight: 600; }
.tiles { grid-column: 1 / -1; display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.tile .v { font-size: 26px; font-weight: 650; }
.tile .l { color: var(--ink2); font-size: 12px; margin-top: 2px; }
textarea { width: 100%; min-height: 64px; resize: vertical; border-radius: 6px;
  border: 1px solid var(--grid); background: var(--page); color: var(--ink);
  padding: 8px; font: inherit; }
button { font: inherit; border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface); color: var(--ink); padding: 6px 12px; cursor: pointer; }
button:hover { border-color: var(--muted); }
button.primary { background: var(--gate); border-color: var(--gate); color: #fff; }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
label.live { display: inline-flex; gap: 6px; align-items: center;
  color: var(--ink2); font-size: 13px; }
#tasklist { max-height: 46vh; overflow: auto; margin-top: 8px; }
.task { padding: 6px 8px; border-radius: 6px; cursor: pointer; display: flex;
  gap: 8px; align-items: baseline; }
.task:hover { background: var(--page); }
.task .id { color: var(--muted); font-size: 12px; white-space: nowrap; }
.task .p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th { text-align: left; color: var(--muted); font-size: 12px; font-weight: 600;
  border-bottom: 1px solid var(--grid); padding: 4px 8px; }
td { padding: 6px 8px; border-bottom: 1px solid var(--grid);
  vertical-align: top; }
td.num, th.num { text-align: right; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin-right: 6px; vertical-align: baseline; }
.src-gate .dot { background: var(--gate); } .src-local .dot { background: var(--local); }
.src-remote .dot { background: var(--remote); } .src-deferred .dot { background: var(--deferred); }
.src-error .dot { background: var(--error); }
.ans { color: var(--ink2); max-width: 420px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.cat { color: var(--muted); font-size: 12px; }
.mix { display: flex; height: 22px; border-radius: 4px; overflow: hidden;
  gap: 2px; background: var(--surface); margin-bottom: 8px; }
.mix div { min-width: 2px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; color: var(--ink2);
  font-size: 13px; }
.bars .bar-row { display: grid; grid-template-columns: 90px 1fr 60px; gap: 8px;
  align-items: center; margin: 4px 0; font-size: 13px; }
.bars .track { height: 14px; }
.bars .fill { height: 100%; background: var(--gate); border-radius: 0 4px 4px 0;
  min-width: 2px; }
.bars .val { text-align: right; font-variant-numeric: tabular-nums; }
.bars .name { color: var(--ink2); overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
#results-card { overflow-x: auto; }
.empty { color: var(--muted); padding: 12px 0; }
</style>
</head>
<body>
<header>
  <h1>Floor-CL Router</h1>
  <span class="sub">classify → gate (free) → remote (tokens). Live mode spends real Fireworks tokens.</span>
</header>
<main>
  <div class="tiles">
    <div class="card tile"><div class="v" id="t-tokens">0</div><div class="l">Fireworks tokens (session)</div></div>
    <div class="card tile"><div class="v" id="t-usd">$0.0000</div><div class="l">Estimated cost</div></div>
    <div class="card tile"><div class="v" id="t-requests">0</div><div class="l">Remote requests</div></div>
    <div class="card tile"><div class="v" id="t-free">0</div><div class="l">Answered free (gate)</div></div>
    <div class="card tile"><div class="v" id="t-saved">0</div><div class="l">Tokens avoided (est.)</div></div>
  </div>

  <div class="card">
    <h2>Route a prompt</h2>
    <textarea id="prompt" placeholder="Type a prompt, or click a dataset task below…"></textarea>
    <div class="row">
      <button class="primary" id="route">Route it</button>
      <button id="route-all">Route all dataset tasks</button>
      <label class="live"><input type="checkbox" id="live"> live Fireworks call (spends tokens)</label>
    </div>
    <h2 style="margin-top:16px">Dataset tasks</h2>
    <div id="tasklist" class="empty">loading…</div>
  </div>

  <div style="display:grid; gap:16px">
    <div class="card">
      <h2>Where answers came from</h2>
      <div class="mix" id="mix"></div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="card bars" id="bars-card" hidden>
      <h2>Tokens per remote call</h2>
      <div id="bars"></div>
    </div>
    <div class="card" id="results-card">
      <h2>Routed tasks</h2>
      <table>
        <thead><tr><th>Task</th><th>Category</th><th>Source</th>
          <th class="num">Tokens</th><th>Answer</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
      <div class="empty" id="rows-empty">Nothing routed yet.</div>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const SOURCES = ["gate", "local", "remote", "deferred", "error"];
const LABELS = {gate: "gate (free)", local: "local (free)", remote: "remote",
                deferred: "deferred (dry run)", error: "error"};
const results = [];
let busy = false;

fetch("/api/dataset").then(r => r.json()).then(({tasks}) => {
  window.dataset = tasks || [];
  const el = $("tasklist");
  el.classList.remove("empty");
  el.innerHTML = "";
  for (const t of window.dataset) {
    const d = document.createElement("div");
    d.className = "task";
    d.innerHTML = `<span class="id">${t.id}</span><span class="p"></span>`;
    d.querySelector(".p").textContent = t.prompt;
    d.title = t.prompt;
    d.onclick = () => { $("prompt").value = t.prompt; routeOne(t.id, t.prompt); };
    el.appendChild(d);
  }
  if (!window.dataset.length) { el.textContent = "no dataset.json found"; el.classList.add("empty"); }
  const auto = Number(new URLSearchParams(location.search).get("auto") || 0);
  (async () => { for (const t of window.dataset.slice(0, auto)) await routeOne(t.id, t.prompt); })();
});

async function routeOne(taskId, prompt) {
  if (busy || !prompt.trim()) return;
  busy = true;
  try {
    const r = await fetch("/api/route", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({task_id: taskId, prompt, live: $("live").checked})});
    const data = await r.json();
    if (!r.ok) { alert(data.error || "request failed"); return; }
    results.push(data);
    render(data.ledger);
  } finally { busy = false; }
}

$("route").onclick = () => routeOne("adhoc-" + (results.length + 1), $("prompt").value);
$("route-all").onclick = async () => {
  if ($("live").checked && !confirm("Live mode: this calls Fireworks for every non-gate task and spends real tokens. Continue?")) return;
  for (const t of (window.dataset || [])) {
    while (busy) await new Promise(res => setTimeout(res, 50));
    await routeOne(t.id, t.prompt);
  }
};

function render(ledger) {
  // stat tiles
  $("t-tokens").textContent = ledger.total_tokens.toLocaleString();
  $("t-usd").textContent = "$" + ledger.estimated_usd.toFixed(4);
  $("t-requests").textContent = ledger.requests;
  const free = results.filter(r => r.source === "gate" || r.source === "local");
  $("t-free").textContent = free.length;
  $("t-saved").textContent = free.reduce((s, r) => s + (r.estimated_tokens ||
    Math.max(1, r.prompt.split(/\s+/).length) + 40), 0).toLocaleString();

  // source mix stacked bar + legend
  const counts = Object.fromEntries(SOURCES.map(s => [s, 0]));
  for (const r of results) counts[r.source] = (counts[r.source] || 0) + 1;
  $("mix").innerHTML = SOURCES.filter(s => counts[s]).map(s =>
    `<div class="src-${s}" style="flex:${counts[s]};background:var(--${s === "error" ? "error" : s})" title="${LABELS[s]}: ${counts[s]}"></div>`
  ).join("") || `<div style="flex:1;background:var(--grid)"></div>`;
  $("legend").innerHTML = SOURCES.filter(s => counts[s]).map(s =>
    `<span class="src-${s}"><span class="dot"></span>${LABELS[s]} · ${counts[s]}</span>`
  ).join("") || "route something to see the mix";

  // tokens per remote call
  const entries = ledger.entries;
  $("bars-card").hidden = !entries.length;
  if (entries.length) {
    const max = Math.max(...entries.map(e => e.total_tokens));
    $("bars").innerHTML = entries.map(e =>
      `<div class="bar-row" title="${e.model} — prompt ${e.prompt_tokens} + completion ${e.completion_tokens}">
        <span class="name">${e.task_id}</span>
        <span class="track"><span class="fill" style="display:block;width:${Math.max(2, 100 * e.total_tokens / max)}%"></span></span>
        <span class="val">${e.total_tokens}</span>
      </div>`).join("");
  }

  // results table (newest first)
  $("rows-empty").style.display = results.length ? "none" : "";
  $("rows").innerHTML = results.slice().reverse().map(r => {
    const tokens = r.source === "remote" ? r.tokens :
      (r.estimated_tokens ? `~${r.estimated_tokens} est.` : "0");
    const answer = r.source === "deferred"
      ? "→ would send " + (r.estimated_tokens || "?") + " tokens to Fireworks"
      : (r.error || r.answer || "");
    return `<tr>
      <td>${r.task_id}</td>
      <td class="cat">${r.category}<br>conf ${r.confidence}</td>
      <td class="src-${r.source}"><span class="dot"></span>${LABELS[r.source] || r.source}</td>
      <td class="num">${tokens}</td>
      <td class="ans" title="${(r.answer || r.error || "").replace(/"/g, "&quot;")}">${answer.replace(/</g, "&lt;")}</td>
    </tr>`;
  }).join("");
}
render({total_tokens: 0, estimated_usd: 0, requests: 0, entries: []});
</script>
</body>
</html>
"""


def main() -> None:
    _load_env_local()
    port = int(os.environ.get("DASHBOARD_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"router dashboard: http://127.0.0.1:{port}"
          + ("" if os.environ.get("FIREWORKS_BASE_URL") else "  (no Fireworks creds — dry-run only)"))
    server.serve_forever()


if __name__ == "__main__":
    main()
