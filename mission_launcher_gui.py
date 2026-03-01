from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEMO_SCRIPT = ROOT / "demo_step3_mission.py"
ANIMATION_HTML = ROOT / "outputs" / "step3_mission_animation.html"
HOST = "127.0.0.1"
PORT = 8765


def _clamp_text(value: Any, default: str) -> str:
    if value is None:
        return default
    txt = str(value).strip()
    return txt if txt else default


class LauncherState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.logs: list[str] = ["Launcher ready.\n"]
        self.status = "idle"

    def append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > 8000:
                self.logs = self.logs[-8000:]

    def snapshot_logs(self, start: int) -> tuple[list[str], int]:
        with self.lock:
            s = max(0, min(start, len(self.logs)))
            chunk = self.logs[s:]
            return chunk, len(self.logs)

    def is_running(self) -> bool:
        with self.lock:
            return self.proc is not None and self.proc.poll() is None

    def start_run(self, config: dict[str, Any]) -> tuple[bool, str]:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return False, "A run is already in progress."

            cmd, seed_note = self._build_cmd(config)
            self.logs.append("\n=== Run Start ===\n")
            if seed_note:
                self.logs.append(seed_note + "\n")
            self.logs.append("$ " + " ".join(cmd) + "\n")
            self.status = "running"
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:  # noqa: BLE001
                self.status = "idle"
                self.logs.append(f"Failed to start process: {exc}\n")
                self.proc = None
                return False, str(exc)

            threading.Thread(target=self._read_output, daemon=True).start()
            return True, "started"

    def stop_run(self) -> tuple[bool, str]:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                return False, "No running process."
            self.proc.terminate()
            self.status = "stopping"
            self.logs.append("Terminate signal sent.\n")
            return True, "terminate sent"

    def _read_output(self) -> None:
        proc: subprocess.Popen[str] | None
        with self.lock:
            proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self.append_log(line)
        rc = proc.wait()
        with self.lock:
            self.logs.append(f"\n=== Run End (exit={rc}) ===\n")
            self.status = "idle"

    @staticmethod
    def _build_cmd(config: dict[str, Any]) -> tuple[list[str], str | None]:
        algorithm = _clamp_text(config.get("algorithm"), "hierarchical_v1")
        reassign_algorithm = _clamp_text(config.get("reassign_algorithm"), "potential_event")
        seed_mode = _clamp_text(config.get("seed_mode"), "random")
        seed = _clamp_text(config.get("seed"), "42")
        scenario = _clamp_text(config.get("scenario_profile"), "random")
        coverage_goal = _clamp_text(config.get("coverage_goal"), "0.90")
        handling_goal = _clamp_text(config.get("handling_goal"), "1.00")
        stage1_steps = _clamp_text(config.get("stage1_steps"), "420")
        track_interval = _clamp_text(config.get("track_interval"), "4")
        visualize = bool(config.get("visualize", True))
        animate = bool(config.get("animate", True))
        extreme_event = bool(config.get("extreme_event", True))

        cmd = [
            sys.executable,
            str(DEMO_SCRIPT),
            "--algorithm",
            algorithm,
            "--reassign-algorithm",
            reassign_algorithm,
            "--scenario-profile",
            scenario,
            "--coverage-goal",
            coverage_goal,
            "--handling-goal",
            handling_goal,
            "--stage1-steps",
            stage1_steps,
            "--track-interval",
            track_interval,
        ]
        seed_note: str | None = None
        if seed_mode == "fixed":
            cmd.extend(["--seed-mode", "fixed"])
            cmd.extend(["--seed", seed])
        else:
            # Pick a fresh seed per click so random mode is explicit and reproducible.
            picked = random.SystemRandom().randint(1, 2_000_000_000)
            seed_note = f"[launcher] random mode picked seed={picked}"
            cmd.extend(["--seed-mode", "fixed", "--seed", str(picked)])
        if not visualize:
            cmd.append("--no-visualize")
        if not animate:
            cmd.append("--no-animate")
        if not extreme_event:
            cmd.append("--no-extreme-event")
        return cmd, seed_note


STATE = LauncherState()


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mission Launcher</title>
  <style>
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #edf6ff 0%, #dceeff 100%);
      color: #0f172a;
    }
    .wrap { max-width: 1320px; margin: 14px auto; padding: 0 10px; }
    .card {
      background: rgba(255,255,255,0.9);
      border: 1px solid #9db4cc;
      border-radius: 12px;
      box-shadow: 0 10px 22px rgba(15,23,42,0.09);
      padding: 10px;
    }
    .grid {
      display: grid;
      grid-template-columns: 370px 1fr;
      gap: 10px;
    }
    .panel {
      border: 1px solid #c8d6e5;
      border-radius: 10px;
      background: #f8fbff;
      padding: 10px;
    }
    .row { margin-bottom: 8px; }
    label { display: block; font-size: 12px; color: #334155; margin-bottom: 4px; }
    select, input[type="text"], input[type="number"] {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #9ab0c8;
      border-radius: 7px;
      padding: 6px 8px;
      font-size: 13px;
      background: #fff;
    }
    .check { display: flex; gap: 12px; align-items: center; font-size: 13px; }
    .check label { margin: 0; display: inline; color: #0f172a; }
    .btns { display: flex; gap: 8px; margin-top: 10px; }
    button {
      border: 1px solid #88a2bc;
      border-radius: 8px;
      background: #fff;
      padding: 7px 10px;
      font-size: 13px;
      cursor: pointer;
    }
    #status { font-size: 13px; color: #1e293b; margin-top: 8px; }
    #logs {
      width: 100%;
      height: 640px;
      box-sizing: border-box;
      border: 1px solid #93a9c0;
      border-radius: 10px;
      background: #0f172a;
      color: #dbeafe;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      overflow: auto;
      white-space: pre-wrap;
    }
    @media (max-width: 1080px) {
      .grid { grid-template-columns: 1fr; }
      #logs { height: 420px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="grid">
        <div class="panel">
          <div class="row">
            <label>Planning Algorithm</label>
            <select id="algorithm">
              <option value="hierarchical_v1">hierarchical_v1</option>
            </select>
          </div>
          <div class="row">
            <label>Reassign Algorithm</label>
            <select id="reassign_algorithm">
              <option value="potential_event" selected>potential_event</option>
              <option value="heuristic">heuristic</option>
              <option value="auction_cbba">auction_cbba</option>
            </select>
          </div>
          <div class="row">
            <label>Scenario Profile</label>
            <select id="scenario_profile">
              <option value="high-coupling">high-coupling</option>
              <option value="clustered">clustered</option>
              <option value="split-corners">split-corners</option>
              <option value="random" selected>random</option>
            </select>
          </div>
          <div class="row">
            <label>Seed Mode</label>
            <select id="seed_mode">
              <option value="random">random</option>
              <option value="fixed">fixed</option>
            </select>
          </div>
          <div class="row">
            <label>Seed</label>
            <input id="seed" type="number" value="42" />
          </div>
          <div class="row">
            <label>Coverage Goal</label>
            <input id="coverage_goal" type="text" value="0.90" />
          </div>
          <div class="row">
            <label>Handling Goal</label>
            <input id="handling_goal" type="text" value="1.00" />
          </div>
          <div class="row">
            <label>Stage1 Steps</label>
            <input id="stage1_steps" type="number" value="420" />
          </div>
          <div class="row">
            <label>Track Interval</label>
            <input id="track_interval" type="number" value="4" />
          </div>
          <div class="row check">
            <input id="visualize" type="checkbox" checked />
            <label for="visualize">Export SVG</label>
            <input id="animate" type="checkbox" checked />
            <label for="animate">Export Animation HTML</label>
            <input id="extreme_event" type="checkbox" checked />
            <label for="extreme_event">Enable Extreme Event</label>
          </div>
          <div class="btns">
            <button id="runBtn">Run</button>
            <button id="stopBtn">Stop</button>
            <button id="openBtn">Open Animation</button>
          </div>
          <div id="status">Status: idle</div>
        </div>
        <div class="panel">
          <div id="logs"></div>
        </div>
      </div>
    </div>
  </div>
  <script>
    let logCursor = 0;
    const logs = document.getElementById("logs");
    const statusEl = document.getElementById("status");
    const seedMode = document.getElementById("seed_mode");
    const seedInput = document.getElementById("seed");

    function setSeedEnabled() {
      seedInput.disabled = seedMode.value !== "fixed";
    }
    seedMode.addEventListener("change", setSeedEnabled);
    setSeedEnabled();

    function payload() {
      return {
        algorithm: document.getElementById("algorithm").value,
        reassign_algorithm: document.getElementById("reassign_algorithm").value,
        scenario_profile: document.getElementById("scenario_profile").value,
        seed_mode: document.getElementById("seed_mode").value,
        seed: document.getElementById("seed").value,
        coverage_goal: document.getElementById("coverage_goal").value,
        handling_goal: document.getElementById("handling_goal").value,
        stage1_steps: document.getElementById("stage1_steps").value,
        track_interval: document.getElementById("track_interval").value,
        visualize: document.getElementById("visualize").checked,
        animate: document.getElementById("animate").checked,
        extreme_event: document.getElementById("extreme_event").checked
      };
    }

    async function post(url, data) {
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data || {})
      });
      return r.json();
    }

    async function poll() {
      try {
        const r = await fetch("/status?from=" + logCursor);
        const data = await r.json();
        statusEl.textContent = "Status: " + data.status;
        if (Array.isArray(data.logs) && data.logs.length > 0) {
          logs.textContent += data.logs.join("");
          logs.scrollTop = logs.scrollHeight;
        }
        logCursor = data.next;
      } catch (e) {
        statusEl.textContent = "Status: disconnected";
      }
      setTimeout(poll, 700);
    }

    document.getElementById("runBtn").onclick = async () => {
      const data = await post("/run", payload());
      if (!data.ok) {
        logs.textContent += "\\nRun failed: " + data.message + "\\n";
        logs.scrollTop = logs.scrollHeight;
      }
    };
    document.getElementById("stopBtn").onclick = async () => {
      await post("/stop", {});
    };
    document.getElementById("openBtn").onclick = async () => {
      const data = await post("/open-animation", {});
      if (!data.ok) {
        logs.textContent += "\\n" + data.message + "\\n";
        logs.scrollTop = logs.scrollHeight;
      }
    };

    poll();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "MissionLauncherHTTP/1.0"

    def _json_response(self, obj: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text_response(self, content: str, content_type: str = "text/html; charset=utf-8") -> None:
        data = content.encode("utf-8")
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._text_response(HTML)
            return
        if parsed.path == "/status":
            query = urllib.parse.parse_qs(parsed.query)
            start = 0
            try:
                start = int(query.get("from", ["0"])[0])
            except ValueError:
                start = 0
            chunk, nxt = STATE.snapshot_logs(start)
            self._json_response(
                {
                    "ok": True,
                    "status": STATE.status,
                    "running": STATE.is_running(),
                    "logs": chunk,
                    "next": nxt,
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND.value)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/run":
            data = self._read_json()
            ok, msg = STATE.start_run(data)
            status = HTTPStatus.OK if ok else HTTPStatus.CONFLICT
            self._json_response({"ok": ok, "message": msg}, status=status)
            return
        if self.path == "/stop":
            ok, msg = STATE.stop_run()
            self._json_response({"ok": ok, "message": msg})
            return
        if self.path == "/open-animation":
            if not ANIMATION_HTML.exists():
                self._json_response({"ok": False, "message": f"Animation not found: {ANIMATION_HTML}"})
                return
            webbrowser.open(ANIMATION_HTML.as_uri())
            self._json_response({"ok": True, "message": "opened"})
            return
        self.send_error(HTTPStatus.NOT_FOUND.value)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        del format, args


def main() -> None:
    if not DEMO_SCRIPT.exists():
        raise FileNotFoundError(f"Missing script: {DEMO_SCRIPT}")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    STATE.append_log(f"Launcher listening at {url}\n")
    print(f"Mission launcher started: {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
