import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agent import JobAgent
from .taxonomy import PROJECT_ROOT


_AGENT: JobAgent | None = None
_LOCK = threading.Lock()


def get_agent() -> JobAgent:
    global _AGENT
    if _AGENT is None:
        with _LOCK:
            if _AGENT is None:
                _AGENT = JobAgent.from_artifacts(PROJECT_ROOT)
    return _AGENT


class Handler(BaseHTTPRequestHandler):
    server_version = "JobAgent/1.0"

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj: dict, status: int = 200) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            html = (PROJECT_ROOT / "web" / "index.html").read_bytes()
            self._send(html, "text/html; charset=utf-8")
            return
        if path == "/api/health":
            self._json({"status": "ok", "agent_loaded": _AGENT is not None})
            return
        if path == "/api/metrics":
            metrics_path = PROJECT_ROOT / "artifacts" / "metrics.json"
            if not metrics_path.exists():
                self._json({"status": "missing", "message": "metrics.json not found"}, 404)
                return
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            self._json({
                "status": "ok",
                "dataset": metrics.get("dataset"),
                "clustering": {k: metrics.get("clustering", {}).get(k) for k in ["k", "silhouette", "stability_ari_mean"]},
                "main": metrics.get("main"),
            })
            return
        self._json({"status": "not_found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json({"status": "error", "message": "invalid JSON body"}, 400)
            return
        if path == "/api/recommend":
            try:
                agent = get_agent()
                top_k = int(payload.pop("top_k", 10)) if isinstance(payload, dict) else 10
                top_k = max(1, min(20, top_k))
                response = agent.run(payload, top_k=top_k)
                self._json(response)
            except Exception as exc:
                self._json({"status": "error", "message": f"推荐失败：{type(exc).__name__}: {exc}"}, 500)
            return
        if path == "/api/feedback":
            try:
                agent = get_agent()
                response = agent.record_feedback(
                    str(payload.get("session_id", "")),
                    str(payload.get("job_id", "")),
                    str(payload.get("action", "")),
                    str(payload.get("note", "")),
                )
                self._json(response)
            except Exception as exc:
                self._json({"status": "error", "message": str(exc)}, 400)
            return
        self._json({"status": "not_found"}, 404)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    get_agent()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"求职岗位推荐 Agent 已启动: http://{args.host}:{args.port}", flush=True)
    print("演示模式: http://127.0.0.1:8000/?demo=1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
