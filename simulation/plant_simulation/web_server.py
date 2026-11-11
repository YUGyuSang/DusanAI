"""
ROS-free HTTP server for the digital-twin dashboard (Python stdlib only).

Endpoints
---------
GET  /                      index.html (static files from web_dir)
GET  /<file>                app.js, style.css, ... (path-traversal safe)
GET  /events                Server-Sent Events: one 'static' event (layout),
                            then a JSON snapshot every 1/sse_rate s
GET  /api/state             current snapshot (polling fallback)
GET  /api/static            layout / ground-truth info
GET  /api/health            {"ok": true, ...}
GET  /map.png /gas.png /rssi.png /image/gauge.jpg /image/thermal.jpg
                            binary blobs from the blob provider (404 if none)
POST /api/command           {"target": "mission"|"locomotion"|"battery", "command": "..."}
                            (Content-Type: application/json, else 415)

The server runs in a daemon thread (ThreadingHTTPServer, one thread per
client, so long-lived SSE streams do not block other requests).
"""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from dusan_common.ros_helpers import to_json

MISSION_COMMANDS = ("start", "patrol", "seek", "rth", "pause", "resume", "abort", "dock")
LOCOMOTION_COMMANDS = ("stand", "sit", "walk", "gait:trot", "gait:crawl", "gait:auto", "recover", "estop",
                       "release", "push", "reset_pose")
_PUSH_RE = re.compile(r"^push:-?\d+(\.\d+)?,-?\d+(\.\d+)?$")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8",
}
BLOB_PATHS = ("/map.png", "/gas.png", "/rssi.png", "/image/gauge.jpg", "/image/thermal.jpg")
MAX_BODY = 64 * 1024


def validate_command(target, command):
    """Return (ok, normalised_command_or_error_message)."""
    target = str(target or "").strip().lower()
    command = str(command if command is not None else "").strip()
    if target == "mission":
        c = command.lower()
        return (True, c) if c in MISSION_COMMANDS else (False, f"unknown mission command '{command}'")
    if target == "locomotion":
        c = command.lower()
        if c in LOCOMOTION_COMMANDS or _PUSH_RE.match(c):
            return True, c
        return False, f"unknown locomotion command '{command}'"
    if target == "battery":
        try:
            v = float(command)
        except ValueError:
            return False, "battery command must be a percentage"
        if not 0.0 <= v <= 100.0:
            return False, "battery percentage must be within 0..100"
        return True, f"{v:g}"
    return False, f"unknown target '{target}' (mission|locomotion|battery)"


class _Handler(BaseHTTPRequestHandler):
    server_version = "DusanTwin/1.0"

    # quiet access log (routed to the app logger at debug level)
    def log_message(self, fmt, *args):  # noqa: D401
        self.server.app.log("debug", "%s - %s" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------ helpers
    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, to_json(obj).encode("utf-8"), "application/json; charset=utf-8",
                   {"Cache-Control": "no-store"})

    # -------------------------------------------------------------- verbs
    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_GET(self):  # noqa: N802
        app = self.server.app
        path = urlparse(self.path).path
        try:
            if path == "/events":
                self._sse()
            elif path == "/api/state":
                self._send(200, app.snapshot_json(), "application/json; charset=utf-8", {"Cache-Control": "no-store"})
            elif path == "/api/static":
                self._json(200, app.static_fn())
            elif path == "/api/health":
                self._json(200, {"ok": True, "port": app.port, "clients": app.sse_clients,
                                 "uptime": round(time.time() - app.t_start, 1)})
            elif path in BLOB_PATHS:
                blob = app.blob(path.lstrip("/"))
                if blob is None:
                    self._send(404, b"not available yet", "text/plain; charset=utf-8", {"Cache-Control": "no-store"})
                else:
                    data, ctype = blob
                    self._send(200, data, ctype, {"Cache-Control": "no-store"})
            else:
                self._static(path)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):  # noqa: N802
        app = self.server.app
        path = urlparse(self.path).path
        if path != "/api/command":
            self._json(404, {"ok": False, "message": "not found"})
            return
        # JSON only: a cross-site page cannot send application/json without a
        # CORS preflight (which this server never grants), so a malicious web
        # page in the operator's browser cannot command the robot.
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._json(415, {"ok": False, "message": "Content-Type must be application/json"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            self._json(400, {"ok": False, "message": "missing or oversized body"})
            return
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._json(400, {"ok": False, "message": "body must be JSON"})
            return
        if not isinstance(body, dict):
            self._json(400, {"ok": False, "message": "body must be a JSON object"})
            return
        ok, res = validate_command(body.get("target"), body.get("command"))
        if not ok:
            self._json(400, {"ok": False, "message": res})
            return
        target = str(body.get("target")).strip().lower()
        try:
            ok, msg = app.command_fn(target, res) if app.command_fn else (False, "commands disabled")
        except Exception as exc:  # pragma: no cover - publisher failure
            ok, msg = False, f"command failed: {exc}"
        self._json(200 if ok else 503, {"ok": bool(ok), "message": msg, "target": target, "command": res})

    # ------------------------------------------------------------ static
    def _static(self, path):
        app = self.server.app
        rel = unquote(path).lstrip("/") or "index.html"
        if rel.endswith("/"):
            rel += "index.html"
        # abspath (not realpath): colcon --symlink-install installs every web file
        # as a symlink into the source tree, so resolving links would put them
        # "outside" root.  abspath still normalises '..' (no path traversal).
        root = os.path.abspath(app.web_dir) if app.web_dir else None
        if not root or not os.path.isdir(root):
            self._send(500, b"dashboard web directory not found", "text/plain; charset=utf-8")
            return
        full = os.path.abspath(os.path.join(root, rel))
        if not (full == root or full.startswith(root + os.sep)) or not os.path.isfile(full):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = CONTENT_TYPES.get(os.path.splitext(full)[1].lower(), "application/octet-stream")
        with open(full, "rb") as f:
            data = f.read()
        self._send(200, data, ctype, {"Cache-Control": "no-cache"})

    # --------------------------------------------------------------- SSE
    def _sse(self):
        app = self.server.app
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        if self.command == "HEAD":
            return
        app.client_delta(+1)
        try:
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.write(b"event: static\ndata: " + to_json(app.static_fn()).encode("utf-8") + b"\n\n")
            self.wfile.flush()
            period = 1.0 / max(0.2, app.sse_rate)
            while not app.stopping.is_set():
                self.wfile.write(b"data: " + app.snapshot_json() + b"\n\n")
                self.wfile.flush()
                if app.stopping.wait(period):
                    break
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            app.client_delta(-1)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # SO_REUSEADDR lets a restarted node rebind while old sockets sit in
    # TIME_WAIT on Linux; on Windows it would allow two servers on one port.
    allow_reuse_address = os.name != "nt"

    def handle_error(self, request, client_address):
        """Browsers drop connections all the time (tab closed, reload): do
        not print tracebacks for that, route anything else to the logger."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError)):
            return
        app = getattr(self, "app", None)
        if app is not None:
            app.log("warn", f"HTTP handler error for {client_address}: {exc!r}")
        else:  # pragma: no cover
            super().handle_error(request, client_address)


class DashboardServer:
    """Owns the ThreadingHTTPServer + its daemon thread."""

    def __init__(self, snapshot_fn, static_fn, web_dir, port=8080, host="0.0.0.0", sse_rate=5.0,
                 command_fn=None, blob_fn=None, logger=None, port_fallbacks=0):
        self.snapshot_fn = snapshot_fn
        self.static_fn = static_fn
        self.web_dir = web_dir
        self.requested_port = int(port)
        self.port = None
        self.host = host
        self.sse_rate = float(sse_rate)
        self.command_fn = command_fn
        self.blob_fn = blob_fn
        self.logger = logger
        self.port_fallbacks = int(port_fallbacks)
        self.stopping = threading.Event()
        self.sse_clients = 0
        self.t_start = time.time()
        self._lock = threading.Lock()
        self._snap_cache = (0.0, b"{}")
        self.httpd = None
        self.thread = None

    def log(self, level, msg):
        if self.logger is not None:
            try:
                self.logger(level, msg)
            except Exception:
                pass

    def client_delta(self, d):
        with self._lock:
            self.sse_clients += d

    def snapshot_json(self):
        """Snapshot encoded once per SSE period and shared by all clients."""
        now = time.monotonic()
        with self._lock:
            t, data = self._snap_cache
            if now - t < 0.5 / max(0.2, self.sse_rate):
                return data
        data = to_json(self.snapshot_fn()).encode("utf-8")
        with self._lock:
            self._snap_cache = (now, data)
        return data

    def blob(self, name):
        if self.blob_fn is None:
            return None
        try:
            return self.blob_fn(name)
        except Exception as exc:
            self.log("warn", f"blob {name} failed: {exc}")
            return None

    def start(self):
        """Bind (trying fallback ports) and serve in a daemon thread. Returns the port."""
        last_exc = None
        ports = [self.requested_port] if self.requested_port == 0 else \
            [self.requested_port + k for k in range(self.port_fallbacks + 1)]
        for p in ports:
            try:
                httpd = _Server((self.host, p), _Handler)
            except OSError as exc:
                last_exc = exc
                self.log("warn", f"port {p} unavailable: {exc}")
                continue
            httpd.app = self
            self.httpd = httpd
            self.port = httpd.server_address[1]
            self.thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.5},
                                           name="dashboard-http", daemon=True)
            self.thread.start()
            return self.port
        raise OSError(f"could not bind the dashboard to ports {ports}: {last_exc}")

    def stop(self):
        self.stopping.set()
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
            self.httpd = None
