"""
The MuJoCo window for the robot arm. The notebook starts this for you
(arm.open_viewer()) and sends it commands over localhost:

    POST /do   {"action": "tap", "color": "red"}   press a button
    POST /do   {"action": "home"}                   tuck the arm
    GET  /state                                     what it's doing
    POST /quit                                      close the window

Run it by hand if you like:  uv run python hands_viewer.py
(macOS:                      uv run mjpython hands_viewer.py)
"""
import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent.arm import COLORS, VIEWER_PORT, ArmSim

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=VIEWER_PORT)
p.add_argument("--headless", action="store_true", help="no window (for tests)")
args = p.parse_args()

sim = ArmSim(render=False, realtime=True)
jobs: "queue.Queue[tuple[dict, threading.Event, dict]]" = queue.Queue()
state = {"busy": False, "last": None, "count": 0}
running = threading.Event()
running.set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json({"status": "ok"})
        if self.path == "/state":
            return self._json(state)
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/quit":
            running.clear()
            return self._json({"ok": True})
        if self.path != "/do":
            return self._json({"error": "not found"}, 404)
        action = req.get("action")
        if action == "tap" and req.get("color") not in COLORS:
            return self._json({"error": f"color must be one of {COLORS}"}, 400)
        if action not in ("tap", "home"):
            return self._json({"error": "action must be tap or home"}, 400)
        done, result = threading.Event(), {}
        jobs.put((req, done, result))
        if req.get("wait", True):
            done.wait(timeout=20)
        return self._json({"ok": True, **result})


server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()


def run_jobs(is_open):
    while running.is_set() and is_open():
        try:
            req, done, result = jobs.get(timeout=0.02)
        except queue.Empty:
            sim.hold(0.02)                     # keep the physics ticking while idle
            continue
        state["busy"] = True
        t0 = time.time()
        if req["action"] == "tap":
            sim.tap(req["color"], film=False)
            state["last"] = req["color"]
            state["count"] += 1
        else:
            sim.home()
        state["busy"] = False
        result.update(action=req["action"], color=req.get("color"), seconds=round(time.time() - t0, 2))
        done.set()


if args.headless:
    print(f"arm ready (headless) on port {args.port}", flush=True)
    run_jobs(lambda: True)
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        v.cam.lookat[:] = (0.12, 0.0, 0.08)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.15, 145, -28
        sim.on_step = v.sync
        print(f"arm ready on port {args.port}", flush=True)
        run_jobs(v.is_running)
server.shutdown()
