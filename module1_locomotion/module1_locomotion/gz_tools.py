"""
Non-blocking wrappers around the ``gz`` command line tool (ROS free).

* push      : persistent wrench on ``spot::base_link`` (ApplyLinkWrench system,
              topic /world/<world>/wrench/persistent), cleared after ``duration``
              via /world/<world>/wrench/clear;
* clear     : remove the persistent wrench;
* reset_pose: /world/<world>/set_pose service (UserCommands system) - upright at
              the given x, y, z with the given yaw; the caller is told when the
              service has answered (``on_done``).

Every command runs in a daemon worker thread with a subprocess timeout, so the
100 Hz control loop is never blocked; at most one job *of each kind* runs at a
time (e.g. a second push while a push is running is rejected and reported, but
a reset is still accepted during a push).  The command lines follow
DESIGN_SPEC §3 and are built by pure functions that are unit tested.

Timing: every ``gz topic`` / ``gz service`` call is a new gz CLI process that
sends its message only after the CLI start-up and the gz-transport discovery
(typically 0.3-1 s, more on a cold start).  Anything timed relative to such a
command must therefore be timed from the *start* of the process (push) or from
its *completion* (reset_pose), never from the request.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import threading
import time
from typing import Callable, List, Optional, Sequence

from dusan_common import robot_model as rm

Runner = Callable[[Sequence[str], float], "tuple[int, str]"]

# longest wall-clock wait for ``duration`` on the simulation clock, as a multiple
# of ``duration`` (a paused simulation / missing /clock must not keep the force on)
SIM_WAIT_WALL_FACTOR = 10.0
SIM_WAIT_WALL_MIN = 1.0
POLL_PERIOD = 0.005


def _fmt(v: float) -> str:
    return f"{float(v):.6g}"


def build_wrench_cmd(world: str, entity: str, force, torque=(0.0, 0.0, 0.0),
                     entity_type: str = "LINK", gz: str = "gz") -> List[str]:
    """gz topic command applying a persistent wrench to a link (or model)."""
    fx, fy, fz = force
    tx, ty, tz = torque
    payload = (f'entity: {{name: "{entity}", type: {entity_type}}}, '
               f'wrench: {{force: {{x: {_fmt(fx)}, y: {_fmt(fy)}, z: {_fmt(fz)}}}, '
               f'torque: {{x: {_fmt(tx)}, y: {_fmt(ty)}, z: {_fmt(tz)}}}}}')
    return [gz, "topic", "-t", f"/world/{world}/wrench/persistent", "-m", "gz.msgs.EntityWrench",
            "-p", payload]


def build_clear_cmd(world: str, entity: str, entity_type: str = "LINK", gz: str = "gz") -> List[str]:
    """gz topic command clearing the persistent wrench of an entity."""
    payload = f'name: "{entity}", type: {entity_type}'
    return [gz, "topic", "-t", f"/world/{world}/wrench/clear", "-m", "gz.msgs.Entity", "-p", payload]


def yaw_to_quat(yaw: float):
    """Upright orientation with heading ``yaw`` -> quaternion (x, y, z, w)."""
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def build_set_pose_cmd(world: str, model: str, x: float, y: float, z: float, yaw: float,
                       timeout_ms: int = 2000, gz: str = "gz") -> List[str]:
    """gz service command placing the model upright at (x, y, z, yaw)."""
    qx, qy, qz, qw = yaw_to_quat(yaw)
    req = (f'name: "{model}", position: {{x: {_fmt(x)}, y: {_fmt(y)}, z: {_fmt(z)}}}, '
           f'orientation: {{x: {_fmt(qx)}, y: {_fmt(qy)}, z: {_fmt(qz)}, w: {_fmt(qw)}}}')
    return [gz, "service", "-s", f"/world/{world}/set_pose", "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean", "--timeout", str(int(timeout_ms)), "--req", req]


def set_pose_reply_ok(output: str) -> bool:
    """True unless the ``gz service`` output reports a failure.

    The CLI prints the reply (``data: true``) on success, ``Service call timed
    out`` when nobody answered (no UserCommands system / no discovery) and
    ``Service call failed`` when the service returned false - the exit code is
    0 in all three cases."""
    low = (output or "").lower()
    return not any(w in low for w in ("false", "timed out", "timeout", "fail"))


def _default_runner(cmd: Sequence[str], timeout: float):
    """Run a command, return (returncode, combined output); never raises."""
    try:
        p = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, f"timeout after {timeout:.1f}s"
    except (OSError, ValueError) as e:
        return -2, str(e)


class GzTools:
    """Asynchronous gz CLI helper (push / clear wrench / reset pose).

    ``clock`` (optional) returns the simulation time [s] (e.g. the ROS clock of
    a node with use_sim_time); the push duration is then measured in simulation
    time, so the impulse does not shrink with a real-time factor below 1.
    Without it wall time is used."""

    def __init__(self, world: str = rm.WORLD_NAME, model: str = rm.ROBOT_NAME, link: str = "base_link",
                 gz_exe: str = "gz", timeout: float = 3.0, runner: Optional[Runner] = None,
                 logger: Optional[Callable[[str], None]] = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Optional[Callable[[], float]] = None):
        self.world = world
        self.model = model
        self.link = link
        self.gz = gz_exe
        self.timeout = float(timeout)
        self.runner = runner or _default_runner
        self.log = logger or (lambda msg: None)
        self._sleep = sleep
        self.clock = clock
        self._monotonic = time.monotonic
        self._lock = threading.Lock()
        self._threads: dict = {}
        self.history: List[dict] = []        # finished jobs: {"job", "cmds", "ok", "output", ...}

    # ------------------------------------------------------------- helpers
    @property
    def entity(self) -> str:
        return f"{self.model}::{self.link}"

    def available(self) -> bool:
        """True if the gz executable is on PATH (or a custom runner is used)."""
        return self.runner is not _default_runner or shutil.which(self.gz) is not None

    def is_busy(self, name: Optional[str] = None) -> bool:
        """True while a job (of kind ``name``, or any) is running."""
        threads = [self._threads.get(name)] if name else list(self._threads.values())
        return any(t is not None and t.is_alive() for t in threads)

    @property
    def busy(self) -> bool:
        return self.is_busy()

    def wait(self, timeout: float = 10.0) -> None:
        """Join the running jobs (tests / shutdown)."""
        for t in list(self._threads.values()):
            if t is not None:
                t.join(timeout)

    def _run(self, cmd: Sequence[str]):
        rc, out = self.runner(cmd, self.timeout)
        return rc == 0, out

    def _clock_now(self) -> Optional[float]:
        """Simulation time from ``clock`` (None without a usable clock)."""
        if self.clock is None:
            return None
        try:
            v = float(self.clock())
        except Exception:        # e.g. node already destroyed during shutdown
            return None
        return v if math.isfinite(v) else None

    def _run_async(self, key: str, cmd: Sequence[str], results: dict):
        """Start ``cmd`` in its own thread -> (wall start, sim start, thread).
        ``results[key]`` = (ok, output, wall seconds) once it has finished."""
        def target():
            t0 = self._monotonic()
            try:
                ok, out = self._run(cmd)
            except Exception as e:      # a custom runner must not kill the push job
                ok, out = False, f"{key}: {e}"
            results[key] = (ok, out, self._monotonic() - t0)

        th = threading.Thread(target=target, name=f"gz_push_{key}", daemon=True)
        t_wall, t_sim = self._monotonic(), self._clock_now()
        th.start()
        return t_wall, t_sim, th

    def _wait_since(self, t0_wall: float, t0_sim: Optional[float], duration: float) -> None:
        """Sleep until ``duration`` has passed since (t0_wall, t0_sim): on the
        simulation clock when there is one (polled, capped in wall time so a
        paused simulation or a missing /clock cannot keep the force on),
        otherwise in wall time."""
        if t0_sim is None:
            self._sleep(max(0.0, duration - (self._monotonic() - t0_wall)))
            return
        cap = max(SIM_WAIT_WALL_MIN, SIM_WAIT_WALL_FACTOR * duration)
        while True:
            now = self._clock_now()
            if now is not None and now - t0_sim >= duration:
                return
            if self._monotonic() - t0_wall >= cap:
                self.log(f"gz_tools: simulation clock did not advance {duration:.2f}s within "
                         f"{cap:.1f}s wall time - clearing the push now")
                return
            self._sleep(POLL_PERIOD)

    def _start(self, name: str, fn) -> bool:
        with self._lock:
            if self.is_busy(name):
                self.log(f"gz_tools: '{name}' rejected, previous command still running")
                return False
            if not self.available():
                self.log(f"gz_tools: '{name}' skipped, '{self.gz}' executable not found")
                return False

            def job():
                try:
                    fn()
                except Exception as e:      # never propagate into the control process
                    self.log(f"gz_tools: '{name}' failed: {e}")

            th = threading.Thread(target=job, name=f"gz_{name}", daemon=True)
            self._threads[name] = th
            th.start()
            return True

    # ------------------------------------------------------------- commands
    def push(self, fx: float, fy: float, fz: float = 0.0, duration: float = 0.2) -> bool:
        """Apply a force [N] on the base link for ``duration`` seconds (async).

        The clear command is *started* ``duration`` after the apply command was
        started (not after the apply process has exited): both gz CLI
        processes carry the same start-up / discovery latency, so the wrench
        is on for ~``duration`` instead of ``duration`` + the whole latency of
        the second process (~1 s, i.e. a 4-8x larger impulse).  ``duration`` is
        measured on the simulation clock when there is one.  After both
        processes have exited the clear is sent once more, unconditionally:
        ``gz topic`` exits 0 even when its message reached nobody, and if the
        apply message arrived after the first clear (slower discovery, cold
        start) the force would otherwise stay on.  The shortest reproducible
        push is limited by the jitter of the CLI latency (~0.05-0.1 s)."""
        apply_cmd = build_wrench_cmd(self.world, self.entity, (fx, fy, fz), gz=self.gz)
        clear_cmd = build_clear_cmd(self.world, self.entity, gz=self.gz)
        dur = max(0.0, float(duration))

        def job():
            res: dict = {}
            t0_wall, t0_sim, th_apply = self._run_async("apply", apply_cmd, res)
            self._wait_since(t0_wall, t0_sim, dur)
            t1_wall, t1_sim, th_clear = self._run_async("clear", clear_cmd, res)
            th_apply.join(self.timeout + 2.0)
            th_clear.join(self.timeout + 2.0)
            ok1, out1, lat1 = res.get("apply", (False, "apply: no result", float("nan")))
            ok2, out2, lat2 = res.get("clear", (False, "clear: no result", float("nan")))
            # safety clear, always (see docstring); retried once if the CLI itself failed
            ok3, out3 = self._run(clear_cmd)
            if not ok3:
                ok3, out3 = self._run(clear_cmd)
            sim = t0_sim is not None and t1_sim is not None
            on_time = (t1_sim - t0_sim) if sim else (t1_wall - t0_wall)
            ok = ok1 and (ok2 or ok3)
            self.history.append({"job": "push", "cmds": [apply_cmd, clear_cmd, clear_cmd], "ok": ok,
                                 "output": out1 + out2 + out3, "on_time": on_time,
                                 "latency": (lat1, lat2)})
            self.log(f"gz_tools: push ({fx:.0f}, {fy:.0f}, {fz:.0f}) N, clear started {on_time:.2f}s "
                     f"{'sim' if sim else 'wall'} time after the apply (requested {dur:.2f}s; gz CLI "
                     f"{lat1:.2f}s / {lat2:.2f}s) "
                     f"{'ok' if ok else 'FAILED: ' + (out1 + out2 + out3).strip()[:200]}")

        return self._start("push", job)

    def clear_wrench(self) -> bool:
        cmd = build_clear_cmd(self.world, self.entity, gz=self.gz)

        def job():
            ok, out = self._run(cmd)
            self.history.append({"job": "clear", "cmds": [cmd], "ok": ok, "output": out})

        return self._start("clear", job)

    def reset_pose(self, x: float, y: float, z: float, yaw: float,
                   on_done: Optional[Callable[[bool], None]] = None) -> bool:
        """Teleport the robot upright to (x, y, z) with heading yaw (async).

        Returns True when the request was started.  ``on_done(ok)`` is then
        called exactly once from the worker thread when the ``gz service`` call
        has returned - ok: the set_pose service accepted the pose (UserCommands
        applies it at the next simulation step); one retry on failure.  Time
        anything that depends on the teleport (settling before standing up)
        from ``on_done``, not from the request.  ``on_done`` must not block and
        should only hand the result over (the job still counts as busy while it
        runs)."""
        cmd = build_set_pose_cmd(self.world, self.model, x, y, z, yaw,
                                 timeout_ms=int(1000 * min(self.timeout, 2.0)), gz=self.gz)

        def job():
            ok = False
            try:
                outs = []
                for _attempt in range(2):
                    rc_ok, out = self._run(cmd)
                    outs.append(out)
                    # gz service prints the reply ("data: true") on success; a missing
                    # service / UserCommands system shows up as a timeout message,
                    # possibly with exit code 0
                    ok = rc_ok and set_pose_reply_ok(out)
                    if ok:
                        break
                self.history.append({"job": "reset_pose", "cmds": [cmd] * len(outs), "ok": ok,
                                     "output": "".join(outs)})
                self.log(f"gz_tools: reset_pose ({x:.2f}, {y:.2f}, {z:.2f}, yaw {yaw:.2f}) "
                         f"{'ok' if ok else 'FAILED: ' + ''.join(outs).strip()[:200]}")
            finally:
                if on_done is not None:
                    try:
                        on_done(bool(ok))
                    except Exception as e:
                        self.log(f"gz_tools: reset_pose on_done callback failed: {e}")

        return self._start("reset_pose", job)
