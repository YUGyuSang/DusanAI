"""
follow_camera: make the Gazebo GUI camera follow the robot (third-person view
behind / above it, like the reference video).

    ros2 run plant_simulation follow_camera
    ros2 run plant_simulation follow_camera --target spot --offset=-2.5,0,1.6 --pgain 0.02

Started by bringup.launch.py with demo:=true (or follow_camera:=true).  It only
uses the `gz` command line tool (no Gazebo Python bindings, no rclpy):

1. waits until Gazebo runs, the robot exists (its /model/<target>/odometry or
   /world/<world>/model/<target>/joint_state topic is advertised) and the GUI
   is up (CameraTracking advertises /gui/camera/pose); gives up quietly after
   --timeout s, e.g. with gui:=false;
2. publishes a gz.msgs.CameraTrack on the GUI topic /gui/track
   (track_mode FOLLOW, follow_target <target>, follow_offset in the robot frame,
   follow_pgain) - understood by gz-gui 8 (Harmonic) ... 10 (Jetty);
3. confirms it on /gui/currently_tracked and re-sends if needed; if that
   never confirms and the GUI offers the older /gui/follow service (Harmonic),
   calls /gui/follow + /gui/follow/offset instead.

config/gui.config also names the target in its CameraTracking plugin
(follow_target / follow_offset / follow_pgain); those values make the GUI wait
for the robot, but the follow mode itself is only switched on by step 2.

The program never fails the launch: every error is logged and it exits 0.
"""
import argparse
import re
import shutil
import subprocess
import sys
import time

FOLLOW_DEFAULTS = {
    "target": "spot",
    "world": "plant",
    "offset": (-2.5, 0.0, 1.6),     # m in the robot frame: behind (-x) and above (+z)
    "pgain": 0.02,                  # per rendered frame (~60 Hz): ~0.8 s time constant
    "timeout": 240.0,               # s waiting for Gazebo, the robot and the GUI
    "attempts": 6,                  # /gui/track sends before falling back to /gui/follow
    "poll": 2.0,                    # s between checks
}
TRACK_TOPIC = "/gui/track"
TRACKED_TOPIC = "/gui/currently_tracked"
GUI_TOPIC = "/gui/camera/pose"


def parse_offset(text):
    """'-2.5,0,1.6' or '-2.5 0 1.6' -> (x, y, z)."""
    parts = [p for p in re.split(r"[,\s]+", str(text).strip()) if p]
    if len(parts) != 3:
        raise ValueError(f"offset needs 3 numbers, got {text!r}")
    return tuple(float(p) for p in parts)


def _num(v):
    return f"{float(v):.4g}"


def track_message(target, offset, pgain, mode="FOLLOW"):
    """Protobuf text of the gz.msgs.CameraTrack request."""
    x, y, z = offset
    return (f'track_mode: {mode}, follow_target: {{name: "{target}"}}, '
            f"follow_offset: {{x: {_num(x)}, y: {_num(y)}, z: {_num(z)}}}, follow_pgain: {_num(pgain)}")


def track_command(gz, target, offset, pgain):
    return [gz, "topic", "-t", TRACK_TOPIC, "-m", "gz.msgs.CameraTrack", "-p", track_message(target, offset, pgain)]


def follow_service_commands(gz, target, offset, timeout_ms=3000):
    """Harmonic-only fallback: /gui/follow (StringMsg) then /gui/follow/offset (Vector3d)."""
    x, y, z = offset
    base = ["--reptype", "gz.msgs.Boolean", "--timeout", str(int(timeout_ms))]
    return [
        [gz, "service", "-s", "/gui/follow", "--reqtype", "gz.msgs.StringMsg"] + base + ["--req", f'data: "{target}"'],
        [gz, "service", "-s", "/gui/follow/offset", "--reqtype", "gz.msgs.Vector3d"] + base
        + ["--req", f"x: {_num(x)}, y: {_num(y)}, z: {_num(z)}"],
    ]


def robot_topics(target, world):
    return (f"/model/{target}/odometry", f"/world/{world}/model/{target}/joint_state")


def is_following(echo_text, target):
    """True if a /gui/currently_tracked sample says FOLLOW* on <target>."""
    m = re.search(r"track_mode:\s*(\w+)", echo_text or "")
    if not m or not m.group(1).upper().startswith("FOLLOW"):
        return False
    t = re.search(r'follow_target\s*\{[^}]*name:\s*"([^"]*)"', echo_text, re.S)
    return bool(t and t.group(1) == target)


def default_runner(cmd, timeout):
    """(returncode, stdout+stderr); (-1, reason) when the command cannot run / times out."""
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                           universal_newlines=True)
        return r.returncode, r.stdout or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or b""
        return -1, out.decode(errors="replace") if isinstance(out, bytes) else str(out)
    except OSError as exc:
        return -1, str(exc)


class FollowCamera:
    """Step logic with injectable runner / clock / sleep (unit-tested)."""

    def __init__(self, gz="gz", target="spot", world="plant", offset=(-2.5, 0.0, 1.6), pgain=0.02,
                 timeout=240.0, attempts=6, poll=2.0, runner=default_runner, clock=time.monotonic,
                 sleep=time.sleep, log=print):
        self.gz, self.target, self.world = gz, target, world
        self.offset, self.pgain = tuple(offset), float(pgain)
        self.timeout, self.attempts, self.poll = float(timeout), int(attempts), float(poll)
        self.run, self.clock, self.sleep, self.log = runner, clock, sleep, log

    def topics(self):
        rc, out = self.run([self.gz, "topic", "-l"], 10.0)
        return set(out.split()) if rc == 0 else set()

    def services(self):
        rc, out = self.run([self.gz, "service", "-l"], 10.0)
        return set(out.split()) if rc == 0 else set()

    def wait_ready(self):
        """'ready' | 'no_gui' | 'no_robot' | 'no_gz' after waiting up to timeout."""
        t0 = self.clock()
        robot = robot_topics(self.target, self.world)
        seen_gz = False
        while True:
            tp = self.topics()
            seen_gz = seen_gz or bool(tp)
            have_robot = any(t in tp for t in robot)
            have_gui = GUI_TOPIC in tp or TRACKED_TOPIC in tp
            if have_robot and have_gui:
                return "ready"
            if self.clock() - t0 >= self.timeout:
                if not seen_gz:
                    return "no_gz"
                return "no_robot" if not have_robot else "no_gui"
            self.sleep(self.poll)

    def confirm(self):
        rc, out = self.run([self.gz, "topic", "-e", "-n", "1", "-t", TRACKED_TOPIC], 4.0)
        return is_following(out, self.target)

    def engage(self):
        """Send /gui/track until confirmed; fall back to /gui/follow.  Returns the method used or ''."""
        for i in range(max(1, self.attempts)):
            rc, out = self.run(track_command(self.gz, self.target, self.offset, self.pgain), 10.0)
            if rc != 0:
                self.log(f"[follow_camera] gz topic -p {TRACK_TOPIC} failed ({rc}): {out.strip()[:200]}")
            self.sleep(1.0)
            if self.confirm():
                return "track"
            self.sleep(self.poll)
        if "/gui/follow" in self.services():
            ok = True
            for cmd in follow_service_commands(self.gz, self.target, self.offset):
                rc, out = self.run(cmd, 10.0)
                ok = ok and rc == 0 and "false" not in out.lower()
                self.sleep(0.5)
            if ok:
                return "follow_service"
        return ""

    def main(self):
        status = self.wait_ready()
        if status == "no_gz":
            self.log(f"[follow_camera] no Gazebo topics after {self.timeout:.0f} s (is gz sim running in this "
                     "GZ_PARTITION?) - giving up")
            return 0
        if status == "no_robot":
            self.log(f"[follow_camera] model '{self.target}' not found after {self.timeout:.0f} s - giving up")
            return 0
        if status == "no_gui":
            self.log("[follow_camera] no Gazebo GUI (gui:=false?) - nothing to follow with")
            return 0
        method = self.engage()
        x, y, z = self.offset
        if method:
            self.log(f"[follow_camera] Gazebo GUI camera follows '{self.target}' (offset {x:g} {y:g} {z:g} m, "
                     f"pgain {self.pgain:g}, via {'/gui/track' if method == 'track' else '/gui/follow'})")
        else:
            self.log("[follow_camera] could not confirm the follow camera - right-click the robot in the Entity "
                     "tree and choose 'Follow' instead")
        return 0


def build_parser():
    d = FOLLOW_DEFAULTS
    ap = argparse.ArgumentParser(prog="follow_camera", description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", default=d["target"], help="model to follow (default %(default)s)")
    ap.add_argument("--world", default=d["world"], help="world name (default %(default)s)")
    ap.add_argument("--offset", default=",".join(f"{v:g}" for v in d["offset"]),
                    help="camera offset x,y,z in the robot frame, m (default %(default)s); use --offset=-2,0,1")
    ap.add_argument("--pgain", type=float, default=d["pgain"], help="follow P gain per frame (default %(default)s)")
    ap.add_argument("--timeout", type=float, default=d["timeout"], help="s to wait for gz / robot / GUI")
    ap.add_argument("--attempts", type=int, default=d["attempts"])
    ap.add_argument("--gz", default="", help="gz executable ('' = from PATH)")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in argv:                       # launch_ros Node(...) appends ROS arguments
        argv = argv[:argv.index("--ros-args")]
    try:
        args = build_parser().parse_args(argv)
        offset = parse_offset(args.offset)
    except (SystemExit, ValueError) as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            return 0
        print(f"[follow_camera] bad arguments ({exc}) - not following", flush=True)
        return 0
    gz = args.gz or shutil.which("gz")
    if not gz:
        print("[follow_camera] the gz command line tool is not on PATH - not following", flush=True)
        return 0

    def log(msg):
        print(msg, flush=True)

    try:
        return FollowCamera(gz, args.target, args.world, offset, args.pgain, args.timeout, args.attempts,
                            log=log).main()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # never fail the launch
        log(f"[follow_camera] unexpected error {type(exc).__name__}: {exc} - not following")
        return 0


if __name__ == "__main__":
    sys.exit(main())
