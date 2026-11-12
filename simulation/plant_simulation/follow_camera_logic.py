"""
follow_camera_logic: the ROS-free / gz-free part of follow_camera (unit-tested with a simulated GUI).

What the Gazebo GUI does (gz-gui CameraTracking + InteractiveViewControl, gz-rendering BaseCamera;
the same code in gz-gui 8 Harmonic, 9 Ionic and 10 Jetty):

* /gui/track (gz.msgs.CameraTrack), CameraTracking::OnTrackSub: any track_mode except USE_LAST
  replaces the mode, an empty target name keeps the target, follow_offset replaces the offset when
  present, a gain is taken only when it is > 1e-5 (else the previous one stays).
* FOLLOW: CameraTracking::OnRender hands the message's follow_pgain to the camera's look-at (track)
  slerp and the message's track_pgain to its POSITION gain (SetTrackPGain(followPGain);
  SetFollowPGain(trackPGain)) - the names are crossed.  Both default to 0.01.
* gz-rendering BaseCamera::PreRender, once per rendered frame: target = robot pose x offset (offset in
  the robot frame, full attitude), camera += (target - camera) * position gain, then the look-at slerp.
* InteractiveViewControl (mouse: wheel / right drag = zoom, middle or Shift+left drag = orbit, left
  drag = pan) moves the camera after that, and nothing writes the result back into the offset: in
  FOLLOW the GUI pulls a camera moved with the mouse back (~0.8 s at 60 fps with position gain 0.02).
* TRACK: the camera position is not followed at all (it stays where it is), it keeps looking at the
  target.  Esc in the 3D view (applied with the next mouse event there) -> mode NONE, targets
  cleared; a later USE_LAST message cannot switch that back on.
* /gui/currently_tracked (50 Hz while subscribed): FOLLOW with target / offset / follow_pgain
  (track_pgain cleared), TRACK with track target / track_pgain, NONE.  /gui/camera/pose (50 Hz while
  subscribed): camera world pose.

So the view a user sets with the mouse is kept the way older Gazebo (Fortress) did it: the
CameraDirector below notices the mouse (the camera moves in a way the follow law cannot explain),
holds the camera's position (position gain ~0, still looking at the robot) while the user moves it,
and once the camera has been still for a moment makes the camera pose where the user left it,
relative to the robot, the new view.  The fall guard holds / releases with that view; keys switch
follow on / off and reset the view; the default view is pulled in front of the building walls.

Why a view is not simply sent as the offset: with a position gain g per rendered frame the follow
law trails a robot walking at v by about v / k, k = fps * -ln(1 - g) (0.4 m/s at 30 fps: 0.67 m,
at 15 fps: 1.3 m); a camera the user left behind a walking robot would glide out by that again
(and every further adjustment a bit more).  So a chosen view is sent as view + v / k (robot's
horizontal velocity, k learnt from the camera samples) and re-sent when that changes: the camera
stays at the view while the robot walks and while it stands.  The default view is sent as it is
(it drops back a little while the robot walks, the look of the demo videos).
"""
import bisect
import math
import re
from collections import namedtuple

DEFAULT_OFFSET = (-2.5, 0.0, 1.6)     # m in the robot frame: behind (-x) and above (+z)
LOOK_GAIN = 0.05                      # look-at slerp per rendered frame (message field follow_pgain)
POS_GAIN = 0.02                       # position per rendered frame (message field track_pgain): ~0.8 s at 60 fps
HOLD_GAIN = 0.00002                   # position gain while held: > 1e-5 (smaller is ignored by the GUI), ~frozen
MIN_GUI_GAIN = 0.00001                # CameraTracking takes a gain only when it is above this
QT_KEY_C, QT_KEY_Z = 0x43, 0x5A       # Gazebo KeyPublisher: Qt key codes (upper-case ASCII)
TRACK_MODES = {0: "NONE", 1: "TRACK", 2: "FOLLOW", 3: "FOLLOW_FREE_LOOK", 4: "FOLLOW_LOOK_AT", 5: "USE_LAST"}
# s: candidate lag of the GUI's robot pose behind ours; negative = our pose (/odom through the bridge,
# odom_tf and rclpy) is the late one (extrapolated).  0.0 first: the start value before any learning.
DELAYS = (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, -0.05, -0.1, -0.15, -0.2)

CAMERA_DEFAULTS = {
    "look_gain": LOOK_GAIN,
    "pos_gain": POS_GAIN,
    "hold_gain": HOLD_GAIN,
    # mouse -> kept view
    "adjust": True,                 # False: no mouse detection (the GUI pulls the camera back as before)
    "trigger": 0.06,                # m: one camera step the follow law cannot explain = the mouse
    "window": 0.35,                 # s ...
    "window_sum": 0.09,             # m ... or this much unexplained motion within `window`
    "quiet": 0.4,                   # s camera still (held) -> the view is taken over
    "still_tol": 0.003,             # m per camera sample counted as still
    "max_hold": 8.0,                # s: take the view over at the latest after this long
    "commit_min": 0.10,             # m moved while held ...
    "big_step": 0.15,               # m ... or this big a first step, else it was no mouse: old view back
    "min_dist": 1.0, "max_dist": 12.0, "min_z": -0.2,   # limits of a view (m, robot frame)
    # fall guard
    "fall_guard": True,             # False: no camera hold while the robot is down
    "tilt_hold_deg": 30.0,          # |roll| or |pitch| above (or a reported fall) -> hold the camera
    "tilt_release_deg": 20.0,       # upright below this (ramp 11.8 deg + gait sway stays below) ...
    "release_hold": 1.0,            # ... with recovery_state NORMAL for this long [s] -> follow again
    "hold_update": 1.0,             # s between levelled-offset refreshes while holding
    "hold_min_change": 0.25,        # m change of the levelled offset worth a refresh
    # a chosen view (mouse / 'c'): offset = view + the follow law's trail behind the walking robot
    "trail": True,                  # False: send the chosen view as it is (it drifts out while walking)
    "trail_min_change": 0.15,       # m change of the trail worth a send ...
    "trail_interval": 0.5,          # ... at most this often [s] (gz-transport); the gz CLI: 2 s
    "trail_span": 0.6,              # s of robot poses for its velocity (one trot period and more)
    # walls (default view only)
    "avoid_walls": True,
    "wall_margin": 0.3,             # m kept between the camera and a wall
    "wall_min_horizontal": 0.6,     # m: never pull the camera closer than this (horizontally)
    "wall_min_change": 0.15,        # m change of the pulled-in offset worth a send ...
    "wall_interval": 0.3,           # ... at most this often [s] (gz-transport); the gz CLI: 1 s
    # transport
    "echo_timeout": 1.5,            # s waiting for /gui/currently_tracked to show a sent offset
    "cli_settle": 0.3,              # s after a `gz topic -p` returned before the camera is watched again
    "key_repeat": 0.5,              # s: the same key again within this = OS auto-repeat
    "start_mute": 3.0,              # s after the start before the mouse detection looks at the camera
}
DOWN_MODES = ("FALLEN", "RECOVERING")


# ---------------------------------------------------------------- small vector / quaternion helpers
def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def q_rot(q, v):
    """Rotate v by the unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx))


def q_inv_rot(q, v):
    w, x, y, z = q
    return q_rot((w, -x, -y, -z), v)


def q_mul(a, b):
    """Hamilton product a * b of (w, x, y, z) quaternions (b first, then a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def q_normalize(q):
    n = math.sqrt(sum(c * c for c in q))
    if not (n > 1e-9 and math.isfinite(n)):
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(c / n for c in q)


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def rpy_of(q):
    """(roll, pitch, yaw) of q = (w, x, y, z), ZYX convention."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch, yaw_of(q)


def quat_rpy(roll, pitch, yaw):
    cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def slerp(a, b, w):
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = tuple(-x for x in b), -dot
    if dot > 0.9995:
        return q_normalize(tuple(x + w * (y - x) for x, y in zip(a, b)))
    th = math.acos(min(dot, 1.0))
    s = math.sin(th)
    return q_normalize(tuple((math.sin((1 - w) * th) * x + math.sin(w * th) * y) / s for x, y in zip(a, b)))


def level_offset(offset, roll, pitch):
    """Robot-frame offset that the GUI's full-attitude follow (R(yaw) R(pitch) R(roll) x offset)
    turns into R(yaw) x offset: Rx(roll)^T Ry(pitch)^T offset - camera behind / above the
    robot's heading whatever its roll and pitch."""
    x, y, z = (float(v) for v in offset)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    x1, z1 = cp * x - sp * z, sp * x + cp * z
    return (x1, cr * y + sr * z1, -sr * y + cr * z1)


def clamp_offset(off, min_dist=1.0, max_dist=12.0, min_z=-0.2):
    """Keep a view between min_dist and max_dist from the robot and not below min_z."""
    x, y, z = (float(v) for v in off)
    d = math.sqrt(x * x + y * y + z * z)
    if not (d > 1e-6 and math.isfinite(d)):
        return (-float(min_dist), 0.0, 0.5)
    s = min(max(d, min_dist), max_dist) / d
    return (x * s, y * s, max(z * s, min_z))


def describe_view(off):
    """'2.5 m behind, 1.6 m up' style text of a robot-frame offset (log lines)."""
    x, y, z = off
    h = math.hypot(x, y)
    az = math.degrees(math.atan2(y, -x)) if h > 1e-6 else 0.0      # 0 = straight behind, + = to the left
    side = "behind" if abs(az) <= 45 else ("in front" if abs(az) >= 135 else ("left" if az > 0 else "right"))
    return f"{h:.1f} m {side} ({az:+.0f} deg), {z:+.1f} m up"


def robot_down(d):
    """/locomotion/state dict: the controller reports a fall / a recovery in progress."""
    rs = str(d.get("recovery_state") or "")
    return (rs not in ("", "NORMAL")) or str(d.get("mode") or "") in DOWN_MODES


# ---------------------------------------------------------------- /gui/track requests
def _num(v):
    return f"{float(v):.6g}"


class TrackRequest(namedtuple("TrackRequest", "mode target offset look pos")):
    """One gz.msgs.CameraTrack for /gui/track.

    mode FOLLOW   (engage / follow again): follow_target, follow_offset, both gains
         USE_LAST (every other change): follow_offset + both gains, the GUI keeps mode and target (so a
                  camera the user switched off with Esc stays off)
         TRACK    ('c': camera stays, keeps looking at the robot): track_target + look gain, which TRACK
                  takes from the message's track_pgain
    look = look-at gain (message follow_pgain), pos = position gain (message track_pgain)."""

    def text(self):
        """Protobuf text for `gz topic -p`."""
        if self.mode == "TRACK":
            return f'track_mode: TRACK, track_target: {{name: "{self.target}"}}, track_pgain: {_num(self.look)}'
        x, y, z = self.offset
        head = f"track_mode: {self.mode}, "
        if self.mode != "USE_LAST":
            head += f'follow_target: {{name: "{self.target}"}}, '
        return (head + f"follow_offset: {{x: {_num(x)}, y: {_num(y)}, z: {_num(z)}}}, "
                f"follow_pgain: {_num(self.look)}, track_pgain: {_num(self.pos)}")

    def fill(self, m, enum):
        """Fill a gz.msgs.CameraTrack protobuf message (gz-transport python bindings)."""
        if self.mode == "TRACK":
            m.track_mode = enum.TRACK
            m.track_target.name = str(self.target)
            m.track_pgain = float(self.look)
            return m
        m.track_mode = getattr(enum, self.mode)
        if self.mode != "USE_LAST":
            m.follow_target.name = str(self.target)
        m.follow_offset.x, m.follow_offset.y, m.follow_offset.z = (float(v) for v in self.offset)
        m.follow_pgain = float(self.look)
        m.track_pgain = float(self.pos)
        return m


def tracked_matches(info, req, tol=1e-3):
    """True if a /gui/currently_tracked sample shows that `req` was applied."""
    if not info:
        return False
    if req.mode == "TRACK":
        return info.get("mode") == "TRACK" and info.get("track_target") == req.target
    if info.get("mode") != "FOLLOW":
        return False
    off = info.get("follow_offset")
    if off is None or max(abs(a - b) for a, b in zip(off, req.offset)) > tol:
        return False
    g = info.get("follow_pgain")
    return g is None or abs(float(g) - float(req.look)) <= 1e-6 * max(1.0, abs(float(req.look)))


# ---------------------------------------------------------------- robot pose buffer
class PoseBuffer:
    """Recent robot poses (t, p, q) for interpolation at t - delay; a time after the newest pose is
    extrapolated (x, y and yaw, at most `extrapolate` s, with the rates of the last `rate_span` s - most
    of a trot period, so the gait's wobble is hardly extrapolated: our /odom can be later than the GUI's
    robot pose)."""

    def __init__(self, keep=1.0, extrapolate=0.25, rate_span=0.3):
        self.t, self.p, self.q, self.keep = [], [], [], float(keep)
        self.extrapolate, self.rate_span = float(extrapolate), float(rate_span)

    def add(self, t, p, q):
        if self.t and t <= self.t[-1]:
            return
        self.t.append(t)
        self.p.append(tuple(p))
        self.q.append(q_normalize(q))
        k = bisect.bisect_left(self.t, t - self.keep)
        if k > 0:
            del self.t[:k], self.p[:k], self.q[:k]

    def latest(self):
        return (self.t[-1], self.p[-1], self.q[-1]) if self.t else None

    def at(self, t):
        if not self.t:
            return None
        i = bisect.bisect_left(self.t, t)
        if i <= 0:
            return self.p[0], self.q[0]
        if i >= len(self.t):
            return self._ahead(t)
        t0, t1 = self.t[i - 1], self.t[i]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        p = tuple(a + w * (b - a) for a, b in zip(self.p[i - 1], self.p[i]))
        return p, slerp(self.q[i - 1], self.q[i], w)

    def _ahead(self, t):
        t1, p1, q1 = self.t[-1], self.p[-1], self.q[-1]
        dt = min(t - t1, self.extrapolate)
        j = bisect.bisect_right(self.t, t1 - self.rate_span) - 1
        if j < 0 and t1 - self.t[0] >= 0.1:
            j = 0
        if dt <= 0.0 or j < 0:
            return p1, q1
        span = t1 - self.t[j]
        vx, vy = (p1[0] - self.p[j][0]) / span, (p1[1] - self.p[j][1]) / span
        sp = math.hypot(vx, vy)
        if sp > 3.0:                          # a respawn / teleport, not motion
            return p1, q1
        w = wrap_angle(yaw_of(q1) - yaw_of(self.q[j])) / span
        w = min(max(w, -3.0), 3.0)
        return (p1[0] + vx * dt, p1[1] + vy * dt, p1[2]), q_normalize(q_mul(yaw_quat(w * dt), q1))

    def velocity(self, t, span=1.0):
        """Least-squares velocity of the poses within span s up to t (the trot's bounce / sway averages
        out), or None with less than 0.3 s of poses."""
        i1 = bisect.bisect_right(self.t, t)
        i0 = bisect.bisect_left(self.t, t - span)
        if i1 - i0 < 3 or self.t[i1 - 1] - self.t[i0] < 0.3:
            return None
        ts = self.t[i0:i1]
        tm = sum(ts) / len(ts)
        den = sum((x - tm) ** 2 for x in ts)
        out = []
        for k in range(3):
            ps = [p[k] for p in self.p[i0:i1]]
            pm = sum(ps) / len(ps)
            out.append(sum((x - tm) * (y - pm) for x, y in zip(ts, ps)) / den)
        return tuple(out)

    def yaw_rate(self, t, span=0.3):
        """|yaw rate| [rad/s] over the last span s up to t (0 without poses)."""
        if len(self.t) < 2:
            return 0.0
        t1 = min(t, self.t[-1])
        t0 = max(t1 - span, self.t[0])
        if t1 - t0 < 0.1:
            return 0.0
        return abs(wrap_angle(yaw_of(self.at(t1)[1]) - yaw_of(self.at(t0)[1]))) / (t1 - t0)

    def mean_yaw(self, t, half=0.2):
        """Yaw at t averaged over [t - half, t + half] (one trot period: the gait's yaw wobble is not a
        heading change); the plain yaw at t when that window is not fully recorded yet."""
        rp = self.at(t)
        if rp is None:
            return None
        yc = yaw_of(rp[1])
        if not self.t or t - half < self.t[0] or t + half > self.t[-1]:
            return yc
        i0, i1 = bisect.bisect_left(self.t, t - half), bisect.bisect_right(self.t, t + half)
        if i1 - i0 < 3:
            return yc
        return yc + sum(wrap_angle(yaw_of(q) - yc) for q in self.q[i0:i1]) / (i1 - i0)


# ---------------------------------------------------------------- mouse detection
class MouseDetector:
    """Tells a camera moved by the mouse from the GUI's follow law.

    Between two /gui/camera/pose samples (t0, c0) -> (t, c) the follow law can only have moved the
    camera along u = (robot + R offset) - c0 by a fraction a in [g, 1 - (1 - g)^n] (g = position gain,
    n = rendered frames: at most ceil(dt * fps_max) + 1, at least one because the pose changed).
    The rest r = d - a u is not the follow law: the mouse.  A robot pose error e only enters as a
    times e (a ~ 0.02), so the lag between our robot pose and the GUI's matters little; the delay
    candidate with the smallest residual on quiet samples is used anyway.  update() returns the
    residual (r, window sum) when |r| > trigger or the sum of r within `window` s > window_sum."""

    def __init__(self, offset, gain, fps_max=120.0, delays=DELAYS, trigger=0.06, window=0.35, window_sum=0.09,
                 max_gap=0.3, delay_tau=3.0, keep=1.5, fps_guess=30.0, rate_tau=20.0):
        self.fps_max, self.delays = float(fps_max), tuple(delays)
        self.trigger, self.window, self.window_sum = float(trigger), float(window), float(window_sum)
        self.max_gap, self.delay_tau = float(max_gap), float(delay_tau)
        self.err = [0.01] * len(self.delays)
        self.robot = PoseBuffer(keep)
        self.prev, self.buf = None, []
        self._cand = [[] for _ in self.delays]       # per delay candidate: residuals within `window`
        self._csum = [(0.0, 0.0, 0.0)] * len(self.delays)
        # follow rate k [1/s] of the GUI's position law (camera += (target - camera) * g per rendered frame
        # ~ dc/dt = k (target - camera), k = fps * -ln(1 - g)): least squares over followed samples
        self.fps_guess, self.rate_tau = float(fps_guess), float(rate_tau)
        self.rate_sxy = self.rate_sxx = 0.0
        self.rate_t, self.rate_pending = None, []
        self.set_follow(offset, gain)
        self.rate_gain = self.gain               # the gain the GUI follows with (not a hold gain)

    @property
    def delay(self):
        return self.delays[min(range(len(self.delays)), key=lambda j: self.err[j])]

    def set_follow(self, offset, gain):
        self.offset, self.gain = tuple(float(v) for v in offset), min(max(float(gain), 0.0), 1.0)
        self.buf = []

    def restart(self, t=None, cam=None):
        """Forget the previous sample (after a send, a gap, a state change)."""
        self.prev = None if t is None else (t, tuple(cam))
        self.buf = []
        self._cand = [[] for _ in self.delays]
        self._csum = [(0.0, 0.0, 0.0)] * len(self.delays)

    def rate(self):
        """Estimated follow rate k [1/s] of the follow gain (a guess of fps_guess frames/s until the robot
        has walked a little: only a camera trailing a moving target tells the rate)."""
        g = min(max(self.rate_gain, 1e-6), 0.999)
        prior = self.fps_guess * -math.log(1.0 - g)
        w0 = 0.05                                # m^2 s: worth ~1-2 s of walking samples
        return (self.rate_sxy + prior * w0) / (self.rate_sxx + w0)

    def fps_estimate(self):
        """Rendered frames per second implied by the follow rate."""
        g = min(max(self.rate_gain, 1e-6), 0.999)
        return self.rate() / -math.log(1.0 - g)

    def _rate_sample(self, t, t0, c0, d, delay):
        """Queue one followed sample for the rate fit (committed once no mouse showed up after it)."""
        tm = 0.5 * (t + t0)                    # the frames in between saw the target of about this moment
        rp = self.robot.at(tm - delay)
        if rp is None or not (MIN_GUI_GAIN * 10 < self.gain < 0.5):
            return
        if self.gain != self.rate_gain:           # another follow gain (--pos-gain): start over
            self.rate_gain, self.rate_sxy, self.rate_sxx, self.rate_pending = self.gain, 0.0, 0.0, []
        p, q = rp
        u = sub(add(p, q_rot(q, self.offset)), c0)
        uu = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
        if uu > 0.0025:                         # |u| > 5 cm: a camera trailing the target
            self.rate_pending.append((t, d[0] * u[0] + d[1] * u[1] + d[2] * u[2], uu * (t - t0)))

    def _rate_commit(self, t, hold_back=0.5):
        while self.rate_pending and self.rate_pending[0][0] <= t - hold_back:
            ts, xy, xx = self.rate_pending.pop(0)
            if self.rate_t is not None:
                f = math.exp(-max(ts - self.rate_t, 0.0) / self.rate_tau)
                self.rate_sxy, self.rate_sxx = self.rate_sxy * f, self.rate_sxx * f
            self.rate_t = ts
            self.rate_sxy += xy
            self.rate_sxx += xx

    def add_robot(self, t, p, q):
        self.robot.add(t, p, q)

    def robot_at(self, t):
        return self.robot.at(t - self.delay)

    def _resid(self, t, t0, c0, d, delay, offset=None, gain=None, amin=None, min_dt=0.0, fps=None):
        rp = self.robot.at(t - delay)
        if rp is None:
            return None
        p, q = rp
        g = self.gain if gain is None else gain
        u = sub(add(p, q_rot(q, self.offset if offset is None else offset)), c0)
        L2 = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
        a = 0.0
        if L2 > 1e-10:
            n = math.ceil(max(t - t0, min_dt) * (self.fps_max if fps is None else fps)) + 1
            amax = 1.0 - (1.0 - g) ** n
            a = min(max((d[0] * u[0] + d[1] * u[1] + d[2] * u[2]) / L2, g if amin is None else amin), amax)
        return (d[0] - a * u[0], d[1] - a * u[1], d[2] - a * u[2])

    def explain(self, t0, c0, t, cam, offset, gain, max_gap=None):
        """Split the camera motion between two samples into (not explained, explained) by a follow law
        with this offset and AT MOST this gain (no lower bound: a held camera that does not move
        explains itself); None when the samples are too far apart (or out of order) to tell."""
        gap = self.max_gap if max_gap is None else float(max_gap)
        if not (0.0 <= t - t0 <= gap):
            return None
        d = sub(cam, c0)
        # samples queued while a blocking `gz topic -p` ran arrive with one time stamp: allow them some frames;
        # frames: as many as the learnt frame rate gives (over a gap, fps_max would let the follow law take
        # up a zoom towards a walking robot, which points the way the camera trails)
        fps = min(max(1.25 * self.fps_estimate(), 10.0), self.fps_max)
        r = self._resid(t, t0, c0, d, self.delay, offset, gain, 0.0, min_dt=0.0 if t > t0 else 0.1, fps=fps)
        if r is None:
            return None
        return r, sub(d, r)

    def update(self, t, cam):
        cam = tuple(float(v) for v in cam)
        if self.prev is None or not (0.0 < t - self.prev[0] <= self.max_gap):
            self.restart(t, cam)
            return None
        t0, c0 = self.prev
        if cam == c0:                     # no frame rendered since the last sample: nothing to learn
            return None
        self.prev = (t, cam)
        d = sub(cam, c0)
        res = [self._resid(t, t0, c0, d, dl) for dl in self.delays]
        if any(r is None for r in res):
            return None
        j = min(range(len(self.delays)), key=lambda k: self.err[k])
        r = res[j]
        # Learn the delay from the residual summed over `window` (what the trigger looks at; per sample the
        # trot's sway dominates) and only while the robot turns: that is where a wrong delay shows
        # (sideways, the follow law cannot take it up) and makes false triggers; walking straight, the
        # sway's ~0.45 s period makes a delay 0.4 s off look about as good, wrong in the next turn.
        sums = []
        for i, ri in enumerate(res):
            cb, cs = self._cand[i], add(self._csum[i], ri)
            cb.append((t, ri))
            while cb and cb[0][0] <= t - self.window:
                cs = sub(cs, cb.pop(0)[1])
            self._csum[i] = cs
            sums.append(min(norm(cs), 0.2))
        if self.robot.yaw_rate(t) > 0.2 and max(sums) - min(sums) > 0.005:
            k = 1.0 - math.exp(-0.02 / self.delay_tau)
            for i, si in enumerate(sums):
                self.err[i] += k * (si - self.err[i])
        self.buf.append((t, r))
        while self.buf and self.buf[0][0] <= t - self.window:
            self.buf.pop(0)
        s = (sum(b[1][0] for b in self.buf), sum(b[1][1] for b in self.buf), sum(b[1][2] for b in self.buf))
        if norm(r) > self.trigger or norm(s) > self.window_sum:
            self.buf = []
            self.rate_pending = []                 # maybe the mouse already: not the follow law
            return r, s
        if norm(r) < 0.01:
            self._rate_sample(t, t0, c0, d, self.delays[j])
        self._rate_commit(t)
        return None


# ---------------------------------------------------------------- fall guard
class FallGuard:
    """When to hold / release the camera while the robot is down (ROS free).

    update(t, roll, pitch, down) -> None, 'hold' when the robot starts to fall (|roll| or |pitch| above
    tilt_hold_deg or a reported fall), 'release' once it is upright (below tilt_release_deg, not down)
    for release_hold s.  Angles in rad, t in s of any monotonic clock."""

    def __init__(self, tilt_hold_deg=30.0, tilt_release_deg=20.0, release_hold=1.0, **_unused):
        self.hold_tilt = math.radians(float(tilt_hold_deg))
        self.release_tilt = math.radians(min(float(tilt_release_deg), float(tilt_hold_deg)))
        self.release_hold = float(release_hold)
        self.holding = False
        self.holds = 0
        self._up_since = None

    def update(self, t, roll, pitch, down=False):
        try:
            roll, pitch = float(roll), float(pitch)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(roll) and math.isfinite(pitch)):
            return None
        tilt = max(abs(roll), abs(pitch))
        if not self.holding:
            if tilt > self.hold_tilt or down:
                self.holding, self._up_since = True, None
                self.holds += 1
                return "hold"
            return None
        if tilt < self.release_tilt and not down:
            if self._up_since is None:
                self._up_since = t
            if t - self._up_since >= self.release_hold:
                self.holding, self._up_since = False, None
                return "release"
        else:
            self._up_since = None
        return None


# ---------------------------------------------------------------- building walls
def layout_walls(layout):
    """2-D wall segments ((x1, y1), (x2, y2)) of the building shells in world_layout.yaml: the corridor
    (long walls, east end, west end beside the door frame), the turbine hall (outer walls, east wall
    beside the opening to the corridor) and the hall's partition beside its door.  Missing keys are
    skipped, so another layout just gives fewer walls."""
    get = layout.get if hasattr(layout, "get") else (lambda k, d=None: d)
    segs = []

    def box_with_gaps(r, gaps):
        x0, x1, y0, y1 = float(r["x_min"]), float(r["x_max"]), float(r["y_min"]), float(r["y_max"])
        sides = {"south": ((x0, y0), (x1, y0)), "north": ((x0, y1), (x1, y1)),
                 "west": ((x0, y0), (x0, y1)), "east": ((x1, y0), (x1, y1))}
        for name, (a, b) in sides.items():
            gap = gaps.get(name)
            if gap is None:
                segs.append((a, b))
                continue
            g0, g1 = sorted(float(v) for v in gap)
            if name in ("west", "east"):
                x = a[0]
                if g0 > y0:
                    segs.append(((x, y0), (x, g0)))
                if g1 < y1:
                    segs.append(((x, g1), (x, y1)))
            else:
                y = a[1]
                if g0 > x0:
                    segs.append(((x0, y), (g0, y)))
                if g1 < x1:
                    segs.append(((g1, y), (x1, y)))

    try:
        corr = dict(get("corridor") or {})
        if "interior" in corr:
            door = corr.get("door_frame") or {}
            gap = (door["y_min"], door["y_max"]) if {"y_min", "y_max"} <= set(door) else None
            box_with_gaps(corr["interior"], {"west": gap} if gap else {})
        fac = dict(get("factory") or {})
        if "interior" in fac:
            op = fac.get("east_opening") or {}
            gap = (op["y_min"], op["y_max"]) if {"y_min", "y_max"} <= set(op) else None
            box_with_gaps(fac["interior"], {"east": gap} if gap else {})
            part = fac.get("partition") or {}
            if {"x_min", "door_y"} <= set(part):
                r = fac["interior"]
                x = float(part["x_min"])
                d0, d1 = sorted(float(v) for v in part["door_y"])
                segs.append(((x, float(r["y_min"])), (x, d0)))
                segs.append(((x, d1), (x, float(r["y_max"]))))
    except (KeyError, TypeError, ValueError):
        pass
    return segs


def _seg_hit(p, c, a, b):
    """Fraction s in [0, 1] along p -> c where it crosses the segment a-b, or None."""
    rx, ry = c[0] - p[0], c[1] - p[1]
    sx, sy = b[0] - a[0], b[1] - a[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-12:
        return None
    qx, qy = a[0] - p[0], a[1] - p[1]
    s = (qx * sy - qy * sx) / den
    u = (qx * ry - qy * rx) / den
    if 0.0 <= s <= 1.0 and 0.0 <= u <= 1.0:
        return s
    return None


class WallMap:
    """Camera collision against vertical building walls (2-D): pull_in(p, c) moves the camera c towards the
    robot p until it is `margin` in front of the first wall between them (never closer than
    min_horizontal); returns c unchanged when nothing is in between."""

    def __init__(self, segments, margin=0.3, min_horizontal=0.6):
        self.segments = [tuple((float(a[0]), float(a[1])) for a in s) for s in segments]
        self.margin, self.min_h = float(margin), float(min_horizontal)

    def __bool__(self):
        return bool(self.segments)

    def pull_in(self, p, c):
        dx, dy = c[0] - p[0], c[1] - p[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return tuple(c)
        # extend the ray by the margin: a camera just in front of a wall is pulled away from it too
        far = (p[0] + dx * (L + self.margin) / L, p[1] + dy * (L + self.margin) / L)
        hit = None
        for a, b in self.segments:
            s = _seg_hit(p, far, a, b)
            if s is not None and (hit is None or s < hit):
                hit = s
        if hit is None:
            return tuple(c)
        keep = max(hit * (L + self.margin) - self.margin, min(self.min_h, L))
        if keep >= L:
            return tuple(c)
        return (p[0] + dx * keep / L, p[1] + dy * keep / L, c[2])


# ---------------------------------------------------------------- the camera state machine
class CameraDirector:
    """Follow camera with a mouse-adjustable, kept view (ROS free, gz free).

    States
      FOLLOW  the GUI follows the robot with the current view (robot-frame offset)
      ADJUST  the user moves the camera with the mouse: position held (position gain hold_gain, still
              looking at the robot); camera still for `quiet` s -> where the user left the camera,
              relative to the robot, is the new view (sent with the follow law's trail, module doc),
              or, if hardly anything moved, the old view comes back
      FALL    robot down: camera held (levelled offset, hold gain); upright again -> follow the view
              (a camera the USER moved meanwhile becomes the view - only with the mouse view on)
      TRIPOD  key 'c': the camera stays where it is and keeps looking at the robot (gz TRACK mode);
              'c' again follows from the camera pose of that moment
      PASSIVE the user took the camera over in the GUI (Esc, right-click Free Look / Track / another
              entity): nothing is sent; 'c' / 'z' / right-click Follow on the robot follow again
    Keys (Gazebo window): 'c' follow on / off, 'z' default view.  The default view (and only that one:
    a view the user chose is kept exactly) is pulled in front of the building walls.

    Inputs: on_robot(t, p, q) (world pose), on_camera(t, pos) (/gui/camera/pose position),
    on_tracked(t, info) (/gui/currently_tracked as a dict, see parse_tracked), on_loco(t, d)
    (/locomotion/state dict), on_key(t, qt_code), tick(t).  Output: send(TrackRequest) -> 'gz' (fast
    link: wait for the echo) | 'cli' (blocking `gz topic -p` that returned) | '' (failed)."""

    def __init__(self, target, offset, send, log=print, clock=None, cfg=None, walls=None, t0=0.0, fast=True):
        c = dict(CAMERA_DEFAULTS)
        c.update({k: v for k, v in dict(cfg or {}).items() if k in CAMERA_DEFAULTS and v is not None})
        self.cfg = c
        self.target = str(target)
        self.default = tuple(float(v) for v in offset)
        self.send_fn, self.log, self.clock = send, log, clock
        self.fast = bool(fast)                # gz-transport link (echo) vs gz CLI (blocking sends)
        self.walls = walls if (walls and c["avoid_walls"]) else None
        self.view = self.default              # robot-frame offset the user wants (default or chosen)
        self.view_auto = True                 # default / reset view: walls may pull it in
        self.eff = self.default               # offset last sent (= what the GUI applies)
        self.state = "FOLLOW"
        self.det = MouseDetector(self.default, c["pos_gain"], trigger=c["trigger"], window=c["window"],
                                 window_sum=c["window_sum"])
        self.guard = FallGuard(c["tilt_hold_deg"], c["tilt_release_deg"], c["release_hold"])
        self.mute_until = float(t0) + float(c["start_mute"])
        self.pending = None                   # (request, t_sent) waiting for its echo
        self.last_send_t = -math.inf
        self.last_auto_t = -math.inf          # last walls / trail send
        self.cam = None                       # (t, pos) newest camera sample
        self.level = None                     # (t, roll, pitch) low-passed trunk attitude (gait sway removed)
        self.att = (0.0, 0.0, False)          # roll, pitch, down from /locomotion/state
        self.adj = None                       # ADJUST / FALL bookkeeping
        self.hold_off, self.hold_t = None, -math.inf
        self._key_last = (None, -math.inf)
        self.stats = {"adopted": 0, "reverted": 0, "holds": 0, "sent": 0, "passive": 0}
        self.echo_misses = 0

    # ------------------------------------------------------------------ helpers
    def _now(self, t):
        if self.clock is None:
            return t
        try:
            return max(float(self.clock()), t)
        except Exception:
            return t

    def _req(self, mode, offset, pos):
        return TrackRequest(mode, self.target, tuple(float(v) for v in offset), self.cfg["look_gain"], pos)

    def _send(self, req, t, why=""):
        try:
            via = self.send_fn(req)
        except Exception as exc:  # never let a transport error stop the camera logic
            self.log(f"[follow_camera] /gui/track send failed ({type(exc).__name__}: {exc})")
            via = ""
        self.stats["sent"] += 1
        now = self._now(t)
        self.last_send_t = now
        if req.mode != "TRACK":
            self.eff = tuple(req.offset)
            self.det.set_follow(req.offset, req.pos)
        self.det.restart()
        if via == "gz":
            self.pending = (req, now)
            self.mute_until = now + self.cfg["echo_timeout"]
        else:
            self.pending = None
            self.mute_until = now + self.cfg["cli_settle"]
        return via

    def _frame(self, t, heading_only=False, smooth=False):
        """(robot position, view frame) at t: the frame of the robot's heading and its low-passed roll /
        pitch (the GUI turns the offset with the full attitude, so the gait sway of the moment must not
        be frozen into the view; a slope the robot stands on is kept), or the heading frame only (after
        a fall).  smooth: heading averaged over one trot period around t."""
        rp = self.det.robot_at(t)
        if rp is None:
            return None
        p, q = rp
        yaw = yaw_of(q)
        if smooth:
            m = self.det.robot.mean_yaw(t - self.det.delay)
            yaw = yaw if m is None else m
        if heading_only or self.level is None:
            return p, yaw_quat(yaw)
        return p, quat_rpy(self.level[1], self.level[2], yaw)

    def _rel(self, t, cam, heading_only=False, smooth=False):
        """Camera position relative to the robot in the view frame (not clamped)."""
        f = self._frame(t, heading_only, smooth)
        return None if f is None else q_inv_rot(f[1], sub(cam, f[0]))

    def _trail(self, t, heading_only=False):
        """How far the follow law trails the robot's current translation, in the view frame: v / k (the
        steady state of camera += (target - camera) * gain per frame behind a target moving at v; zero
        for a standing robot).  Turning is left out: a turn ends after a few seconds; so is the height (a
        fall / getting up is no walk; a ramp's climb is small)."""
        v = self.det.robot.velocity(t - self.det.delay, self.cfg["trail_span"])
        if v is None or math.hypot(v[0], v[1]) > 1.5:      # too few poses / a respawn (teleport): no walk
            return (0.0, 0.0, 0.0)
        v = (v[0], v[1], 0.0)
        k = max(self.det.rate(), 0.05)
        f = self._frame(t, heading_only)
        if f is None:
            return (0.0, 0.0, 0.0)
        return q_inv_rot(f[1], tuple(c / k for c in v))

    def _relative(self, t, cam, heading_only=False, smooth=False):
        """_rel as a view (clamped to the view limits)."""
        o = self._rel(t, cam, heading_only, smooth)
        if o is None:
            return None
        c = self.cfg
        return clamp_offset(o, c["min_dist"], c["max_dist"], c["min_z"])

    def _track_level(self, t, q, tau=1.0):
        roll, pitch, _ = rpy_of(q)
        if max(abs(roll), abs(pitch)) > math.radians(25.0) or self.guard.holding:
            return                            # a fall is no slope: keep the last upright value
        if self.level is None or not (0.0 < t - self.level[0] < 1.0):
            self.level = (t, roll, pitch)
            return
        k = 1.0 - math.exp(-(t - self.level[0]) / tau)
        self.level = (t, self.level[1] + k * (roll - self.level[1]), self.level[2] + k * (pitch - self.level[2]))

    def _follow_offset(self, t):
        """Offset to send for following the view now.  A view the user chose: the view + the follow law's
        trail behind the robot's current translation (module doc), so the camera is at the view while
        the robot walks and while it stands.  The default view as before (it drops back a little while
        the robot walks, the look of the demo videos), pulled in front of walls."""
        c = self.cfg
        if not self.view_auto:
            if not c["trail"]:
                return self.view
            tr = self._trail(t)
            if norm(tr) < 0.05:                   # standing: exactly the view
                return self.view
            return clamp_offset(add(self.view, tr), c["min_dist"], c["max_dist"], c["min_z"])
        if not self.walls:
            return self.view
        lt = self.det.robot.latest()
        if lt is None:
            return self.view
        _, p, q = lt
        yq = yaw_quat(yaw_of(q))
        c = add(p, q_rot(yq, self.view))
        c2 = self.walls.pull_in(p, c)
        if c2 == tuple(c):
            return self.view
        return q_inv_rot(yq, sub(c2, p))

    def _hold_offset(self):
        roll, pitch, _ = self.att
        return level_offset(self.view, roll, pitch)

    # ------------------------------------------------------------------ transitions
    def _enter_follow(self, t, mode="USE_LAST", why=""):
        if self.guard.holding:                # robot (still) down: follow = hold until it is up
            self.state = "FALL"
            self.adj = self._fall_book()
            off = self._hold_offset()
            self.hold_off, self.hold_t = off, t
            return self._send(self._req(mode, off, self.cfg["hold_gain"]), t, why)
        self.state = "FOLLOW"
        self.adj = None
        off = self._follow_offset(t)
        self.last_auto_t = t
        return self._send(self._req(mode, off, self.cfg["pos_gain"]), t, why)

    def _passive(self, t, info):
        mode = info.get("mode") or "NONE"
        what = ("follow switched off in the GUI (Esc)" if mode == "NONE" else
                f"GUI camera now {mode} {info.get('follow_target') or info.get('track_target') or ''}".rstrip())
        self.state, self.adj = "PASSIVE", None
        self.stats["passive"] += 1
        self.log(f"[follow_camera] {what} - leaving the camera alone; Gazebo window key 'c' follows again "
                 "from the current view, 'z' from the default view")

    # ------------------------------------------------------------------ inputs
    def on_robot(self, t, p, q):
        self.det.add_robot(t, p, q)
        self._track_level(t, q_normalize(q))
        c = self.cfg
        if self.state != "FOLLOW" or t < self.mute_until:
            return
        if self.view_auto and self.walls:
            interval, change, why = (c["wall_interval"] if self.fast else 1.0), c["wall_min_change"], "walls"
        elif not self.view_auto and c["trail"]:
            roll, pitch, _ = rpy_of(q_normalize(q))
            if self.att[2] or max(abs(roll), abs(pitch)) > math.radians(c["tilt_release_deg"]):
                return                            # stumbling / falling: the fall guard takes over
            interval, change, why = (c["trail_interval"] if self.fast else 2.0), c["trail_min_change"], "trail"
        else:
            return
        if t - self.last_auto_t >= interval:
            off = self._follow_offset(t)
            if norm(sub(off, self.eff)) > change or (off == self.view and off != self.eff):
                self.last_auto_t = t
                self._send(self._req("USE_LAST", off, c["pos_gain"]), t, why)

    def on_camera(self, t, pos):
        pos = tuple(float(v) for v in pos)
        if not all(math.isfinite(v) for v in pos):
            return
        self.cam = (t, pos)
        if self.state == "FOLLOW":
            if t < self.mute_until:
                self.det.restart(t, pos)
                return
            hit = self.det.update(t, pos)        # also learns the follow rate (the trail of a 'c' view)
            if hit is not None and self.cfg["adjust"]:
                self._start_adjust(t, pos, hit)
        elif self.state == "ADJUST":
            self._adjust_sample(t, pos)
        elif self.state == "FALL":
            self._fall_sample(t, pos)

    @staticmethod
    def _fall_book():
        return {"ref": None, "path": 0.0, "last": None, "t_last": None}

    def _fall_sample(self, t, pos):
        """While the camera is held for a fall: add up the camera motion the hold cannot explain (the
        user's).  The hold gain still creeps the camera towards a robot that slides / turns while it
        recovers (minutes: decimetres), which is not the user moving it."""
        a = self.adj
        if a is None:
            return
        if t < self.mute_until + 0.1:         # the hold (or its refresh) is maybe not applied yet
            a["last"] = None
            return
        c = self.cfg
        if a["ref"] is None:
            a["ref"] = pos
        if a["last"] is not None and self.hold_off is not None:
            x = self.det.explain(a["t_last"], a["last"], t, pos, self.hold_off, c["hold_gain"])
            if x is not None and norm(x[0]) > c["still_tol"]:
                a["path"] += norm(x[0])
        a["last"], a["t_last"] = pos, t

    def _start_adjust(self, t, pos, hit):
        r, s = hit
        off = self._relative(t, pos) or self.eff
        self.adj = {"t0": t, "pre": max(norm(r), norm(s)), "ref": None, "last": pos, "last_move": t,
                    "start_eff": self.eff, "t_last": t, "path": 0.0, "hold_off": off,
                    # where the user left the camera: the last camera move the follow law cannot explain,
                    # minus what the follow law added before the hold took effect
                    "user_t": t, "user_pos": pos, "follow": (0.0, 0.0, 0.0), "trig_pos": pos}
        self.state = "ADJUST"
        self.stats["holds"] += 1
        self._send(self._req("USE_LAST", off, self.cfg["hold_gain"]), t, "mouse")

    def _adjust_sample(self, t, pos):
        a = self.adj
        c = self.cfg
        if norm(sub(pos, a["last"])) > c["still_tol"]:
            a["last_move"] = t
        held = not (t < self.mute_until or a["ref"] is None)
        off, gain = (a["hold_off"], c["hold_gain"]) if held else (a["start_eff"], c["pos_gain"])
        # a blocking `gz topic -p` leaves a gap of up to ~1 s in the samples: still split it
        x = self.det.explain(a["t_last"], a["last"], t, pos, off, gain, max_gap=2.5)
        r, e = x if x is not None else (sub(pos, a["last"]), (0.0, 0.0, 0.0))
        a["follow"] = add(a["follow"], e)
        if norm(r) > c["still_tol"]:
            a["user_t"], a["user_pos"] = t, sub(pos, a["follow"])
        if not held and 0.0 < t - a["t_last"] <= self.det.max_gap:
            # the hold is (maybe) still on its way and the follow law may still move the camera: count
            # only what it cannot explain as the user's - a quick orbit can end before the hold arrives
            a["path"] += norm(r)
        a["last"], a["t_last"] = pos, t
        if t < self.mute_until:
            a["last_move"] = max(a["last_move"], t)
            a["ref"] = None
            return
        if a["ref"] is None:
            a["ref"] = pos
            a["last_move"] = max(a["last_move"], t)
            return
        if t - a["last_move"] >= c["quiet"] or t - a["t0"] >= c["max_hold"]:
            self._end_adjust(t, pos)

    def _adjusted_view(self, a, pos):
        """The view after an ADJUST: where the user left the camera, relative to the robot (sent with the
        trail, _follow_offset); None when the camera hardly moved (or no robot pose)."""
        c = self.cfg
        moved = norm(sub(pos, a["ref"])) if a["ref"] is not None and pos is not None else 0.0
        # what the user moved it in all (also inside a gap left by a blocking `gz topic -p`)
        net = norm(sub(a["user_pos"], a["trig_pos"]))
        if a["pre"] + a["path"] < c["big_step"] and moved < c["commit_min"] and net < c["commit_min"]:
            return None
        return self._relative(a["user_t"], a["user_pos"], smooth=True)

    def _end_adjust(self, t, pos, lost=False):
        a = self.adj
        c = self.cfg
        new = None if lost else self._adjusted_view(a, pos)
        if new is None:
            moved = norm(sub(pos, a["ref"])) if a["ref"] is not None else 0.0
            self.stats["reverted"] += 1
            self.state, self.adj = "FOLLOW", None
            self._send(self._req("USE_LAST", self._follow_offset(t), c["pos_gain"]), t, "revert")
            self.log(f"[follow_camera] camera hardly moved ({a['pre'] + a['path'] + moved:.2f} m"
                     + (", no camera pose" if lost else "") + ") - previous view kept")
            return
        self.view, self.view_auto = new, False
        self.stats["adopted"] += 1
        self.state, self.adj = "FOLLOW", None
        self.last_auto_t = t
        self._send(self._req("USE_LAST", self._follow_offset(t), c["pos_gain"]), t, "adopt")
        self.log(f"[follow_camera] view kept: camera {describe_view(new)} of the robot "
                 "('z' = default view, 'c' = camera stays / follows again)")

    def on_loco(self, t, d):
        if not isinstance(d, dict):
            return
        try:
            roll, pitch = float(d.get("roll")), float(d.get("pitch"))
        except (TypeError, ValueError):
            return
        down = robot_down(d)
        if math.isfinite(roll) and math.isfinite(pitch):
            self.att = (roll, pitch, down)
        c = self.cfg
        act = self.guard.update(t, roll, pitch, down) if c["fall_guard"] else None
        if act == "hold" and self.state in ("FOLLOW", "ADJUST"):
            if self.state == "ADJUST" and self.adj:
                # the user was setting a view when the robot fell: take it over now (the robot was still
                # up when the user moved the camera), the camera is held from here on
                new = self._adjusted_view(self.adj, self.cam[1] if self.cam else None)
                if new is not None:
                    self.view, self.view_auto = new, False
                    self.stats["adopted"] += 1
                    self.log(f"[follow_camera] view kept: camera {describe_view(new)} of the robot")
            self.state = "FALL"
            self.adj = self._fall_book()
            off = self._hold_offset()
            self.hold_off, self.hold_t = off, t
            self._send(self._req("USE_LAST", off, c["hold_gain"]), t, "fall")
            state = d.get("recovery_state") if down else ""
            self.log("[follow_camera] robot falling" + (f" ({state})" if state else "")
                     + " - camera held, still looking at it")
        elif act == "release" and self.state == "FALL":
            a = self.adj or self._fall_book()
            msg = "robot upright again - following"
            chosen = not self.view_auto
            if c["adjust"] and self.cam is not None and a["ref"] is not None:
                # only a camera the USER moved while it was held becomes the view (the hold's creep
                # towards a robot that slid / turned while recovering is not counted)
                moved = norm(sub(self.cam[1], a["ref"]))
                if a["path"] >= c["big_step"] and moved >= c["commit_min"]:
                    new = self._relative(self.cam[0], self.cam[1], heading_only=True)
                    if new is not None:
                        self.view, self.view_auto = new, False
                        self.stats["adopted"] += 1
                        msg += f" from the view you set while it was down ({describe_view(new)})"
                        chosen = False
            if chosen:
                msg += f" with the view you chose ({describe_view(self.view)})"
            self._enter_follow(t, "USE_LAST", "release")
            self.log("[follow_camera] " + msg)
        elif self.state == "FALL" and self.guard.holding and t - self.hold_t >= c["hold_update"]:
            off = self._hold_offset()
            if self.hold_off is None or norm(sub(off, self.hold_off)) > c["hold_min_change"]:
                self.hold_off, self.hold_t = off, t
                self._send(self._req("USE_LAST", off, c["hold_gain"]), t, "fall update")

    def on_key(self, t, code):
        try:
            code = int(code)
        except (TypeError, ValueError):
            return
        if code not in (QT_KEY_C, QT_KEY_Z):
            return
        last_code, last_t = self._key_last
        self._key_last = (code, t)
        if code == last_code and t - last_t < self.cfg["key_repeat"]:
            return                            # auto-repeat of a held key
        if code == QT_KEY_Z:
            self.view, self.view_auto = self.default, True
            self._enter_follow(t, "FOLLOW", "key z")
            self.log(f"[follow_camera] 'z': default view ({describe_view(self.default)}), following")
            return
        if self.state in ("FOLLOW", "ADJUST", "FALL"):
            self.state, self.adj = "TRIPOD", None
            self._send(TrackRequest("TRACK", self.target, self.eff, self.cfg["look_gain"], self.cfg["look_gain"]),
                       t, "key c")
            self.log("[follow_camera] 'c': camera stays here and keeps looking at the robot - move it with the "
                     "mouse, 'c' again follows from that view")
            return
        new = None
        if self.cam is not None and t - self.cam[0] <= 2.0:
            new = self._relative(self.cam[0], self.cam[1], heading_only=self.guard.holding)
        if new is not None:
            self.view, self.view_auto = new, False
            self._enter_follow(t, "FOLLOW", "key c")
            self.log(f"[follow_camera] 'c': following from this view ({describe_view(new)})")
        else:
            self._enter_follow(t, "FOLLOW", "key c")
            self.log(f"[follow_camera] 'c': following again ({describe_view(self.view)}; no camera pose seen)")

    def on_tracked(self, t, info):
        if not isinstance(info, dict):
            return
        if self.pending is not None:
            req, ts = self.pending
            if t >= ts and tracked_matches(info, req):
                self.pending = None
                self.echo_misses = 0
                self.mute_until = min(self.mute_until, t + 0.05)
                return
            if t - ts < self.cfg["echo_timeout"]:
                return                        # an older status, or the GUI has not applied it yet
            self.pending = None
            self.echo_misses += 1
        if t < self.last_send_t + 0.1:        # sampled before / while our last message went out
            return
        mode = info.get("mode") or "NONE"
        ours_follow = mode == "FOLLOW" and info.get("follow_target") == self.target
        ours_track = mode == "TRACK" and info.get("track_target") == self.target
        if self.state in ("FOLLOW", "ADJUST", "FALL"):
            if not ours_follow:
                self._passive(t, info)
        elif self.state == "TRIPOD":
            if ours_follow:                   # right-click -> Follow: our gains / view again
                self._enter_follow(t, "USE_LAST", "gui follow")
                self.log("[follow_camera] follow switched on in the GUI - following with the view "
                         f"({describe_view(self.view)})")
            elif not ours_track:
                self._passive(t, info)
        elif self.state == "PASSIVE" and ours_follow:
            self._enter_follow(t, "USE_LAST", "gui follow")
            self.log(f"[follow_camera] follow switched on in the GUI - following ({describe_view(self.view)})")

    def tick(self, t):
        """Timers: a send whose echo never came (counted: the outer loop may switch transports); an ADJUST
        whose camera samples stopped (GUI gone / hidden) ends with the old view."""
        if self.pending is not None and t - self.pending[1] > self.cfg["echo_timeout"]:
            self.pending = None
            self.echo_misses += 1
        if self.state == "ADJUST" and self.adj is not None and t - self.adj["t_last"] > 2.0:
            self._end_adjust(t, self.adj["last"], lost=True)


def parse_tracked(text):
    """`gz topic -e -t /gui/currently_tracked` text -> dict(mode, follow_target, track_target,
    follow_offset, follow_pgain) (proto3 text omits zero fields: no track_mode = NONE)."""
    text = text or ""
    m = re.search(r"track_mode:\s*(\w+)", text)
    mode = m.group(1).upper() if m else "NONE"
    if mode.isdigit():
        mode = TRACK_MODES.get(int(mode), mode)

    def name(field):
        n = re.search(field + r'\s*\{[^}]*name:\s*"([^"]*)"', text, re.S)
        return n.group(1) if n else ""
    off = None
    b = re.search(r"follow_offset\s*\{([^}]*)\}", text, re.S)
    if b:
        vals = {}
        for k in "xyz":
            v = re.search(r"\b" + k + r":\s*([-+0-9.eE]+)", b.group(1))
            vals[k] = float(v.group(1)) if v else 0.0
        off = (vals["x"], vals["y"], vals["z"])
    g = re.search(r"follow_pgain:\s*([-+0-9.eE]+)", text)
    return {"mode": mode, "follow_target": name("follow_target"), "track_target": name("track_target"),
            "follow_offset": off, "follow_pgain": float(g.group(1)) if g else None}


def tracked_from_msg(m):
    """gz.msgs.CameraTrack (bindings) -> the dict of parse_tracked."""
    has = getattr(m, "HasField", None)
    off = None
    try:
        if has is None or has("follow_offset"):
            off = (float(m.follow_offset.x), float(m.follow_offset.y), float(m.follow_offset.z))
    except ValueError:
        off = None
    return {"mode": TRACK_MODES.get(int(m.track_mode), str(m.track_mode)),
            "follow_target": str(m.follow_target.name), "track_target": str(m.track_target.name),
            "follow_offset": off, "follow_pgain": float(m.follow_pgain) or None}

