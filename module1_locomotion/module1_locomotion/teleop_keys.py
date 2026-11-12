"""
Keyboard teleop logic (ROS free, unit-tested): key decoding, key map
(DESIGN_SPEC section 7), velocity generation in 'hold' and 'latch' modes,
command events and the status line.

Key tokens are single characters ('w', ' ', '+', ...) or the names
``UP DOWN LEFT RIGHT ESC CTRL_C``.  Terminal bytes are turned into tokens by
``KeyDecoder`` (handles ANSI escape sequences for the arrow keys and tells a
lone ESC apart from the start of a sequence by a short timeout).  Gazebo's
KeyPublisher GUI plugin publishes Qt key codes, converted by
``qt_key_to_token``.

Velocity modes
--------------
hold   a motion key commands +/- max * speed_scale on its axis while it is
       pressed: every key event (OS auto-repeat) refreshes the axis, and the
       axis drops to zero ``hold_timeout`` (0.6 s) after the last event.  The
       timeout bridges the auto-repeat DELAY (typically 250-600 ms between
       the first press and the first repeat) so motion does not stutter.
latch  each press moves the axis target one step (``1 / latch_steps`` of the
       scaled maximum) and the target persists until changed / 'x'.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields as dc_fields, replace as dc_replace
from typing import Dict, List, Optional, Tuple

UP, DOWN, LEFT, RIGHT, ESC, CTRL_C = "UP", "DOWN", "LEFT", "RIGHT", "ESC", "CTRL_C"
GZ_ESC = "GZ_ESC"     # Escape pressed in the Gazebo window (stop, not quit)

# Qt::Key codes (Gazebo KeyPublisher publishes QKeyEvent::key())
QT_KEY_ESCAPE = 0x01000000
QT_KEY_LEFT = 0x01000012
QT_KEY_UP = 0x01000013
QT_KEY_RIGHT = 0x01000014
QT_KEY_DOWN = 0x01000015
_QT_SPECIAL = {QT_KEY_LEFT: LEFT, QT_KEY_UP: UP, QT_KEY_RIGHT: RIGHT, QT_KEY_DOWN: DOWN,
               QT_KEY_ESCAPE: GZ_ESC}

AXES = ("vx", "vy", "wz")

# token -> action tuple
KEYMAP: Dict[str, tuple] = {
    "w": ("move", "vx", +1), UP: ("move", "vx", +1),
    "s": ("move", "vx", -1), DOWN: ("move", "vx", -1),
    "a": ("move", "wz", +1), LEFT: ("move", "wz", +1),
    "d": ("move", "wz", -1), RIGHT: ("move", "wz", -1),
    "q": ("move", "vy", +1),
    "e": ("move", "vy", -1),
    "x": ("stop",), GZ_ESC: ("stop",),
    " ": ("estop",),
    "+": ("scale", +1), "=": ("scale", +1),
    "-": ("scale", -1), "_": ("scale", -1),
    "1": ("loco", "stand"), "2": ("loco", "sit"),
    "3": ("loco", "gait:trot"), "4": ("loco", "gait:crawl"), "5": ("loco", "gait:auto"),
    "r": ("loco", "recover"), "f": ("loco", "push"), "0": ("loco", "reset_pose"),
    "v": ("loco", "mode:toggle"),     # locomotion mode stable <-> mpc (applied while standing)
    "m": ("override",),
    "p": ("mission", "patrol"), "g": ("mission", "seek"), "h": ("mission", "rth"),
    "t": ("mission", "start"), "o": ("pause_toggle",),
    "b": ("battery", "test"), "n": ("battery", "full"),
    "?": ("help",),
    ESC: ("quit",), CTRL_C: ("quit",),
}

HELP_TEXT = """\
+------------------------------------------------------------------------+
|  DusanAI quadruped keyboard teleop                                     |
|  w/s : forward / backward        a/d : turn left / right               |
|  q/e : strafe left / right       arrows: same as w/s/a/d               |
|  x   : stop (zero velocity)      space: E-STOP toggle                  |
|  +/- : speed scale               m   : manual override toggle          |
|  1 stand  2 sit  3 trot  4 crawl  5 auto-gait                          |
|  r recover   f push test (fall)   0 reset pose                         |
|  v locomotion mode stable <-> MPC (switches while standing)            |
|  p patrol  g gas seek  h return home  t start mission  o pause/resume  |
|  b battery test -> 19 %          n battery 100 %                       |
|  ?  help                         ESC / Ctrl-C : quit                   |
|  (same keys work with focus on the Gazebo window via KeyPublisher)     |
|  Gazebo window only: c camera stays / follows again, z camera default  |
+------------------------------------------------------------------------+"""


def qt_key_to_token(code: int) -> Optional[str]:
    """Qt key code (std_msgs/Int32 from /keyboard/keypress) -> key token."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return None
    if code in _QT_SPECIAL:
        return _QT_SPECIAL[code]
    if 0x20 <= code <= 0x7E:
        return chr(code).lower()      # Qt letter codes are the upper-case ASCII values
    return None


def normalise_token(tok: str) -> str:
    """Upper-case letters (Shift / CapsLock) map to the same action."""
    if len(tok) == 1 and tok.isalpha():
        return tok.lower()
    return tok


class KeyDecoder:
    """Incremental terminal byte decoder.

    ``feed(bytes)`` returns the complete tokens; a trailing lone ESC stays
    pending (it may be the start of an arrow-key sequence) until more bytes
    arrive or ``flush()`` is called after a short timeout."""

    def __init__(self):
        self._state = 0          # 0 normal, 1 after ESC, 2 inside CSI/SS3
        self._seq = ""

    @property
    def pending(self) -> bool:
        return self._state != 0

    def feed(self, data: bytes) -> List[str]:
        out: List[str] = []
        for b in data:
            ch = chr(b)
            if self._state == 0:
                if b == 0x1B:
                    self._state = 1
                elif b == 0x03:
                    out.append(CTRL_C)
                elif b in (0x0A, 0x0D, 0x7F, 0x08, 0x09):
                    continue
                elif 0x20 <= b <= 0x7E:
                    out.append(ch)
            elif self._state == 1:
                if ch in "[O":
                    self._state = 2
                    self._seq = ""
                elif b == 0x1B:
                    out.append(ESC)          # ESC ESC -> first one is a lone ESC
                else:
                    out.append(ESC)
                    self._state = 0
                    out.extend(self.feed(bytes([b])))
            else:  # CSI / SS3 body: parameters until a final byte 0x40..0x7E
                if 0x40 <= b <= 0x7E:
                    tok = {"A": UP, "B": DOWN, "C": RIGHT, "D": LEFT}.get(ch)
                    if tok:
                        out.append(tok)
                    self._state = 0
                    self._seq = ""
                else:
                    self._seq += ch
                    if len(self._seq) > 8:   # garbage - give up
                        self._state = 0
                        self._seq = ""
        return out

    def flush(self) -> List[str]:
        """Called when no byte followed for a while: a pending ESC is a real ESC."""
        st = self._state
        self._state = 0
        self._seq = ""
        return [ESC] if st == 1 else []


@dataclass
class TeleopConfig:
    """Tunables (ROS parameters of keyboard_teleop)."""
    mode: str = "hold"               # hold | latch
    max_vx: float = 0.5              # [m/s] at speed_scale 1
    max_vy: float = 0.3              # [m/s]
    max_wz: float = 0.8              # [rad/s]
    speed_scale: float = 0.5         # initial scale
    scale_step: float = 0.1
    min_scale: float = 0.1
    max_scale: float = 1.0
    hold_timeout: float = 0.6        # [s] bridges the OS auto-repeat delay
    latch_steps: int = 5             # presses from 0 to the (scaled) maximum
    battery_test_level: float = 19.0
    battery_full_level: float = 100.0

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "TeleopConfig":
        base = cls()
        cur = {f.name: getattr(base, f.name) for f in dc_fields(base)}
        clean = {}
        for k, v in (d or {}).items():
            if k not in cur or v is None:
                continue
            dv = cur[k]
            clean[k] = (int(round(float(v))) if isinstance(dv, int) and not isinstance(dv, bool)
                        else float(v) if isinstance(dv, float) else str(v))
        cfg = dc_replace(base, **clean)
        if cfg.mode not in ("hold", "latch"):
            cfg = dc_replace(cfg, mode="hold")
        return cfg


@dataclass
class Event:
    """Side effect requested by a key press."""
    kind: str                         # locomotion | mission | battery | override | estop | quit | help | info
    value: object = None


class TeleopState:
    """Teleop state machine.  Feed tokens with ``handle_key(tok, now)`` and
    read the command with ``velocity(now)``; ``now`` is a monotonic wall
    time in seconds (key repeat is a wall-clock phenomenon)."""

    def __init__(self, cfg: Optional[TeleopConfig] = None):
        self.cfg = cfg or TeleopConfig()
        self.speed_scale = self._clamp_scale(self.cfg.speed_scale)
        self.estop = False
        self.manual_override = False
        self.mission_state: Optional[str] = None
        self._paused_local = False
        self._dir = {a: 0 for a in AXES}          # hold: direction of the last press
        self._t = {a: -math.inf for a in AXES}    # hold: time of the last press
        self._level = {a: 0 for a in AXES}        # latch: integer step level
        self.last_key: Optional[str] = None
        self.key_count = 0

    # ----------------------------------------------------------------- helpers
    def _clamp_scale(self, s: float) -> float:
        return round(min(self.cfg.max_scale, max(self.cfg.min_scale, float(s))), 3)

    def _max(self, axis: str) -> float:
        return {"vx": self.cfg.max_vx, "vy": self.cfg.max_vy, "wz": self.cfg.max_wz}[axis]

    def zero(self) -> None:
        for a in AXES:
            self._dir[a] = 0
            self._t[a] = -math.inf
            self._level[a] = 0

    # ----------------------------------------------------------------- input
    def handle_key(self, token: str, now: float) -> List[Event]:
        tok = normalise_token(token)
        act = KEYMAP.get(tok)
        if act is None:
            return []
        self.last_key = tok
        self.key_count += 1
        kind = act[0]
        if kind == "move":
            axis, sgn = act[1], act[2]
            if self.estop:
                return [Event("info", "E-STOP active - press SPACE to release")]
            if self.cfg.mode == "latch":
                n = max(1, self.cfg.latch_steps)
                self._level[axis] = max(-n, min(n, self._level[axis] + sgn))
            else:
                self._dir[axis] = sgn
                self._t[axis] = now
            return []
        if kind == "stop":
            self.zero()
            return [Event("info", "stop")]
        if kind == "estop":
            self.zero()
            self.estop = not self.estop
            cmd = "estop" if self.estop else "release"
            return [Event("estop", self.estop), Event("locomotion", cmd)]
        if kind == "scale":
            self.speed_scale = self._clamp_scale(self.speed_scale + act[1] * self.cfg.scale_step)
            return [Event("info", f"speed scale {self.speed_scale:.2f}")]
        if kind == "loco":
            return [Event("locomotion", act[1])]
        if kind == "override":
            self.manual_override = not self.manual_override
            return [Event("override", self.manual_override)]
        if kind == "mission":
            return [Event("mission", act[1])]
        if kind == "pause_toggle":
            st = (self.mission_state or "").upper()
            if st:
                cmd = "resume" if st == "PAUSED" else "pause"
            else:
                cmd = "resume" if self._paused_local else "pause"
            self._paused_local = cmd == "pause"
            return [Event("mission", cmd)]
        if kind == "battery":
            lvl = self.cfg.battery_test_level if act[1] == "test" else self.cfg.battery_full_level
            return [Event("battery", float(lvl))]
        if kind == "help":
            return [Event("help")]
        if kind == "quit":
            return [Event("quit")]
        return []

    # ---------------------------------------------------------------- output
    def velocity(self, now: float) -> Tuple[float, float, float]:
        """(vx, vy, wz) command at time ``now``."""
        if self.estop:
            return (0.0, 0.0, 0.0)
        out = []
        for a in AXES:
            vmax = self._max(a) * self.speed_scale
            if self.cfg.mode == "latch":
                n = max(1, self.cfg.latch_steps)
                out.append(vmax * self._level[a] / n)
            else:
                active = (now - self._t[a]) <= self.cfg.hold_timeout
                out.append(vmax * self._dir[a] if active else 0.0)
        return tuple(float(v) + 0.0 for v in out)

    def moving(self, now: float) -> bool:
        return any(abs(v) > 1e-9 for v in self.velocity(now))

    def streaming(self, now: float) -> bool:
        """True while /cmd_vel/teleop must be streamed (driving, manual
        override or E-STOP - the latter two keep teleop's mux priority with a
        zero command)."""
        return self.moving(now) or self.manual_override or self.estop


class CommandStreamer:
    """Rate limiter for /cmd_vel/teleop: while ``streaming`` it emits the
    command at ``rate`` Hz; on the transition to not streaming it emits
    exactly ONE zero command, then stays silent (so the cmd_vel mux can fall
    back to the autonomous sources after its timeout)."""

    def __init__(self, rate: float = 20.0):
        self.period = 1.0 / max(1e-3, float(rate))
        self._next: Optional[float] = None
        self._was_streaming = False

    def step(self, now: float, streaming: bool,
             vel: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Command to publish now, or None."""
        if self._next is not None and now < self._next:
            return None
        if self._next is not None and now - self._next < self.period:
            self._next += self.period             # keep a steady cadence
        else:
            self._next = now + self.period        # first call / fell behind
        if streaming:
            self._was_streaming = True
            return (float(vel[0]), float(vel[1]), float(vel[2]))
        if self._was_streaming:
            self._was_streaming = False
            return (0.0, 0.0, 0.0)
        return None


MISSION_STALE_AFTER = 5.0     # [s] wall clock without /mission/state -> shown as stale (mission_manager down?)


def mission_label(mission: Optional[str], t_last: Optional[float], now: float,
                  stale_after: float = MISSION_STALE_AFTER) -> Optional[str]:
    """Mission state for the status line; marked stale when /mission/state
    stopped (it is republished every 0.5 s): the last state is not the truth
    any more and a 't' would not be answered."""
    if t_last is None:
        return None
    age = now - t_last
    if age > stale_after:
        return f"{mission or '?'}(stale {age:.0f}s)"
    return mission


def mission_stale(t_last: Optional[float], now: float, stale_after: float = MISSION_STALE_AFTER) -> bool:
    """True when no /mission/state arrived for ``stale_after`` s (or never)."""
    return t_last is None or now - t_last > stale_after


def format_status(state: TeleopState, now: float, loco_mode: Optional[str] = None,
                  loco_gait: Optional[str] = None, battery_pct: Optional[float] = None,
                  mission: Optional[str] = None, gz_keys: int = 0, width: int = 160,
                  loco_algo: Optional[str] = None) -> str:
    """Compact one-line live status for the terminal (cut to ``width``);
    ``loco_algo`` = the controller's locomotion mode (stable | mpc)."""
    vx, vy, wz = state.velocity(now)
    parts = [
        f"[{state.cfg.mode}] x{state.speed_scale:.2f}",
        f"vx{vx:+.2f} vy{vy:+.2f} wz{wz:+.2f}",
        "MANUAL" if state.manual_override else "auto",
    ]
    if state.estop:
        parts.append("E-STOP")
    loco = loco_mode or "?"
    if loco_gait:
        loco += f"/{loco_gait}"
    if loco_algo:
        loco += f" [{loco_algo}]"
    parts.append(f"loco {loco}")
    parts.append(f"batt {battery_pct:.0f}%" if battery_pct is not None and math.isfinite(battery_pct)
                 else "batt ?")
    parts.append(f"mission {mission or '?'}")
    if gz_keys:
        parts.append(f"gz {gz_keys}")
    line = " | ".join(parts)
    return line[:max(10, int(width))]


class MissionCommandTracker:
    """Reliable delivery of the mission keys (t p g h o) to mission_manager.

    /mission/command is a VOLATILE topic: a key pressed before mission_manager
    is up (start-up of a loaded 8 GB laptop, DDS discovery of a teleop started
    later, a mission_manager restart) was lost, and a rejected command ('start'
    while return_to_home keeps the robot on the charger, 'start' while a
    mission runs ...) was only visible in the bringup log.  This tracker (ROS
    free, wall clock):

    * sends the key at once and keeps it until /mission/state acknowledges it
      (``cmd_seq`` / ``recent_commands``); while mission_manager is silent the
      key waits and is re-sent as soon as it answers again (expires after
      ``max_age`` s, so an old key never starts a mission much later),
    * re-sends up to ``max_sends`` times when no acknowledgement arrives within
      ``ack_timeout`` s (a repeated 'start' is harmless: rejected as 'already
      running' or ignored as a duplicate),
    * reports the outcome: accepted + message, REJECTED + reason, or gave up.

    A mission_manager without the acknowledgement fields (older build) gets
    every key once, as before."""

    def __init__(self, ack_timeout=2.0, max_sends=3, max_age=60.0, stale_after=MISSION_STALE_AFTER):
        self.ack_timeout = float(ack_timeout)
        self.max_sends = max(1, int(max_sends))
        self.max_age = float(max_age)
        self.stale_after = float(stale_after)
        self.pending = None            # dict cmd, t_key, t_sent, sends, seq0
        self.seq = None                # cmd_seq of the last /mission/state (None: unknown / old build)
        self.recent = []
        self.t_state = None

    def alive(self, now):
        return not mission_stale(self.t_state, now, self.stale_after)

    def press(self, cmd, now):
        """Key pressed -> (send_now, [messages])."""
        self.pending = {"cmd": str(cmd), "t_key": now, "t_sent": None, "sends": 0, "seq0": None}
        self._send(now)
        if self.alive(now):
            return True, []
        return True, [f"   ! mission_manager is not answering (/mission/state "
                      f"{'never received' if self.t_state is None else 'stale'}) - '{cmd}' is re-sent as soon "
                      f"as it is up (for {self.max_age:.0f} s); if it never comes up, look for a Traceback / "
                      "'process has died' in the bringup terminal"]

    def _send(self, now):
        p = self.pending
        p["sends"] += 1
        p["t_sent"] = now
        p["seq0"] = self.seq

    def on_state(self, d, now):
        """/mission/state dict -> [messages] (acknowledgement of the pending key)."""
        self.t_state = now
        if not isinstance(d, dict):
            return []
        seq = d.get("cmd_seq")
        prev = self.seq
        self.seq = int(seq) if isinstance(seq, (int, float)) and not isinstance(seq, bool) else None
        if self.pending is not None and prev is not None and self.seq is not None and self.seq < prev:
            # cmd_seq went backwards: mission_manager restarted and counts from 0 again - any record
            # of the pending command in the new process is the answer to it
            self.pending["seq0"] = -1
        rc = d.get("recent_commands")
        self.recent = [r for r in rc if isinstance(r, dict)] if isinstance(rc, list) else []
        p = self.pending
        if p is None:
            return []
        seq0 = p["seq0"] if p["seq0"] is not None else -1
        for r in self.recent:
            rs = r.get("seq")
            if isinstance(rs, (int, float)) and rs > seq0 and str(r.get("cmd")) == p["cmd"]:
                self.pending = None
                msg = str(r.get("message") or "")
                if r.get("accepted"):
                    return [f"mission: '{p['cmd']}' accepted" + (f" - {msg}" if msg else "")]
                return [f"   ! mission: '{p['cmd']}' REJECTED - {msg or 'see the bringup log'}"]
        return []

    def poll(self, now):
        """Periodic -> (command to re-send or None, [messages])."""
        p = self.pending
        if p is None:
            return None, []
        if now - p["t_key"] > self.max_age:
            self.pending = None
            return None, [f"   ! '{p['cmd']}' dropped: mission_manager did not answer within {self.max_age:.0f} s"]
        if not self.alive(now):
            return None, []               # keep it until mission_manager answers again
        if self.seq is None:
            self.pending = None           # older mission_manager without acknowledgement fields: sent, done
            return None, []
        if now - p["t_sent"] >= self.ack_timeout:
            if p["sends"] >= self.max_sends:
                self.pending = None
                return None, [f"   ! '{p['cmd']}' not acknowledged after {p['sends']} sends - check the "
                              "bringup terminal of mission_manager"]
            self._send(now)
            return p["cmd"], [f"-> /mission/command  {p['cmd']}  (re-sent: no acknowledgement yet)"]
        return None, []
