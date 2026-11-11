"""
ROS-free velocity command multiplexer (DESIGN_SPEC section 4, "cmd_vel mux").

Priorities, highest first, each source with a freshness timeout (0.5 s):

    teleop  - also selected while the /teleop/active flag is true (manual
              override: a stale teleop command then yields ZERO velocity so
              the autonomy cannot drive while the operator holds control)
    rth     - selected while /safety/rth_active is true (stale -> zero, the
              robot holds still while return_to_home is planning)
    seek    - gas source seeker
    patrol  - mission manager

Selection rule:
    1. teleop flag set               -> 'teleop' (command if fresh else zero)
    2. teleop command fresh          -> 'teleop'
    3. rth flag set                  -> 'rth' (command if fresh else zero, if
                                        hold_on_stale_flag) / skip otherwise
    4. first fresh source by priority (rth, seek, patrol)
    5. nothing fresh                 -> 'none', zero velocity

Flags expire: a flag that has not been re-sent for ``flag_timeout`` seconds
counts as false (both publishers heartbeat: keyboard_teleop every 0.5 s,
return_to_home at 5 Hz), so a teleop that is killed while the manual
override is on cannot leave the robot stuck on a zero 'teleop' command.
``flag_timeout`` 0 keeps the old latched behaviour.  If the clock jumps
backwards (simulation reset) a stored flag is re-stamped with the new time,
i.e. it stays valid for one more ``flag_timeout`` and is then dropped unless
its publisher is still alive.
"""
import math
from dataclasses import dataclass, field

ZERO = (0.0, 0.0, 0.0)

MUX_DEFAULTS = {
    "rate": 20.0,                 # Hz output on /cmd_vel
    "timeout": 0.5,               # s freshness for every source
    "teleop_timeout": 0.5,
    "rth_timeout": 0.5,
    "seek_timeout": 0.5,
    "patrol_timeout": 0.5,
    "flag_timeout": 1.5,          # s validity of a flag without a new message (0 = never expire)
    "hold_on_stale_flag": True,   # flag set but command stale -> hold zero
    "max_linear": 1.0,            # m/s clamp (|vx|, |vy|)
    "max_angular": 1.5,           # rad/s clamp
    "active_publish_period": 0.5,  # s, /cmd_vel_mux/active republish period (also on change)
}

SOURCE_PRIORITY = ("teleop", "rth", "seek", "patrol")   # highest first
FLAG_FOR_SOURCE = {"teleop": "teleop_active", "rth": "rth_active"}


@dataclass
class _Slot:
    cmd: tuple = ZERO
    t: float = -math.inf


@dataclass
class _Flag:
    value: bool = False
    t: float = -math.inf


def sanitize_cmd(cmd, max_linear=1.0, max_angular=1.5):
    """Clamp a (vx, vy, wz) triple; any non-finite component -> whole command zero."""
    try:
        vx, vy, wz = (float(c) for c in cmd)
    except (TypeError, ValueError):
        return ZERO
    if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(wz)):
        return ZERO
    clamp = lambda v, m: max(-m, min(m, v))  # noqa: E731
    return (clamp(vx, max_linear), clamp(vy, max_linear), clamp(wz, max_angular))


@dataclass
class CmdVelMux:
    """Priority multiplexer.  Feed it commands/flags with timestamps, call
    :meth:`select` at the output rate."""

    timeout: float = 0.5
    timeouts: dict = field(default_factory=dict)       # per-source override
    flag_timeout: float = 1.5                          # 0 = flags latched until the next message
    hold_on_stale_flag: bool = True
    max_linear: float = 1.0
    max_angular: float = 1.5

    def __post_init__(self):
        self._slots = {name: _Slot() for name in SOURCE_PRIORITY}
        self._flags = {"teleop_active": _Flag(), "rth_active": _Flag()}

    @classmethod
    def from_params(cls, p):
        q = dict(MUX_DEFAULTS)
        q.update(p or {})
        return cls(timeout=float(q["timeout"]),
                   timeouts={s: float(q.get(f"{s}_timeout", q["timeout"])) for s in SOURCE_PRIORITY},
                   flag_timeout=float(q["flag_timeout"]),
                   hold_on_stale_flag=bool(q["hold_on_stale_flag"]),
                   max_linear=float(q["max_linear"]), max_angular=float(q["max_angular"]))

    # ----------------------------------------------------------------- inputs
    def update(self, source, cmd, t):
        if source not in self._slots:
            raise KeyError(f"unknown mux source {source!r}")
        self._slots[source] = _Slot(sanitize_cmd(cmd, self.max_linear, self.max_angular), float(t))

    def set_flag(self, name, value, t):
        if name not in self._flags:
            raise KeyError(f"unknown mux flag {name!r}")
        self._flags[name] = _Flag(bool(value), float(t))

    # ---------------------------------------------------------------- queries
    def is_fresh(self, source, t):
        slot = self._slots[source]
        return (t - slot.t) <= self.timeouts.get(source, self.timeout) and t >= slot.t - 1e-6

    def flag(self, name, t):
        f = self._flags[name]
        if not f.value:
            return False
        if self.flag_timeout > 0.0:
            if t < f.t - 1e-6:                 # clock jumped back (sim reset): restart the validity window
                self._flags[name] = f = _Flag(f.value, float(t))
            if (t - f.t) > self.flag_timeout:
                return False
        return True

    def select(self, t):
        """Return (source_name, (vx, vy, wz)).  source 'none' -> zero."""
        if self.flag("teleop_active", t):
            return "teleop", (self._slots["teleop"].cmd if self.is_fresh("teleop", t) else ZERO)
        if self.is_fresh("teleop", t):
            return "teleop", self._slots["teleop"].cmd
        if self.flag("rth_active", t):
            if self.is_fresh("rth", t):
                return "rth", self._slots["rth"].cmd
            if self.hold_on_stale_flag:
                return "rth", ZERO
        for name in ("rth", "seek", "patrol"):
            if self.is_fresh(name, t):
                return name, self._slots[name].cmd
        return "none", ZERO

    def status(self, t):
        """Diagnostics dict (freshness / flags) for logging or the dashboard."""
        return {
            "fresh": {s: self.is_fresh(s, t) for s in SOURCE_PRIORITY},
            "teleop_active": self.flag("teleop_active", t),
            "rth_active": self.flag("rth_active", t),
        }
