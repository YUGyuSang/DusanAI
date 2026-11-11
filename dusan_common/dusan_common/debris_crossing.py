"""
Lying-pipe crossings of the patrol / self-RTH paths (ROS free, numpy only).

A* on the terrain cost layer (``dusan_common.terrain_cost``) crosses a lying
pipe only where no free detour exists (C-OBS5 spans the corridor), on the
cheapest line through the costly cells: a zig-zag right at the pipe (Gazebo
mission check: turns of 40-45 deg with the front feet 0.3 m before the pipe
and again while the hind feet were over it; 4 of 5 runs fell there).  A
legged robot crosses a pipe best straight, square to it, at a steady pace,
without stopping, turning or strafing while a foot can reach the pipe.  Two
parts, both driven by the layout (``world_layout.yaml`` obstacles of type
pipe):

* :func:`straighten_crossings` (planning) - every crossing of a pipe inside
  its span becomes a straight segment through the original crossing point,
  square to the pipe (or tilted by up to ``max_angle`` when walls leave no
  room), from ``approach`` m before the pipe axis to ``depart`` m after it;
  the path joins it at its last vertex before that slab and leaves it at the
  first vertex after.  Every new segment is checked against the nav map
  (``line_ok``); a crossing that cannot be straightened keeps its path.
* :class:`CrossingGovernor` (path following, every control tick) - from the
  approach slab on (``zone`` + ``align_margin`` m before the pipe axis along
  the crossing) the robot first turns in place onto the crossing heading
  while its feet are still clear of the pipe (|heading error| >
  ``align_tol``), then walks over it at a constant ``speed`` with no
  sideways command and the yaw rate clipped to ``wz`` (default 0: the
  locomotion controller's heading hold keeps the heading) until its centre
  is ``zone`` m past the pipe axis (hind feet clear).
  Passing next to a pipe that is not crossed (its end, e.g. the north gap at
  C-OBS1 or the partition door end of F-OBS3): speed / yaw rate limited to
  ``near_vx`` / ``near_wz``, no strafing, within ``near_dist`` of the pipe.

Parameters (mission_manager, ``crossing_*``; :data:`CROSSING_DEFAULTS`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import wrap_angle
from .terrain_cost import _obstacles, footprint_distance

CROSSING_DEFAULTS: Dict[str, object] = {
    "crossing_enabled": True,
    "crossing_approach": 0.9,      # [m] straight run before the pipe axis (robot centre)
    "crossing_depart": 0.8,        # [m] straight run after it
    "crossing_max_angle": 0.35,    # [rad] largest tilt from square when the square line is blocked
    "crossing_end_margin": 0.15,   # [m] a path within this of a pipe end still counts as crossing it
    "crossing_max_oblique": 1.05,  # [rad] ... when it meets the pipe axis at most this far from square
    "crossing_zone": 0.6,          # [m] |centre - pipe axis| in which a foot can reach the pipe
    "crossing_align_margin": 0.3,  # [m] before the zone: turn onto the crossing heading here
    "crossing_align_tol": 0.2,     # [rad] heading error that needs the turn in place
    "crossing_align_wz": 0.4,      # [rad/s] turn rate of that alignment
    "crossing_speed": 0.25,        # [m/s] constant pace over the pipe
    "crossing_wz": 0.0,            # [rad/s] yaw rate limit while over the pipe (0: module1's heading hold
                                   #   keeps the heading; corrections twisted the body with a foot held at the pipe)
    "crossing_near_dist": 0.45,    # [m] centre to a pipe that is passed, not crossed
    "crossing_near_vx": 0.25,      # [m/s]
    "crossing_near_wz": 0.35,      # [rad/s]
}


@dataclass
class CrossingParams:
    enabled: bool = True
    approach: float = 0.9
    depart: float = 0.8
    max_angle: float = 0.35
    end_margin: float = 0.15
    max_oblique: float = 1.05
    zone: float = 0.6
    align_margin: float = 0.3
    align_tol: float = 0.2
    align_wz: float = 0.4
    speed: float = 0.25
    wz: float = 0.0
    near_dist: float = 0.45
    near_vx: float = 0.25
    near_wz: float = 0.35

    @classmethod
    def from_params(cls, p: Optional[dict], prefix: str = "crossing_") -> "CrossingParams":
        p = p or {}
        base = cls()
        kw = {}
        for f in base.__dataclass_fields__:
            v = p.get(prefix + f, getattr(base, f))
            if f == "enabled":
                kw[f] = bool(v)
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float(getattr(base, f))
            kw[f] = v if math.isfinite(v) else float(getattr(base, f))
        return cls(**kw)


class Pipe:
    """Lying pipe of the layout: axis through ``c`` along ``u``, normal ``n``,
    half length, radius."""

    def __init__(self, ob: dict):
        self.ob = ob
        self.id = str(ob.get("id", "pipe"))
        self.c = np.array([float(ob["center"][0]), float(ob["center"][1])])
        yaw = float(ob.get("yaw", 0.0))
        self.u = np.array([math.cos(yaw), math.sin(yaw)])
        self.n = np.array([-math.sin(yaw), math.cos(yaw)])
        self.half = 0.5 * float(ob["length"])
        self.r = 0.5 * float(ob["diameter"])

    def side(self, x: float, y: float) -> float:
        """Signed distance of (x, y) to the pipe axis LINE (along ``n``)."""
        return float((x - self.c[0]) * self.n[0] + (y - self.c[1]) * self.n[1])

    def along(self, x: float, y: float) -> float:
        return float((x - self.c[0]) * self.u[0] + (y - self.c[1]) * self.u[1])

    def distance(self, x: float, y: float) -> float:
        """Distance to the pipe surface footprint (0 on it)."""
        return float(footprint_distance(self.ob, x, y))


def layout_pipes(source) -> List[Pipe]:
    """The lying pipes of a Layout / layout dict / obstacle list."""
    out = []
    for ob in _obstacles(source):
        if ob.get("type") != "pipe":
            continue
        try:
            out.append(Pipe(ob))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


@dataclass
class Crossing:
    pipe: Pipe
    index: int                  # the path crosses between vertex index and index + 1
    point: np.ndarray           # crossing point on the pipe axis line
    sign: float                 # +1: travel along +n, -1: along -n
    dist: float = 0.0           # path distance from the path start to the crossing

    @property
    def direction(self) -> np.ndarray:
        return self.sign * self.pipe.n

    @property
    def heading(self) -> float:
        d = self.direction
        return math.atan2(float(d[1]), float(d[0]))

    def progress(self, x: float, y: float) -> float:
        """Signed distance of (x, y) to the pipe axis along the travel direction
        (< 0 before the pipe, > 0 past it)."""
        return self.sign * self.pipe.side(x, y)


def find_crossings(path, pipes: Sequence[Pipe], end_margin: float = 0.15, start: int = 0,
                   max_dist: float = math.inf, max_oblique: float = 1.05) -> List[Crossing]:
    """Crossings of the pipe axes inside the pipe spans (+ the pipe radius and
    ``end_margin``) by the polyline ``path`` from vertex ``start`` on, in path
    order, up to ``max_dist`` m of path.  A segment that meets the axis more
    than ``max_oblique`` rad away from square runs along the pipe (a path
    rounding a pipe end): not a crossing."""
    cos_ob = math.cos(max_oblique)
    if path is None or len(path) < 2 or not pipes:
        return []
    P = np.asarray(path, dtype=float)[:, :2]
    out = []
    acc = 0.0
    for i in range(int(start), len(P) - 1):
        a, b = P[i], P[i + 1]
        seg = float(np.hypot(*(b - a)))
        if acc > max_dist:
            break
        for pp in pipes:
            sa, sb = pp.side(a[0], a[1]), pp.side(b[0], b[1])
            if sa == sb or sa * sb > 0.0 or (sa == 0.0 and i > start):
                continue
            f = sa / (sa - sb)
            q = a + f * (b - a)
            if abs(pp.along(q[0], q[1])) > pp.half + pp.r + end_margin:
                continue
            if seg > 1e-9 and abs(float((b - a) @ pp.n)) / seg < cos_ob:
                continue
            out.append(Crossing(pp, i, q, 1.0 if sb > sa else -1.0, acc + f * seg))
        acc += seg
    out.sort(key=lambda c: c.dist)
    return out


def _rot(v: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def straighten_crossings(path, pipes: Sequence[Pipe], line_ok: Optional[Callable] = None,
                         params: Optional[CrossingParams] = None) -> Tuple[List[Tuple[float, float]], int]:
    """``path`` with every pipe crossing replaced by a straight segment square to
    the pipe (see the module docstring).  ``line_ok(a, b)`` -> bool checks a
    new segment against the map (None: every segment is fine).  Returns
    (path, number of straightened crossings).  A crossing whose start / goal
    lies inside the approach / depart slab, or that cannot be straightened
    collision free, keeps its original vertices."""
    prm = params or CrossingParams()
    P = [(float(p[0]), float(p[1])) for p in (path or [])]
    if len(P) < 2 or not pipes or not prm.enabled:
        return P, 0
    ok = line_ok or (lambda a, b: True)
    n_done = 0
    k = 0
    angles = [0.0]
    step = math.radians(10.0)
    a = step
    while a <= float(prm.max_angle) + 1e-9:
        angles += [a, -a]
        a += step
    while True:
        crs = find_crossings(P, pipes, prm.end_margin, start=k, max_oblique=prm.max_oblique)
        if not crs:
            break
        cr = crs[0]
        i = cr.index
        j0 = i
        while j0 >= 0 and cr.progress(*P[j0]) > -prm.approach + 1e-6:
            j0 -= 1
        j1 = i + 1
        while j1 < len(P) and cr.progress(*P[j1]) < prm.depart - 1e-6:
            j1 += 1
        if j0 < 0 or j1 >= len(P):
            k = i + 1                         # starts / ends next to the pipe: keep
            continue
        best = None
        for ang in angles:
            d = _rot(cr.direction, ang)
            ca = math.cos(ang)
            E = cr.point - (prm.approach / ca) * d
            X = cr.point + (prm.depart / ca) * d
            if abs(cr.pipe.along(E[0], E[1])) > cr.pipe.half + 0.5 and abs(cr.pipe.along(X[0], X[1])) > \
                    cr.pipe.half + 0.5:
                continue
            e, x = tuple(map(float, E)), tuple(map(float, X))
            if ok(P[j0], e) and ok(e, x) and ok(x, P[j1]):
                best = (e, x)
                break
        if best is None:
            k = i + 1
            continue
        P = P[:j0 + 1] + list(best) + P[j1:]
        n_done += 1
        k = j0 + 2                            # continue after the new exit vertex
    return P, n_done


class CrossingGovernor:
    """Velocity governor of the path follower near lying pipes (module docstring)."""

    def __init__(self, source, params: Optional[CrossingParams] = None):
        self.p = params or CrossingParams()
        self.pipes = layout_pipes(source) if source is not None else []
        self.active: Optional[Crossing] = None     # crossing in progress
        self.state = ""                            # "" | approach | align | cross | near
        self.n_crossings = 0

    @property
    def enabled(self) -> bool:
        return bool(self.p.enabled) and bool(self.pipes)

    def reset(self) -> None:
        self.active = None
        self.state = ""

    def _next_crossing(self, x: float, y: float, remaining) -> Optional[Crossing]:
        if remaining is None or len(remaining) < 1:
            return None
        P = np.vstack([[x, y], np.asarray(remaining, dtype=float)[:, :2]])
        crs = find_crossings(P, self.pipes, self.p.end_margin,
                             max_dist=self.p.zone + self.p.align_margin + 1.0, max_oblique=self.p.max_oblique)
        return crs[0] if crs else None

    def govern(self, x: float, y: float, yaw: float, cmd, remaining=None) -> Tuple[float, float, float]:
        """(vx, vy, wz) of the follower -> governed (vx, vy, wz).  ``remaining``:
        the path still ahead (pure pursuit remaining_path)."""
        vx, vy, wz = (float(v) for v in cmd)
        if not self.enabled:
            self.state = ""
            return vx, vy, wz
        p = self.p
        cr = self.active
        if cr is not None:
            s = cr.progress(x, y)
            if s > p.zone or abs(cr.pipe.along(x, y)) > cr.pipe.half + 1.0:
                cr = self.active = None           # hind feet clear (or walked off along the pipe)
            elif s < -(p.zone + p.align_margin + 0.3):
                nxt = self._next_crossing(x, y, remaining)
                if nxt is None or nxt.pipe is not cr.pipe:
                    cr = self.active = None       # re-planned away from it
        if cr is None:
            nxt = self._next_crossing(x, y, remaining)
            if nxt is not None and nxt.progress(x, y) >= -(p.zone + p.align_margin):
                cr = self.active = nxt
                self.n_crossings += 1
        if cr is not None:
            s = cr.progress(x, y)
            err = wrap_angle(cr.heading - yaw)
            if s < -p.zone and abs(err) > p.align_tol:
                self.state = "align"              # feet still clear of the pipe: turn onto the crossing
                return 0.0, 0.0, float(np.clip(1.5 * err, -p.align_wz, p.align_wz))
            if s < -p.zone:
                self.state = "approach"
                return min(vx, max(p.speed, 0.0)) if vx > 0.0 else max(p.speed, 0.0), 0.0, \
                    float(np.clip(wz, -p.wz, p.wz))
            self.state = "cross"                  # a foot can reach the pipe: steady, straight
            return float(p.speed), 0.0, float(np.clip(wz, -p.wz, p.wz))
        d = min((pp.distance(x, y) for pp in self.pipes), default=math.inf)
        if d < p.near_dist:
            # slower along the SAME curve (every component scaled alike): clipping only the yaw
            # rate or dropping the sideways part widens the turn into the door frame / pipe end
            self.state = "near"
            f = 1.0
            if vx > p.near_vx > 0.0:
                f = min(f, p.near_vx / vx)
            if abs(wz) > p.near_wz > 0.0:
                f = min(f, p.near_wz / abs(wz))
            return vx * f, vy * f, wz * f
        self.state = ""
        return vx, vy, wz
