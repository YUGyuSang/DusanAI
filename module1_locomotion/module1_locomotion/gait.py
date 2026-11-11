"""
Gait scheduling, contact-schedule prediction, swing trajectories and static
stability helpers (ROS free, numpy only).

Phase convention
----------------
A periodic gait is described by ``period`` [s], ``duty`` (stance fraction) and,
per leg, the global phase ``offset`` at which that leg *starts its swing*.  For a
global phase ``phi`` in [0, 1) the local phase of leg i is
``l_i = (phi - offset_i) mod 1``; the leg swings while ``0 <= l_i < 1 - duty``
(swing progress ``s = l_i / (1 - duty)``) and is in stance otherwise.

Built-in gaits (leg order FL, FR, RL, RR):

=======  ======  =====  ============================  ===========================
name     period  duty   swing offsets                 remark
=======  ======  =====  ============================  ===========================
stand    -       1.00   -                             all feet on the ground
trot     0.40 s  0.50   FL 0, FR .5, RL .5, RR 0      diagonal pairs
crawl    1.20 s  0.75   FL 0, FR .5, RL .75, RR .25   4-beat FL -> RR -> FR -> RL
=======  ======  =====  ============================  ===========================

Gait transitions
----------------
Requests are queued and executed only at a *safe boundary* of the running gait so
that no foot is ever pulled away mid-swing:

* stand -> walking gait: immediately, the new gait starts at phase 0;
* walking -> stand: at the next swing-end boundary at which no other leg is in
  the middle of a swing (trot: every half cycle, crawl: every quarter);
* walking -> other walking gait: at the end of the current cycle (phase wrap),
  the new gait starts at phase 0.  At phase 0 no leg of trot or crawl is
  mid-swing, so the touch-down / lift-off sequence stays continuous.

``contacts_at`` predicts the schedule over an arbitrary future time grid
including a pending transition, which the MPC uses for its 5 s horizon.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from dusan_common import robot_model as rm

LEG_FL, LEG_FR, LEG_RL, LEG_RR = 0, 1, 2, 3
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Gait patterns
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaitPattern:
    """A periodic gait.  ``offsets`` = global phase at which each leg starts swinging."""
    name: str
    period: float
    duty: float
    offsets: tuple = (0.0, 0.0, 0.0, 0.0)

    @property
    def is_stand(self) -> bool:
        return self.duty >= 1.0 - 1e-12

    @property
    def swing_fraction(self) -> float:
        return max(0.0, 1.0 - self.duty)

    @property
    def swing_duration(self) -> float:
        return self.swing_fraction * self.period

    @property
    def stance_duration(self) -> float:
        return self.duty * self.period

    def local_phase(self, phi):
        """(…) global phase -> (…, 4) local leg phases in [0, 1).

        Phases are rounded to 1e-9 so that a phase obtained by integration
        (update) and one obtained in closed form (prediction) classify a
        boundary instant identically despite floating-point drift."""
        phi = np.asarray(phi, dtype=float)
        lp = np.round(np.mod(phi[..., None] - np.asarray(self.offsets, dtype=float), 1.0), 9)
        return np.where(lp >= 1.0, 0.0, lp)

    def contacts(self, phi) -> np.ndarray:
        """(…) global phase -> (…, 4) bool stance flags."""
        phi = np.asarray(phi, dtype=float)
        if self.is_stand:
            return np.ones(phi.shape + (4,), dtype=bool)
        return self.local_phase(phi) >= round(self.swing_fraction, 9)

    def mid_swing(self, phi) -> np.ndarray:
        """(4,) True for legs strictly inside their swing at phase ``phi``
        (a leg that is exactly starting or ending its swing is not mid-swing)."""
        if self.is_stand:
            return np.zeros(4, dtype=bool)
        lp = self.local_phase(phi)
        return (lp > 1e-7) & (lp < self.swing_fraction - 1e-7)

    def stand_boundaries(self) -> np.ndarray:
        """Sorted phases at which the gait may be stopped (all feet down)."""
        if self.is_stand:
            return np.array([0.0])
        cands = sorted({round((o + self.swing_fraction) % 1.0, 9) for o in self.offsets} | {0.0})
        good = [b for b in cands if not np.any(self.mid_swing(b))]
        return np.array(good if good else [0.0])


def make_gaits(trot_period: float = 0.40, trot_duty: float = 0.50,
               crawl_period: float = 1.20, crawl_duty: float = 0.75) -> Dict[str, GaitPattern]:
    """Standard gait library (DESIGN_SPEC §5)."""
    crawl_sf = 1.0 - crawl_duty
    # 4-beat sequence FL -> RR -> FR -> RL, one quarter cycle apart
    crawl_offsets = (0.0, 0.5, 0.75, 0.25)
    if crawl_sf > 0.25 + 1e-9:
        raise ValueError("crawl duty must be >= 0.75 (at most one leg in swing)")
    return {
        "stand": GaitPattern("stand", 1.0, 1.0, (0.0, 0.0, 0.0, 0.0)),
        "trot": GaitPattern("trot", trot_period, trot_duty, (0.0, 0.5, 0.5, 0.0)),
        "crawl": GaitPattern("crawl", crawl_period, crawl_duty, crawl_offsets),
    }


@dataclass
class GaitState:
    """Result of :meth:`GaitScheduler.update`."""
    t: float
    name: str
    phase: float
    contacts: np.ndarray                 # (4,) bool
    swing_progress: np.ndarray           # (4,) in [0,1] (0 for stance legs)
    stance_progress: np.ndarray          # (4,) in [0,1] (0 for swing legs)
    swing_duration: float
    stance_duration: float
    liftoff: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool))
    touchdown: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool))
    switched: bool = False
    pending: Optional[str] = None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
class GaitScheduler:
    """Phase-based gait scheduler with safe transitions and horizon prediction."""

    def __init__(self, gaits: Optional[Dict[str, GaitPattern]] = None, initial: str = "stand"):
        self.gaits = dict(gaits) if gaits else make_gaits()
        if "stand" not in self.gaits:
            self.gaits["stand"] = GaitPattern("stand", 1.0, 1.0)
        self.pattern = self.gaits[initial]
        self.phase = 0.0
        self.t = None
        self.pending: Optional[GaitPattern] = None
        self._contacts = self.pattern.contacts(self.phase)

    # ------------------------------------------------------------- basics
    @property
    def name(self) -> str:
        return self.pattern.name

    def snapshot(self) -> "GaitScheduler":
        """Independent copy (used to hand the schedule to the MPC worker)."""
        return copy.deepcopy(self)

    def reset(self, t: float, gait: str = "stand", phase: float = 0.0) -> None:
        self.pattern = self.gaits[gait]
        self.phase = float(phase) % 1.0
        self.t = float(t)
        self.pending = None
        self._contacts = self.pattern.contacts(self.phase)

    def request(self, gait: str) -> None:
        """Ask for a gait change; it is executed at the next safe boundary."""
        if gait not in self.gaits:
            raise KeyError(f"unknown gait '{gait}'")
        target = self.gaits[gait]
        if target.name == self.pattern.name:
            self.pending = None
        else:
            self.pending = target

    def set_pattern(self, gait: str, pattern: GaitPattern) -> None:
        """Replace a gait definition (e.g. new period from parameters)."""
        self.gaits[gait] = pattern
        if self.pattern.name == gait:
            self.pattern = pattern

    # ------------------------------------------------------------- switching
    def _switch_delay(self) -> float:
        """Time [s] from now until the pending transition is executed (inf if none)."""
        if self.pending is None:
            return math.inf
        cur = self.pattern
        if cur.is_stand:
            return 0.0
        if self.pending.is_stand:
            bounds = cur.stand_boundaries()
        else:
            bounds = np.array([0.0])
        dphi = np.mod(bounds - self.phase, 1.0)
        dphi[dphi < 1e-9] = 0.0
        return float(np.min(dphi)) * cur.period

    # ------------------------------------------------------------- update
    def update(self, t: float) -> GaitState:
        """Advance the schedule to time ``t`` and return the gait state."""
        t = float(t)
        if self.t is None:
            self.t = t
        dt = max(0.0, t - self.t)
        self.t = t
        switched = False
        prev_contacts = self._contacts.copy()
        if self.pending is not None:
            delay = self._switch_delay()
            if delay <= dt + 1e-12:
                # advance the old gait up to the boundary, then start the new one
                rest = dt - delay
                self.pattern = self.pending
                self.pending = None
                self.phase = 0.0 if self.pattern.is_stand else (rest / self.pattern.period) % 1.0
                switched = True
                dt = 0.0
        if not self.pattern.is_stand and dt > 0.0:
            self.phase = (self.phase + dt / self.pattern.period) % 1.0
        return self._state(prev_contacts, switched)

    def _state(self, prev_contacts, switched) -> GaitState:
        pat = self.pattern
        contacts = pat.contacts(self.phase)
        sw = np.zeros(4)
        st = np.zeros(4)
        if not pat.is_stand:
            lp = pat.local_phase(self.phase)
            sf = pat.swing_fraction
            sw = np.where(contacts, 0.0, np.clip(lp / max(sf, _EPS), 0.0, 1.0))
            st = np.where(contacts, np.clip((lp - sf) / max(pat.duty, _EPS), 0.0, 1.0), 0.0)
        self._contacts = contacts
        return GaitState(
            t=self.t if self.t is not None else 0.0, name=pat.name, phase=self.phase,
            contacts=contacts.copy(), swing_progress=sw, stance_progress=st,
            swing_duration=pat.swing_duration, stance_duration=pat.stance_duration,
            liftoff=prev_contacts & ~contacts, touchdown=~prev_contacts & contacts,
            switched=switched, pending=self.pending.name if self.pending else None)

    def current(self) -> GaitState:
        """Gait state at the last update time without advancing."""
        return self._state(self._contacts.copy(), False)

    # ------------------------------------------------------------- prediction
    def _segments(self, times):
        """Pattern/phase for each relative time (vectorised).  Returns
        (phase_old, phase_new, use_new mask, new pattern or None)."""
        times = np.asarray(times, dtype=float)
        cur = self.pattern
        phase_old = self.phase + (0.0 if cur.is_stand else 1.0) * times / cur.period
        delay = self._switch_delay()
        if self.pending is None or not math.isfinite(delay):
            return phase_old, None, np.zeros(times.shape, dtype=bool), None
        new = self.pending
        use_new = times >= delay - 1e-12
        phase_new = 0.0 if new.is_stand else (times - delay) / new.period
        return phase_old, np.broadcast_to(phase_new, times.shape), use_new, new

    def contacts_at(self, times) -> np.ndarray:
        """Predicted stance flags (T, 4) at relative times ``times`` [s] from the
        last update (negative times extrapolate the current gait backwards)."""
        times = np.asarray(times, dtype=float)
        phase_old, phase_new, use_new, new = self._segments(times)
        c = self.pattern.contacts(phase_old)
        if new is not None and np.any(use_new):
            c_new = new.contacts(phase_new)
            c = np.where(use_new[..., None], c_new, c)
        return c

    def pattern_at(self, t_rel: float) -> GaitPattern:
        """Gait pattern active at relative time ``t_rel``."""
        delay = self._switch_delay()
        if self.pending is not None and t_rel >= delay - 1e-12:
            return self.pending
        return self.pattern

    def sway_at(self, times, feet_xy_body, gain: float = 0.5, smooth: float = 0.5,
                dt: float = 0.01) -> np.ndarray:
        """Body-frame CoM sway offsets (T, 2) that move the CoM towards the
        centroid of the current support polygon (static stability of the
        crawl).  The raw centroid offset (support-feet centroid minus the
        centroid of all four feet) is zero for stand and for trot (diagonal
        pairs are centred) and is smoothed with a Hann window of width
        ``smooth`` seconds so the reference is continuous across beats."""
        times = np.asarray(times, dtype=float)
        if times.size == 0:
            return np.zeros((0, 2))
        feet = np.asarray(feet_xy_body, dtype=float)[:, :2]
        centre = feet.mean(axis=0)
        half = max(smooth, 2 * dt) / 2.0
        grid = np.arange(times.min() - half, times.max() + half + dt, dt)
        c = self.contacts_at(grid).astype(float)                   # (G, 4)
        n = c.sum(axis=1, keepdims=True)
        cent = np.where(n > 0, (c @ feet) / np.maximum(n, 1.0), centre) - centre
        w = np.hanning(max(3, int(round(2 * half / dt)) | 1))
        w /= w.sum()
        pad = len(w) // 2
        sm = np.stack([np.convolve(np.pad(cent[:, j], pad, mode="edge"), w, mode="valid")
                       for j in range(2)], axis=1)
        out = np.stack([np.interp(times, grid, sm[:, j]) for j in range(2)], axis=1)
        return gain * out


# ---------------------------------------------------------------------------
# Swing trajectory
# ---------------------------------------------------------------------------
def _quintic(u):
    """Minimum-jerk blend 10u^3 - 15u^4 + 6u^5 and its derivative."""
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u), 30.0 * u * u * (1.0 - u) ** 2


def _bezier3(u):
    """Cubic Bezier with control values (0, 0, 1, 1) = 3u^2 - 2u^3 and derivative."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u)


def swing_apex(p0, p1, step_height: float, obstacle_z: float = float("nan"),
               clearance: float = 0.04, max_lift: float = 0.30) -> float:
    """Swing apex height (foot centre, world z).

    apex = max( max(z0, z1) + step_height,  obstacle_top + foot_radius + clearance )
    capped at max(z0, z1) + max_lift.  ``obstacle_z`` is the highest terrain
    point along the swing (NaN if unknown)."""
    base = max(float(p0[2]), float(p1[2]))
    apex = base + step_height
    if obstacle_z is not None and math.isfinite(obstacle_z):
        apex = max(apex, float(obstacle_z) + rm.FOOT_RADIUS + clearance)
    return min(apex, base + max_lift)


class SwingTrajectory:
    """Foot swing from ``p0`` to ``p1`` (world, foot centre) through ``apex_z``.

    * horizontal: minimum-jerk (quintic) blend over s in [lam, 1 - lam];
    * vertical: cubic-Bezier rise over [0, 0.5 - lam], hold at the apex,
      cubic-Bezier descent over [0.5 + lam, 1].

    ``lam`` (lift_fraction, 0..0.3) > 0 lifts the foot before moving it
    forward and finishes the horizontal motion before the final descent —
    used when the apex is raised by an obstacle (step edge, pipe, stair).
    Zero velocity at lift-off and touch-down in all axes.
    """

    def __init__(self, p0, p1, apex_z: float, duration: float, lift_fraction: float = 0.0):
        self.p0 = np.asarray(p0, dtype=float).copy()
        self.p1 = np.asarray(p1, dtype=float).copy()
        self.apex_z = float(max(apex_z, self.p0[2], self.p1[2]))
        self.duration = max(float(duration), 1e-3)
        self.lam = float(np.clip(lift_fraction, 0.0, 0.3))

    def retarget(self, p1, apex_z: Optional[float] = None) -> None:
        """Move the landing point (allowed during the first part of the swing)."""
        self.p1 = np.asarray(p1, dtype=float).copy()
        if apex_z is not None:
            self.apex_z = float(max(apex_z, self.p0[2], self.p1[2]))
        else:
            self.apex_z = float(max(self.apex_z, self.p0[2], self.p1[2]))

    def evaluate(self, s: float):
        """Position (3,) and time-derivative velocity (3,) at swing progress s."""
        s = float(np.clip(s, 0.0, 1.0))
        lam = self.lam
        # horizontal
        span = 1.0 - 2.0 * lam
        u = (s - lam) / span
        h, dh = _quintic(u)
        dh = dh / span if 0.0 < u < 1.0 else 0.0
        pos = np.empty(3)
        vel = np.zeros(3)
        pos[:2] = self.p0[:2] + (self.p1[:2] - self.p0[:2]) * h
        vel[:2] = (self.p1[:2] - self.p0[:2]) * dh
        # vertical
        s_up, s_dn = 0.5 - lam, 0.5 + lam
        if s <= s_up:
            b, db = _bezier3(s / s_up)
            pos[2] = self.p0[2] + (self.apex_z - self.p0[2]) * b
            vel[2] = (self.apex_z - self.p0[2]) * db / s_up
        elif s < s_dn:
            pos[2] = self.apex_z
        else:
            b, db = _bezier3((s - s_dn) / (1.0 - s_dn))
            pos[2] = self.apex_z + (self.p1[2] - self.apex_z) * b
            vel[2] = (self.p1[2] - self.apex_z) * db / (1.0 - s_dn)
        return pos, vel / self.duration

    def position(self, s: float) -> np.ndarray:
        return self.evaluate(s)[0]


# ---------------------------------------------------------------------------
# Static stability
# ---------------------------------------------------------------------------
def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (monotone chain) of (n,2) points."""
    pts = np.unique(np.round(np.asarray(pts, dtype=float), 12), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _point_segment_distance(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def support_margin(feet_xy, contacts: Sequence[bool], com_xy) -> float:
    """Static stability margin [m]: signed distance of the CoM ground projection
    to the boundary of the support polygon (positive inside).  With two contacts
    the polygon is a segment and the result is minus the distance to it; with
    fewer contacts -inf."""
    feet = np.asarray(feet_xy, dtype=float)[:, :2]
    c = np.asarray(contacts, dtype=bool)
    p = np.asarray(com_xy, dtype=float)[:2]
    pts = feet[c]
    if len(pts) == 0:
        return -math.inf
    if len(pts) == 1:
        return -float(np.linalg.norm(p - pts[0]))
    hull = _convex_hull(pts)
    if len(hull) == 2:
        return -_point_segment_distance(p, hull[0], hull[1])
    margin = math.inf
    inside = True
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        e = b - a
        nrm = math.hypot(e[0], e[1])
        if nrm < 1e-12:
            continue
        # left normal points inside a CCW polygon
        sd = (e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) / nrm
        if sd < 0:
            inside = False
        margin = min(margin, sd)
    if inside:
        return float(margin)
    return -min(_point_segment_distance(p, hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull)))
