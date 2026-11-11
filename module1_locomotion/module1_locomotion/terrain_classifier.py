"""
Terrain classification in front of the robot (numpy only) -> /terrain/info.

Region of interest: a corridor in the robot heading frame,
``corridor_near <= x' <= corridor_far`` (default 0.3 .. 1.5 m ahead) and
``|y'| <= corridor_half_width``.  Metrics computed on the known cells:

* ``roughness``  q90 of the plane-fit residual layer            [m]
* ``max_step``   q95 of the foot-window step layer                [m]
* ``slope_deg``  dominant inclination ALONG the heading (signed, + = uphill):
                 heading component of the per-cell gradient over planar cells,
                 the larger-magnitude of its 25th / 75th percentile (so a ramp
                 occupying >= 25 % of the corridor is reported).  For stairs the
                 global least-squares incline of the corridor is reported.
* height profile along the heading (median height per 6 cm bin in a narrow
  strip) -> detrended edge detection: an "edge" is a run of <= edge_max_bins
  consecutive bins whose height jump exceeds edge_min_diff; its rise is the
  summed jump.  Rises in [stair_min_rise, stair_max_rise] are step edges,
  larger ones are walls / drop-offs.  An up-edge followed by a down-edge
  within bump_max_width is a BUMP (lying pipe), not a stair.  A step edge is
  CONFIRMED when a tread-like plateau (>= stair_min_tread long, flat within
  plateau_tol) follows it; a lone unconfirmed rise (e.g. a pipe whose back
  side is occluded) only counts as rough terrain.

Class decision (first match):
  obstacle  >= obstacle_min_cells cells with step >= obstacle_step, or an edge
            higher than stair_max_rise                   (ahead_clear = False)
  stairs    at least one confirmed step edge that is not part of a bump
  unknown   known fraction < min_known_fraction (positive obstacle / stairs
            evidence above wins: close to stairs or a tall obstacle the area
            behind is occluded, so the coverage drops legitimately)
  rough     roughness > rough_roughness, a bump or an unconfirmed step
            (incl. a jump that bridges an occluded tread)
  slope     |slope_deg| >= slope_min_deg
  flat      otherwise

recommended gait: crawl for unknown / obstacle / stairs / rough and for slopes
steeper than slope_crawl_deg, else trot.  ``TerrainClassifier`` adds
hysteresis: switching to crawl needs ``crawl_confirm_count`` consecutive
crawl votes, switching back to trot needs ``trot_hold_time`` seconds of
"trot" votes evaluated with all thresholds scaled by ``hysteresis_ratio``
(< 1, i.e. stricter), so the gait does not flicker on borderline terrain.

Friction prior (``slippery``): a steel plate is geometrically flat, so the
depth data alone classifies it as ``flat``.  With a :class:`FrictionMap`
(friction_map.py: the plates / friction zones of world_layout.yaml) the
classifier checks the robot footprint (``friction_rear`` behind base_link)
plus the look-ahead corridor (``friction_far`` ahead, ``friction_half_width``)
for zones with ``mu < slippery_mu``.  Such a zone makes the terrain
``slippery`` (it replaces flat / slope and unknown - the layout knows a plate is
flat, missing depth on shiny steel is no geometric hazard; rough, obstacle and
stairs keep their class and gait, the flag is still reported) with ``slippery_gait`` as the
recommendation, and adds ``slippery`` / ``friction_mu`` / ``friction_zones`` /
``friction_distance`` to /terrain/info.  Hysteresis: the flag is set at once
and held for ``slippery_hold_time`` after the last overlap.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, fields as dc_fields, replace as dc_replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CLASSES = ("flat", "rough", "stairs", "slope", "obstacle", "unknown", "slippery")
GAITS = ("trot", "crawl")
# geometric classes a low-friction zone overrides (obstacle / stairs keep theirs)
SLIPPERY_OVERRIDES = ("flat", "slope", "unknown")


@dataclass
class ClassifierParams:
    """Tunables (ROS parameters ``classifier.<name>``)."""
    corridor_near: float = 0.30
    corridor_far: float = 1.50
    corridor_half_width: float = 0.30
    profile_half_width: float = 0.15     # strip used for the height profile
    min_known_fraction: float = 0.30
    roughness_quantile: float = 90.0
    step_quantile: float = 95.0
    rough_roughness: float = 0.012       # [m] q90 residual above -> rough
    slope_min_deg: float = 6.0           # |slope| above -> 'slope'
    slope_crawl_deg: float = 12.0        # |slope| above -> crawl
    planar_roughness: float = 0.015      # [m] cells used for the dominant slope
    profile_bin: float = 0.06            # [m]
    profile_min_cells: int = 2
    edge_min_diff: float = 0.02          # [m] per-bin jump that starts an edge
    edge_max_bins: int = 4
    stair_min_rise: float = 0.08         # [m]
    stair_max_rise: float = 0.22         # [m] higher edges = obstacle / drop-off
    bump_max_width: float = 0.30         # [m]
    stair_min_tread: float = 0.15        # [m] plateau after an edge that confirms a step
    plateau_tol: float = 0.03            # [m] flatness of that plateau
    obstacle_step: float = 0.22          # [m]
    obstacle_min_cells: int = 3
    hysteresis_ratio: float = 0.75
    crawl_confirm_count: int = 2
    trot_hold_time: float = 1.5          # [s]
    initial_gait: str = "trot"
    # friction prior (FrictionMap from world_layout.yaml)
    use_friction_prior: bool = True
    friction_far: float = 1.0            # [m] look-ahead of the friction check
    friction_rear: float = 0.35          # [m] footprint behind base_link (rear feet)
    friction_half_width: float = 0.30    # [m]
    slippery_mu: float = 0.35            # zones with mu below are slippery
    slippery_gait: str = "trot"          # recommendation on slippery ground (physics check: crawl fell)
    slippery_hold_time: float = 1.0      # [s] flag held after the last overlap

    @classmethod
    def from_dict(cls, d: Optional[dict] = None, **overrides) -> "ClassifierParams":
        return cls().updated(**{**(d or {}), **overrides})

    def updated(self, **kw) -> "ClassifierParams":
        cur = {f.name: getattr(self, f.name) for f in dc_fields(self)}
        clean = {}
        for k, v in kw.items():
            if k not in cur or v is None:
                continue
            d = cur[k]
            if isinstance(d, bool):
                clean[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            elif isinstance(d, int):
                clean[k] = int(round(float(v)))
            elif isinstance(d, float):
                clean[k] = float(v)
            else:
                clean[k] = str(v)
        return dc_replace(self, **clean)


def _pose_xy_yaw(robot_pose: Sequence[float]) -> Tuple[float, float, float]:
    """(x, y, yaw) from a 6-tuple (x, y, z, roll, pitch, yaw) or a 3-tuple."""
    p = [float(v) for v in robot_pose]
    if len(p) >= 6:
        return p[0], p[1], p[5]
    if len(p) == 3:
        return p[0], p[1], p[2]
    raise ValueError("robot_pose must be (x, y, yaw) or (x, y, z, roll, pitch, yaw)")


def _nanpct(a: np.ndarray, q: float) -> float:
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def height_profile(x_ahead: np.ndarray, z: np.ndarray, near: float, far: float, bin_w: float,
                   min_cells: int = 2, max_gap: int = 2, return_mask: bool = False):
    """Median height per bin along the heading; interior gaps of <= max_gap
    bins are linearly interpolated, leading / trailing gaps stay NaN.
    ``return_mask`` additionally returns the boolean mask of interpolated bins."""
    nb = max(1, int(math.ceil((far - near) / bin_w - 1e-9)))
    idx = np.floor((x_ahead - near) / bin_w).astype(int)
    ok = (idx >= 0) & (idx < nb) & np.isfinite(z)
    prof = np.full(nb, np.nan)
    if ok.any():
        idx, zz = idx[ok], z[ok]
        order = np.argsort(idx, kind="stable")
        idx, zz = idx[order], zz[order]
        bins, start, counts = np.unique(idx, return_index=True, return_counts=True)
        for b, s, n in zip(bins, start, counts):
            if n >= min_cells:
                prof[b] = float(np.median(zz[s:s + n]))
    interp = np.zeros(nb, dtype=bool)
    fin = np.flatnonzero(np.isfinite(prof))
    if fin.size >= 2:
        for a, b in zip(fin[:-1], fin[1:]):
            if 1 < b - a <= max_gap + 1:
                prof[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [prof[a], prof[b]])
                interp[a + 1:b] = True
    return (prof, interp) if return_mask else prof


def detect_edges(profile: np.ndarray, near: float, bin_w: float, min_diff: float,
                 max_bins: int, min_tread: float = 0.15, plateau_tol: float = 0.03,
                 interpolated: Optional[np.ndarray] = None) -> List[Tuple[float, float, bool, bool]]:
    """Height discontinuities in a profile -> [(distance ahead, rise, confirmed, bridged)].

    The median bin-to-bin difference (the ramp trend) is removed first, so a
    constant slope never produces edges; a run of same-sign jumps longer
    than ``max_bins`` is a short ramp, not an edge.  ``confirmed`` = the
    ``min_tread`` after the edge is known and flat (a stair tread / platform).
    ``bridged`` = the edge spans interpolated bins (an occluded tread between
    two risers): its rise is not a measured wall height."""
    d = np.diff(profile)
    fin = np.isfinite(d)
    if fin.sum() < 1:
        return []
    trend = float(np.median(d[fin])) if fin.sum() >= 3 else 0.0
    dd = d - trend
    n_tread = max(2, int(math.ceil(min_tread / bin_w - 1e-9)))
    edges = []
    i, n = 0, dd.size
    while i < n:
        if not np.isfinite(dd[i]) or abs(dd[i]) <= min_diff:
            i += 1
            continue
        sgn = math.copysign(1.0, dd[i])
        j, rise = i, 0.0
        while j < n and np.isfinite(dd[j]) and math.copysign(1.0, dd[j]) == sgn and abs(dd[j]) > 0.5 * min_diff:
            rise += float(dd[j])
            j += 1
        if j - i <= max_bins:
            pos = near + ((i + j - 1) / 2.0 + 1.0) * bin_w
            after = profile[j:j + n_tread]                    # bins following the edge
            after = after - trend * np.arange(after.size)     # remove the ramp trend
            ok = (after.size == n_tread and bool(np.all(np.isfinite(after)))
                  and float(np.ptp(after)) <= plateau_tol)
            bridged = bool(interpolated is not None and np.any(interpolated[i:j + 1]))
            if bridged:
                ok = False
            edges.append((pos, rise, ok, bridged))
        i = j
    return edges


def split_bumps(edges: List[Tuple[float, float]], max_width: float):
    """Pair an edge with the NEXT edge of opposite sign closer than max_width
    (a lying pipe / beam).  Returns (remaining stair edges, number of bumps)."""
    rem, bumps, k = [], 0, 0
    while k < len(edges):
        if (k + 1 < len(edges) and edges[k][1] * edges[k + 1][1] < 0
                and edges[k + 1][0] - edges[k][0] <= max_width):
            bumps += 1
            k += 2
            continue
        rem.append(edges[k])
        k += 1
    return rem, bumps


def compute_metrics(emap, robot_pose: Sequence[float], p: ClassifierParams) -> Dict:
    """Raw corridor metrics (no class decision)."""
    x, y, yaw = _pose_xy_yaw(robot_pose)
    c, s = math.cos(yaw), math.sin(yaw)
    geo = emap._geometry()
    X, Y = emap.grid_xy()
    dx, dy = X - x, Y - y
    xf = dx * c + dy * s
    yl = -dx * s + dy * c
    corr = (xf >= p.corridor_near) & (xf <= p.corridor_far) & (np.abs(yl) <= p.corridor_half_width)
    res = emap.resolution
    expected = (p.corridor_far - p.corridor_near) * 2.0 * p.corridor_half_width / (res * res)
    n_expected = max(float(corr.sum()), expected, 1.0)
    elev = geo["elevation"]
    known = corr & np.isfinite(elev)
    nk = int(known.sum())
    nan = float("nan")
    m = {"known_fraction": nk / n_expected, "n_known": nk, "roughness": nan, "max_step": nan,
         "slope_deg": nan, "incline_deg": nan, "blocked_cells": 0, "edges": [],
         "height_change": nan}
    if nk < 5:
        return m
    z = elev[known]
    xk, yk = xf[known], yl[known]
    rough = geo["roughness"][known]
    step = geo["step"][known]
    m["roughness"] = _nanpct(rough, p.roughness_quantile)
    m["max_step"] = _nanpct(step, p.step_quantile)
    m["blocked_cells"] = int(np.sum(np.nan_to_num(step, nan=0.0) >= p.obstacle_step))
    # dominant slope along the heading over planar cells
    g = geo["gradient_x"][known] * c + geo["gradient_y"][known] * s
    planar = (np.isfinite(g) & (np.nan_to_num(rough, nan=np.inf) < p.planar_roughness)
              & (np.nan_to_num(step, nan=np.inf) < p.stair_min_rise))
    if planar.sum() >= 5:
        q25, q75 = np.percentile(g[planar], [25.0, 75.0])
        gd = q75 if abs(q75) >= abs(q25) else q25
        m["slope_deg"] = math.degrees(math.atan(gd))
    # global least-squares incline  z = a + b x' + c y'
    if nk >= 8 and np.ptp(xk) > 2 * res:
        A = np.stack([np.ones(nk), xk, yk], -1)
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        m["incline_deg"] = math.degrees(math.atan(coef[1]))
    # height profile in a narrow strip + edge detection
    strip = np.abs(yk) <= p.profile_half_width
    prof, interp = height_profile(xk[strip], z[strip], p.corridor_near, p.corridor_far, p.profile_bin,
                                  p.profile_min_cells, return_mask=True)
    m["edges"] = detect_edges(prof, p.corridor_near, p.profile_bin, p.edge_min_diff, p.edge_max_bins,
                              p.stair_min_tread, p.plateau_tol, interp)
    fin = np.flatnonzero(np.isfinite(prof))
    if fin.size >= 2:
        m["height_change"] = float(prof[fin[-1]] - prof[fin[0]])
    return m


def decide(m: Dict, p: ClassifierParams, scale: float = 1.0) -> Tuple[str, str, bool]:
    """Metrics -> (class, recommended gait, ahead_clear).  ``scale`` < 1 makes
    the rough/stairs/slope thresholds stricter (used for hysteresis).

    Positive evidence of an obstacle or of stairs is used even when the
    corridor is poorly covered: the treads of an ascending staircase above
    the camera and the area behind a tall obstacle are occluded, so close to
    such features the known fraction legitimately drops below
    ``min_known_fraction``.  Without such evidence low coverage -> unknown."""
    low_coverage = m["known_fraction"] < p.min_known_fraction
    edges = m["edges"]
    bridged = lambda e: len(e) > 3 and e[3]
    # a measured jump higher than a stair riser is a wall / drop-off; a bridged one
    # (occluded tread between risers) is only an unconfirmed step
    big = [e for e in edges if abs(e[1]) > p.stair_max_rise and not bridged(e)]
    if m["blocked_cells"] >= p.obstacle_min_cells or big:
        return "obstacle", "crawl", False
    stair_edges = [e for e in edges if abs(e[1]) >= p.stair_min_rise * scale
                   and (abs(e[1]) <= p.stair_max_rise or bridged(e))]
    remaining, bumps = split_bumps(stair_edges, p.bump_max_width)
    confirmed = [e for e in remaining if len(e) < 3 or e[2]]
    if confirmed:
        return "stairs", "crawl", True
    if low_coverage:
        return "unknown", "crawl", False
    rough = m["roughness"]
    if bumps > 0 or remaining or (math.isfinite(rough) and rough > p.rough_roughness * scale):
        return "rough", "crawl", True
    sd = m["slope_deg"]
    if math.isfinite(sd) and abs(sd) >= p.slope_min_deg * scale:
        gait = "crawl" if abs(sd) >= p.slope_crawl_deg * scale else "trot"
        return "slope", gait, True
    return "flat", "trot", True


def apply_friction(cls: str, gait: str, slippery: bool, p: ClassifierParams) -> Tuple[str, str]:
    """Class / recommended gait with the friction prior: a slippery zone
    replaces the geometric classes of SLIPPERY_OVERRIDES by ``slippery``."""
    if slippery and cls in SLIPPERY_OVERRIDES:
        g = str(p.slippery_gait).lower()
        return "slippery", (g if g in GAITS else gait)
    return cls, gait


def friction_query(friction_map, robot_pose: Sequence[float], p: ClassifierParams):
    """(mu, zone ids, distance ahead) of the low-friction zones overlapping the
    footprint + look-ahead corridor; (None, [], inf) without a map."""
    if friction_map is None or not p.use_friction_prior or len(friction_map) == 0:
        return None, [], math.inf
    x, y, yaw = _pose_xy_yaw(robot_pose)
    mu, ids, dist = friction_map.query(x, y, yaw, p.friction_far, p.friction_half_width, rear=p.friction_rear)
    return (mu if ids else None), ids, dist


def _round(v, nd=4):
    if isinstance(v, float):
        return round(v, nd) if math.isfinite(v) else None
    return v


def _result(m: Dict, cls: str, gait: str, clear: bool) -> Dict:
    slope = m["incline_deg"] if cls == "stairs" and math.isfinite(m["incline_deg"]) else m["slope_deg"]
    return {
        "roughness": _round(m["roughness"]),
        "max_step": _round(m["max_step"]),
        "slope_deg": _round(slope, 2),
        "class": cls,
        "recommended_gait": gait,
        "ahead_clear": bool(clear),
        "known_fraction": _round(m["known_fraction"], 3),
        "incline_deg": _round(m["incline_deg"], 2),
        "height_change": _round(m["height_change"]),
        "blocked_cells": int(m["blocked_cells"]),
        "edges": [[round(e[0], 3), round(e[1], 3), bool(e[2]) if len(e) > 2 else True]
                  for e in m["edges"]],
    }


def _friction_fields(slippery: bool, mu, ids, dist) -> Dict:
    return {"slippery": bool(slippery),
            "friction_mu": _round(float(mu), 3) if mu is not None else None,
            "friction_zones": list(ids),
            "friction_distance": _round(float(dist), 3) if math.isfinite(dist) else None}


def classify(emap, robot_pose: Sequence[float], params: Optional[ClassifierParams] = None,
             friction_map=None) -> Dict:
    """Stateless classification -> /terrain/info dict (no gait / friction hysteresis)."""
    p = params or ClassifierParams()
    m = compute_metrics(emap, robot_pose, p)
    cls, gait, clear = decide(m, p)
    mu, ids, dist = friction_query(friction_map, robot_pose, p)
    slippery = mu is not None and mu < p.slippery_mu
    cls, gait = apply_friction(cls, gait, slippery, p)
    out = _result(m, cls, gait, clear)
    out.update(_friction_fields(slippery, mu if slippery else None, ids if slippery else [], dist))
    return out


class TerrainClassifier:
    """Stateful classifier with gait-recommendation and friction hysteresis."""

    def __init__(self, params: Optional[ClassifierParams] = None, friction_map=None, **overrides):
        self.params = (params or ClassifierParams()).updated(**overrides)
        self.friction_map = friction_map
        self.reset()

    def set_friction_map(self, friction_map) -> None:
        """Install / replace the friction prior (a friction_map.FrictionMap or None)."""
        self.friction_map = friction_map

    def reset(self) -> None:
        g = self.params.initial_gait if self.params.initial_gait in GAITS else "trot"
        self.gait = g
        self._crawl_votes = 0
        self._trot_since: Optional[float] = None
        self._frames = 0
        self._slip_seen: Optional[float] = None     # time of the last slippery overlap
        self._slip_mu: Optional[float] = None
        self._slip_ids: List[str] = []
        self.last: Dict = {}

    def _friction(self, robot_pose: Sequence[float], t: float):
        """Hysteretic friction prior -> (slippery, mu, zone ids, distance)."""
        p = self.params
        mu, ids, dist = friction_query(self.friction_map, robot_pose, p)
        if self._slip_seen is not None and t < self._slip_seen:
            self._slip_seen = None                       # clock jumped back (simulation reset)
        if mu is not None and mu < p.slippery_mu:
            self._slip_seen = t
            self._slip_mu, self._slip_ids = float(mu), list(ids)
            return True, mu, ids, dist
        if self._slip_seen is not None and t - self._slip_seen < p.slippery_hold_time:
            return True, self._slip_mu, self._slip_ids, dist
        self._slip_seen, self._slip_mu, self._slip_ids = None, None, []
        return False, None, [], dist

    def update(self, emap, robot_pose: Sequence[float], t: Optional[float] = None) -> Dict:
        p = self.params
        self._frames += 1
        if t is None:
            t = self._frames * 0.2      # assume the nominal 5 Hz rate
        m = compute_metrics(emap, robot_pose, p)
        slippery, mu, ids, dist = self._friction(robot_pose, t)
        cls, raw_gait, clear = decide(m, p)
        cls, raw_gait = apply_friction(cls, raw_gait, slippery, p)
        if self.gait == "trot":
            self._crawl_votes = self._crawl_votes + 1 if raw_gait == "crawl" else 0
            if self._crawl_votes >= max(1, p.crawl_confirm_count):
                self.gait = "crawl"
                self._crawl_votes = 0
                self._trot_since = None
        else:
            strict_cls, strict_gait, _ = decide(m, p, p.hysteresis_ratio)
            _, strict_gait = apply_friction(strict_cls, strict_gait, slippery, p)
            if strict_gait == "trot":
                if self._trot_since is None or t < self._trot_since:   # first vote / clock reset
                    self._trot_since = t
                if t - self._trot_since >= p.trot_hold_time:
                    self.gait = "trot"
                    self._trot_since = None
            else:
                self._trot_since = None
        out = _result(m, cls, self.gait, clear)
        out["raw_gait"] = raw_gait
        out.update(_friction_fields(slippery, mu, ids, dist))
        self.last = out
        return out
