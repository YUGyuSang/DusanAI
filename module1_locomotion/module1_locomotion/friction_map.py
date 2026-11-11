"""
Friction prior of the plant: low-friction areas from world_layout.yaml
(python / numpy only, no ROS) used by the terrain classifier.

A steel plate is geometrically flat (1 cm thick), so the elevation map can
not tell it from the floor.  The layout - the single source of truth of the
digital twin - knows where the plates are and their friction coefficient:

* ``obstacles`` entries of ``type: plate`` with ``rect: [x_min, y_min,
  x_max, y_max]`` [m] and ``mu`` (C-OBS4 mu 0.2, F-OBS9 mu 0.15);
* an optional top-level ``friction_zones`` list for other low-friction areas
  (oil spill, wet floor, ...): ``{id, rect: [x_min, y_min, x_max, y_max], mu}``.

Zones are axis-aligned rectangles in the world (= map) frame.  The robot
odometry of the twin is the world-frame ground truth (odom = map, see
config/locomotion.yaml ``world_frame``), so the zones are used directly in
the odom frame of the terrain mapper.

:meth:`FrictionMap.query` returns the lowest friction coefficient of the
zones that overlap the robot footprint plus the look-ahead corridor (an
oriented rectangle in the heading frame, separating-axis test), so the
classifier can warn before the front feet reach a plate.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MU = 1.0           # friction of everything outside a zone (unknown = assume good grip)


@dataclass(frozen=True)
class FrictionZone:
    """Axis-aligned low-friction rectangle (world / odom frame)."""
    zone_id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    mu: float

    @classmethod
    def from_rect(cls, zone_id: str, rect: Sequence[float], mu: float) -> "FrictionZone":
        r = [float(v) for v in rect]
        if len(r) != 4 or not all(math.isfinite(v) for v in r):
            raise ValueError(f"friction zone {zone_id!r}: rect must be [x_min, y_min, x_max, y_max]")
        mu = float(mu)
        if not (math.isfinite(mu) and mu > 0.0):
            raise ValueError(f"friction zone {zone_id!r}: mu must be > 0, got {mu}")
        return cls(str(zone_id), min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3]), mu)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x_min - margin <= x <= self.x_max + margin
                and self.y_min - margin <= y <= self.y_max + margin)

    def corners(self, margin: float = 0.0) -> np.ndarray:
        m = float(margin)
        return np.array([[self.x_min - m, self.y_min - m], [self.x_max + m, self.y_min - m],
                         [self.x_max + m, self.y_max + m], [self.x_min - m, self.y_max + m]])

    def as_dict(self) -> Dict:
        return {"id": self.zone_id, "rect": [self.x_min, self.y_min, self.x_max, self.y_max], "mu": self.mu}


def _project(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    p = corners @ axis
    return float(p.min()), float(p.max())


def convex_quads_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test of two convex quadrilaterals (4, 2) given as
    corner lists in order; touching edges count as overlap."""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            nn = float(np.hypot(n[0], n[1]))
            if nn < 1e-12:
                continue
            n = n / nn
            a0, a1 = _project(a, n)
            b0, b1 = _project(b, n)
            if a1 < b0 - 1e-9 or b1 < a0 - 1e-9:
                return False
    return True


def corridor_corners(x: float, y: float, yaw: float, rear: float, far: float, half_width: float) -> np.ndarray:
    """World corners (4, 2) of the heading-frame rectangle
    ``-rear <= x' <= far``, ``|y'| <= half_width`` around (x, y, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[-rear, -half_width], [far, -half_width], [far, half_width], [-rear, half_width]])
    return np.stack([x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]], axis=1)


class FrictionMap:
    """Set of :class:`FrictionZone` with point / corridor queries."""

    def __init__(self, zones: Iterable[FrictionZone] = (), default_mu: float = DEFAULT_MU):
        self.zones: List[FrictionZone] = list(zones)
        self.default_mu = float(default_mu)

    def __len__(self) -> int:
        return len(self.zones)

    def __bool__(self) -> bool:
        return bool(self.zones)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_zone_dicts(cls, items: Iterable[dict], default_mu: float = DEFAULT_MU) -> "FrictionMap":
        """[{id, rect, mu}, ...] (the ``friction_zones`` format)."""
        zones = []
        for k, z in enumerate(items or []):
            if not isinstance(z, dict) or "rect" not in z or "mu" not in z:
                raise ValueError(f"friction zone #{k}: needs 'rect' and 'mu', got {z!r}")
            zones.append(FrictionZone.from_rect(z.get("id", f"zone{k}"), z["rect"], z["mu"]))
        return cls(zones, default_mu)

    @classmethod
    def from_layout(cls, data: Optional[dict], max_mu: float = 0.6,
                    default_mu: float = DEFAULT_MU) -> "FrictionMap":
        """Zones of a parsed world_layout.yaml: plates (``obstacles`` of type
        ``plate`` with ``rect`` and ``mu``) and ``friction_zones``.  Only zones
        with ``mu < max_mu`` are kept (the rest is ordinary floor)."""
        data = data or {}
        zones: List[FrictionZone] = []
        for ob in data.get("obstacles", []) or []:
            if not isinstance(ob, dict) or str(ob.get("type", "")).lower() != "plate":
                continue
            if "rect" not in ob or "mu" not in ob:
                continue
            zones.append(FrictionZone.from_rect(ob.get("id", f"plate{len(zones)}"), ob["rect"], ob["mu"]))
        extra = cls.from_zone_dicts(data.get("friction_zones", []) or [], default_mu).zones
        zones += extra
        return cls([z for z in zones if z.mu < float(max_mu)], default_mu)

    @classmethod
    def load(cls, path: str, max_mu: float = 0.6, default_mu: float = DEFAULT_MU) -> "FrictionMap":
        import yaml
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"world layout not found: {path!r}")
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_layout(yaml.safe_load(f), max_mu, default_mu)

    # ------------------------------------------------------------ queries
    def mu_at(self, x: float, y: float, margin: float = 0.0) -> float:
        """Lowest mu of the zones containing (x, y) (``default_mu`` outside)."""
        mus = [z.mu for z in self.zones if z.contains(float(x), float(y), margin)]
        return min(mus) if mus else self.default_mu

    def query(self, x: float, y: float, yaw: float, far: float, half_width: float, rear: float = 0.0,
              margin: float = 0.0) -> Tuple[float, List[str], float]:
        """Zones overlapping the heading-frame rectangle ``-rear .. far`` x
        ``+-half_width`` around the robot -> (lowest mu, zone ids, distance
        ahead [m] to the nearest overlapping zone along the heading, 0 when
        the robot stands on it).  ``(default_mu, [], inf)`` when none."""
        if not self.zones:
            return self.default_mu, [], math.inf
        quad = corridor_corners(float(x), float(y), float(yaw), float(rear), float(far), float(half_width))
        c, s = math.cos(yaw), math.sin(yaw)
        mu, ids, dist = self.default_mu, [], math.inf
        for z in self.zones:
            zc = z.corners(margin)
            if not convex_quads_overlap(quad, zc):
                continue
            ids.append(z.zone_id)
            mu = min(mu, z.mu)
            # nearest point of the zone ahead: min heading-frame x' of its corners,
            # clipped to the part of the corridor in front of the robot
            xs = (zc[:, 0] - x) * c + (zc[:, 1] - y) * s
            dist = min(dist, max(0.0, float(xs.min())))
        return mu, ids, dist

    def as_list(self) -> List[Dict]:
        return [z.as_dict() for z in self.zones]
