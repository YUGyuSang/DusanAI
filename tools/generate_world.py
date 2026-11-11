#!/usr/bin/env python3
"""
Generate the virtual plant for Gazebo (gz-sim 8 Harmonic ... 10 Jetty):

    simulation/worlds/plant.sdf              world 'plant' (SDF 1.9)
    simulation/models/gauge_g01..g08/        gauge stand + textured dial (OBJ)
    simulation/models/charging_station/      dock pad (marking texture) + backboard
    simulation/models/gas_leak_valve/        valve prop + smoke particle emitter
    simulation/models/access_point/          Wi-Fi AP prop (visual only)
    simulation/maps/plant_map.pgm/.yaml      2-D nav map (map_server format)
    simulation/config/generated/world_geometry.yaml   everything that was placed
    docs/images/world_layout.png, docs/images/nav_map.png

Inputs: simulation/config/world_layout.yaml (dusan_common.layout.Layout) and
the converted environment meshes simulation/models/<name>/meshes/<name>.obj
(tools/convert_environment.py).  Run on Windows or Linux (numpy, OpenCV,
PyYAML; matplotlib only for the PNG figures):

    python tools/generate_world.py              # world + models + map + figures
    python tools/generate_world.py --no-images

Pipeline
--------
1. Meshes -> world.  The OBJ meshes are placed with the layout poses (and the
   layout scale for machines).  Back faces added for double-sided rendering
   are removed again for the analysis.
2. Architecture analysis (measured, not hand-typed):
   * floor levels  = up-facing triangle clusters with large area;
   * wall faces    = robust median of the innermost occupied cell of a robot
     height slab (z = floor + 0.05..1.0 m, 2.5 cm raster) along each wall;
     doorways = gaps in that profile; interior walls (the hall/back-room
     partition at x ~ 9.8) = raster columns occupied over > 75 % of the room;
   * stairs        = clusters of up-facing faces at intermediate heights
     (each tread -> one box from the lower floor to the tread);
   * ramp          = least-squares plane z = a x + b through the up-facing
     faces of the ramp material -> rotated box (pitch = atan(-a)).
   Primitive collisions are built from these numbers so that the surfaces the
   robot walks on coincide with the visual mesh (the depth camera sees visuals).
3. Robot-height voxel test.  All meshes are voxelised (5 cm cells, 5 cm
   bands).  A cell is blocked by mesh geometry when a voxel exists between
   floor + h_lo and floor + 0.70 m, where floor is the walkable height of
   that cell (from the collision primitives) and h_lo = 0.20 m for buildings
   (stair risers of 0.18 m stay walkable) and 0.03 m for machines.  Building
   geometry not covered by the explicit primitives (columns, door jambs, wall
   pipes ...) becomes extra collision boxes; machine footprints are decomposed
   into boxes (greedy maximal rectangles).  The dense footprints themselves go
   into the nav map.
4. World content from the layout: obstacles (pipe / pallet / rubble / plate /
   drum / platform), pipe rack + thermal hot spots, gauges, gas leak prop +
   particle emitter, charging station, access points, lights, physics and
   systems.
5. Nav map (0.05 m working grid, max-pooled to the layout resolution):
   free = walkable floor (any level, incl. stairs, ramp, low debris), minus
   mesh obstacles, minus forced-occupied primitives (walls, machines, drums,
   pallets, rack supports, gauge stands, valve prop, dock backboard), minus
   drop-offs (|dz| > 0.22 m between neighbouring walkable cells, e.g. the
   ramp/stairs edge), restricted to the free region connected to the dock.

Gauge dial orientation (see build_gauges / dial_disc_vertices)
---------------------------------------------------------------
The dial OBJ lives in the gauge model frame: dial plane = local Y-Z plane at
x = 0, face normal = +X, texture up = +Z.  A viewer in front (on +X, looking
along -X, up +Z) has his right hand along +Y (right = forward x up =
(-X) x Z = +Y).  Hence image column u grows with +Y and the OBJ v (0 = image
bottom) grows with +Z:  u = 0.5 + y / D,  v = 0.5 + z / D,  D = dial_diameter.
The texture side therefore spans exactly D metres - the convention of
module2_inspection (gauge_synth / gauge_association: white face radius =
gauge_design.R_FACE * D, housing radius = R_HOUSING * D) - and the textured
disc of radius D / 2 is the inscribed circle of the texture (its outer 3 %
ring is the housing colour of the texture).  A housing cylinder of radius
D / 2 + 12 mm sits 1.5 mm behind the disc.
Fan triangles (centre, P(phi_k), P(phi_k+1)) with increasing phi are counter-
clockwise seen from +X, i.e. front facing.  The model is posed with
R = Rz(yaw) Ry(pitch), yaw = atan2(n_y, n_x), pitch = -asin(n_z), which maps
+X onto the layout normal n and keeps +Z as close to world up as possible;
the dial centre is the model origin = layout position.  Seen from the front
the image is therefore upright and not mirrored (tools/test/test_world.py
checks this by projecting the scale end points through the written OBJ).
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_HERE, os.path.join(REPO, "dusan_common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from convert_environment import (Mesh, fmt, fmt_vec, model_config_xml,  # noqa: E402
                                 read_obj, thermal_plugin_xml)
from dusan_common import gauge_design  # noqa: E402
from dusan_common.grid_map import OccupancyGridMap  # noqa: E402
from dusan_common.layout import Layout  # noqa: E402

SIM_DIR = os.path.join(REPO, "simulation")
MODELS_DIR = os.path.join(SIM_DIR, "models")
WORLD_PATH = os.path.join(SIM_DIR, "worlds", "plant.sdf")
GENERATED_DIR = os.path.join(SIM_DIR, "config", "generated")
IMAGES_DIR = os.path.join(REPO, "docs", "images")
LAYOUT_PATH = os.path.join(SIM_DIR, "config", "world_layout.yaml")
SDF_VERSION = "1.9"

# ---------------------------------------------------------------------------
# Tunables (metres unless stated)
# ---------------------------------------------------------------------------
VOX_RES = 0.05            # voxel / working grid resolution (xy)
VOX_DZ = 0.05             # voxel band height
VOX_Z_LO = -1.00          # voxelised z range (covers both corridor levels
VOX_Z_HI = 2.00           #   + robot height + collision box tops)
SLAB_RES = 0.025          # raster used to measure wall faces / doorways
CLEAR_LO_BUILDING = 0.20  # geometry below floor+0.20 is steppable (risers 0.18)
CLEAR_LO_MACHINE = 0.03   # any machine part above 3 cm blocks the robot
CLEAR_HI = 0.70           # robot height incl. sensors (+ margin)
MACHINE_NAV_HI = 1.50     # nav map: no planning under machine parts lower than this
STEP_MAX = 0.22           # larger height jumps between walkable cells = drop-off
WALL_THICKNESS = 0.25
FACTORY_WALL_HEIGHT = 4.0
FLOOR_THICKNESS = 0.30
RAMP_THICKNESS = 0.10
FLOOR_MU = 0.9
THRESHOLD_H = 0.004       # doorway threshold plate height above the floor
RAMP_DECK_DROP = 0.025    # visual deck below the perforated ramp grating
ROBOT_RADIUS = 0.30

FORCED_OCCUPIED = "occupied"   # nav classes of primitives
NAV_FREE = "free"
NAV_AUTO = "auto"
NAV_WALK = "walkable"
NAV_IGNORE = "ignore"

# SDF materials of the primitives: diffuse rgb, specular, emissive, (metalness, roughness)
MATERIALS = {
    "steel_pipe": {"diffuse": (0.56, 0.58, 0.60), "specular": 0.6, "pbr": (0.6, 0.35)},
    "rack_pipe": {"diffuse": (0.62, 0.56, 0.38), "specular": 0.4, "pbr": (0.4, 0.45)},
    "flange": {"diffuse": (0.45, 0.46, 0.48), "specular": 0.5, "pbr": (0.6, 0.4)},
    "bearing": {"diffuse": (0.30, 0.42, 0.55), "specular": 0.4, "pbr": (0.4, 0.45)},
    "steel_plate": {"diffuse": (0.50, 0.52, 0.55), "specular": 0.7, "pbr": (0.7, 0.3)},
    "rack_support": {"diffuse": (0.20, 0.33, 0.55), "specular": 0.3, "pbr": (0.3, 0.5)},
    "drum_blue": {"diffuse": (0.08, 0.22, 0.55), "specular": 0.4, "pbr": (0.3, 0.4)},
    "drum_red": {"diffuse": (0.60, 0.10, 0.08), "specular": 0.4, "pbr": (0.3, 0.4)},
    "platform": {"diffuse": (0.80, 0.62, 0.08), "specular": 0.3, "pbr": (0.2, 0.6)},
    "gauge_housing": {"diffuse": (0.22, 0.22, 0.24), "specular": 0.6, "pbr": (0.6, 0.3)},
    "gauge_stand": {"diffuse": (0.30, 0.31, 0.33), "specular": 0.3, "pbr": (0.4, 0.5)},
    "gauge_plate": {"diffuse": (0.78, 0.78, 0.74), "specular": 0.2, "pbr": (0.1, 0.7)},
    "valve_red": {"diffuse": (0.75, 0.08, 0.06), "specular": 0.4, "pbr": (0.3, 0.4)},
    "valve_body": {"diffuse": (0.85, 0.65, 0.10), "specular": 0.4, "pbr": (0.3, 0.4)},
    "dock_pad": {"diffuse": (0.12, 0.12, 0.13), "specular": 0.2, "pbr": (0.0, 0.8)},
    "dock_board": {"diffuse": (0.18, 0.19, 0.22), "specular": 0.3, "pbr": (0.2, 0.6)},
    "copper": {"diffuse": (0.80, 0.50, 0.25), "specular": 0.8, "pbr": (0.9, 0.25)},
    "led_green": {"diffuse": (0.1, 0.9, 0.2), "specular": 0.2, "emissive": (0.1, 1.0, 0.2)},
    "led_blue": {"diffuse": (0.1, 0.4, 1.0), "specular": 0.2, "emissive": (0.1, 0.4, 1.0)},
    "ap_white": {"diffuse": (0.92, 0.92, 0.90), "specular": 0.3, "pbr": (0.0, 0.5)},
    "antenna": {"diffuse": (0.10, 0.10, 0.10), "specular": 0.3, "pbr": (0.0, 0.6)},
    "filler_wall": {"diffuse": (0.60, 0.60, 0.58), "specular": 0.1, "pbr": (0.0, 0.9)},
}
RUBBLE_SHADES = [(0.52, 0.49, 0.45), (0.45, 0.43, 0.40), (0.60, 0.57, 0.52), (0.40, 0.36, 0.32),
                 (0.55, 0.52, 0.50)]


# =============================================================================
# Geometry primitives
# =============================================================================
def rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """SDF/gz convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


@dataclass
class Prim:
    """One box / cylinder of the world (collision and/or visual).

    pose = (x, y, z, roll, pitch, yaw) of the primitive centre in the frame of
    its model (world frame unless the model has a pose); cylinders follow SDF
    (axis = local z, size = (radius, length))."""
    name: str
    shape: str
    pose: tuple
    size: tuple
    group: str
    nav: str = NAV_AUTO
    collision: bool = True
    visual: bool = True
    material: str | None = None
    colour: tuple | None = None
    temperature: float | None = None
    mu: float | None = None
    kind: str = ""                    # semantic class for figures / summary
    world_pose: tuple | None = None   # pose in the world frame (set by the builder)

    def wpose(self) -> tuple:
        return self.world_pose if self.world_pose is not None else self.pose

    def rotation(self) -> np.ndarray:
        return rot_rpy(*self.wpose()[3:6])

    def sample_points(self, n_circle: int = 24) -> np.ndarray:
        """Corners (box) or cap circle points (cylinder) in the world frame."""
        R, c = self.rotation(), np.asarray(self.wpose()[:3], float)
        if self.shape == "box":
            sx, sy, sz = self.size
            loc = np.array([[a * sx / 2, b * sy / 2, d * sz / 2]
                            for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
        else:
            r, L = self.size
            ang = np.linspace(0, 2 * math.pi, n_circle, endpoint=False)
            ring = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
            loc = np.concatenate([np.c_[ring, np.full(n_circle, -L / 2)], np.c_[ring, np.full(n_circle, L / 2)]])
        return loc @ R.T + c

    def footprint(self) -> np.ndarray:
        """Convex hull (N,2) of the projection on the ground."""
        import cv2
        p = self.sample_points()[:, :2].astype(np.float32)
        return cv2.convexHull(p).reshape(-1, 2).astype(float)

    def z_range(self) -> tuple[float, float]:
        z = self.sample_points()[:, 2]
        return float(z.min()), float(z.max())

    def top_plane(self):
        """(point, unit normal) of the top face of a box (walkable surfaces)."""
        R, c = self.rotation(), np.asarray(self.wpose()[:3], float)
        n = R[:, 2]
        if n[2] < 0:
            n = -n
        return c + n * self.size[2] / 2.0, n

    def top_z_at(self, x, y):
        p, n = self.top_plane()
        return p[2] - (n[0] * (np.asarray(x) - p[0]) + n[1] * (np.asarray(y) - p[1])) / n[2]


def box(name, centre, size, group, yaw=0.0, roll=0.0, pitch=0.0, **kw) -> Prim:
    return Prim(name, "box", (centre[0], centre[1], centre[2], roll, pitch, yaw), tuple(size), group, **kw)


def box_from_bounds(name, x0, x1, y0, y1, z0, z1, group, **kw) -> Prim:
    return box(name, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), group, **kw)


def cylinder(name, centre, radius, length, group, roll=0.0, pitch=0.0, yaw=0.0, **kw) -> Prim:
    return Prim(name, "cylinder", (centre[0], centre[1], centre[2], roll, pitch, yaw), (radius, length), group, **kw)


def axis_rpy(axis) -> tuple[float, float, float]:
    """(roll, pitch, yaw) turning the local +Z axis onto `axis`."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    yaw = math.atan2(a[1], a[0])
    pitch = math.atan2(math.hypot(a[0], a[1]), a[2])
    return 0.0, pitch, yaw if math.hypot(a[0], a[1]) > 1e-9 else 0.0


# =============================================================================
# Working grid helpers
# =============================================================================
class Grid:
    """Row-major raster: cell (iy, ix) centre = (x0 + (ix+.5) res, y0 + (iy+.5) res)."""

    def __init__(self, x0, y0, x1, y1, res):
        self.x0, self.y0, self.res = float(x0), float(y0), float(res)
        self.w = int(round((x1 - x0) / res))
        self.h = int(round((y1 - y0) / res))

    @property
    def shape(self):
        return (self.h, self.w)

    def centres(self):
        xs = self.x0 + (np.arange(self.w) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.h) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    def ix(self, x):
        return int(math.floor((x - self.x0) / self.res))

    def iy(self, y):
        return int(math.floor((y - self.y0) / self.res))

    def poly_px(self, xy) -> np.ndarray:
        """World polygon -> cv2 int32 points with 4 fractional bits."""
        xy = np.asarray(xy, float)
        px = ((xy[:, 0] - self.x0) / self.res - 0.5) * 16.0
        py = ((xy[:, 1] - self.y0) / self.res - 0.5) * 16.0
        return np.round(np.stack([px, py], -1)).astype(np.int32).reshape(-1, 1, 2)

    def fill(self, img, xy, value=1, outline=True):
        import cv2
        pts = self.poly_px(xy)
        cv2.fillPoly(img, [pts], value, cv2.LINE_8, 4)
        if outline:
            cv2.polylines(img, [pts], True, value, 1, cv2.LINE_8, 4)


def clip_polygon_z(pts, z0, z1):
    """Sutherland-Hodgman clip of a 3-D polygon (list of tuples) to z0 <= z <= z1."""
    def clip(poly, keep_ge, zc):
        out = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            ka = a[2] >= zc if keep_ge else a[2] <= zc
            kb = b[2] >= zc if keep_ge else b[2] <= zc
            if ka:
                out.append(a)
            if ka != kb:
                t = (zc - a[2]) / (b[2] - a[2])
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), zc))
        return out
    p = clip(pts, True, z0)
    if not p:
        return []
    return clip(p, False, z1)


def rasterise_slab(tris: np.ndarray, grid: Grid, z0: float, z1: float, img=None, value=1) -> np.ndarray:
    """Conservative 2-D raster of everything of a triangle soup inside a z slab
    (filled projection + outline, so vertical faces are drawn as lines)."""
    import cv2
    if img is None:
        img = np.zeros(grid.shape, np.uint8)
    zmin, zmax = tris[:, :, 2].min(1), tris[:, :, 2].max(1)
    sel = np.nonzero((zmax >= z0) & (zmin <= z1))[0]
    for k in sel:
        t = tris[k]
        if zmin[k] >= z0 and zmax[k] <= z1:
            poly = t
        else:
            poly = clip_polygon_z([tuple(p) for p in t], z0, z1)
            if len(poly) < 2:
                continue
            poly = np.asarray(poly)
        pts = grid.poly_px(poly[:, :2])
        cv2.fillPoly(img, [pts], value, cv2.LINE_8, 4)
        cv2.polylines(img, [pts], True, value, 1, cv2.LINE_8, 4)
    return img


def voxelise(tris: np.ndarray, grid: Grid, z_lo=VOX_Z_LO, z_hi=VOX_Z_HI, dz=VOX_DZ,
             vox: np.ndarray | None = None) -> np.ndarray:
    """(n_bands, h, w) uint8 occupancy of a triangle soup (conservative)."""
    import cv2
    nb = int(round((z_hi - z_lo) / dz))
    if vox is None:
        vox = np.zeros((nb, grid.h, grid.w), np.uint8)
    zmin, zmax = tris[:, :, 2].min(1), tris[:, :, 2].max(1)
    xmin, xmax = tris[:, :, 0].min(1), tris[:, :, 0].max(1)
    ymin, ymax = tris[:, :, 1].min(1), tris[:, :, 1].max(1)
    gx1, gy1 = grid.x0 + grid.w * grid.res, grid.y0 + grid.h * grid.res
    sel = np.nonzero((zmax >= z_lo) & (zmin <= z_hi) & (xmax >= grid.x0) & (xmin <= gx1)
                     & (ymax >= grid.y0) & (ymin <= gy1))[0]
    for k in sel:
        t = tris[k]
        b0 = max(0, int(math.floor((zmin[k] - z_lo) / dz)))
        b1 = min(nb - 1, int(math.floor((zmax[k] - z_lo) / dz)))
        tl = [tuple(p) for p in t]
        for b in range(b0, b1 + 1):
            za, zb = z_lo + b * dz, z_lo + (b + 1) * dz
            if zmin[k] >= za and zmax[k] <= zb:
                poly = t
            else:
                poly = clip_polygon_z(tl, za, zb)
                if len(poly) < 2:
                    continue
                poly = np.asarray(poly)
            pts = grid.poly_px(poly[:, :2])
            cv2.fillPoly(vox[b], [pts], 1, cv2.LINE_8, 4)
            cv2.polylines(vox[b], [pts], True, 1, 1, cv2.LINE_8, 4)
    return vox


def band_occupied(vox: np.ndarray, floor: np.ndarray, lo: float, hi: float,
                  z_lo=VOX_Z_LO, dz=VOX_DZ) -> np.ndarray:
    """True where vox has a voxel whose band centre lies in [floor+lo, floor+hi]
    (cells with NaN floor -> False).  Cumulative sums make it O(cells)."""
    nb = vox.shape[0]
    cs = np.concatenate([np.zeros((1,) + vox.shape[1:], np.int32), np.cumsum(vox, axis=0, dtype=np.int32)])
    f = np.where(np.isfinite(floor), floor, -1e3)
    b_lo = np.clip(np.ceil((f + lo - z_lo) / dz - 0.5), 0, nb).astype(np.int64)
    b_hi = np.clip(np.floor((f + hi - z_lo) / dz - 0.5), -1, nb - 1).astype(np.int64)
    upper = np.take_along_axis(cs, (b_hi + 1)[None], 0)[0]
    lower = np.take_along_axis(cs, b_lo[None], 0)[0]
    return (upper - lower > 0) & (b_hi >= b_lo) & np.isfinite(floor)


def voxel_top(vox: np.ndarray, mask: np.ndarray, z_lo=VOX_Z_LO, dz=VOX_DZ) -> float:
    """Top z of the voxels inside a 2-D mask (NaN if none)."""
    col = vox[:, mask].any(axis=1)
    idx = np.nonzero(col)[0]
    return float(z_lo + (idx.max() + 1) * dz) if len(idx) else float("nan")


def dropoff_mask(hm: np.ndarray, step_max: float) -> np.ndarray:
    """Cells of a height map (NaN = not walkable) that have an 8-neighbour
    walkable cell more than step_max higher or lower (both sides marked)."""
    H, W = hm.shape
    drop = np.zeros(hm.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ya, yb = slice(0, H - dy), slice(dy, H)
        xa, xb = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
        a, b = hm[ya, xa], hm[yb, xb]
        with np.errstate(invalid="ignore"):
            jump = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > step_max)
        drop[ya, xa] |= jump
        drop[yb, xb] |= jump
    return drop


def reachable_free(free: np.ndarray, res: float, x0: float, y0: float, start_xy, radius: float) -> np.ndarray:
    """Free cells the robot can actually occupy: robot centres need a
    clearance >= radius (same test as dusan_common.planning.astar); the
    centres 8-connected to the start are flood filled and every free cell
    within radius of such a centre is kept.  Pockets that are only reachable
    through gaps narrower than the robot (e.g. inside machine piping) become
    occupied.  Removing them cannot reduce the clearance of a reachable centre."""
    import cv2
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * res
    ok = dist >= radius
    n, lab = cv2.connectedComponents(ok.astype(np.uint8), connectivity=8)
    ix = int(math.floor((start_xy[0] - x0) / res))
    iy = int(math.floor((start_xy[1] - y0) / res))
    seed = lab[iy, ix]
    if seed == 0:                                   # snap the start to the nearest valid centre
        ys, xs = np.nonzero(ok)
        if len(xs) == 0:
            return free
        k = int(np.argmin((xs - ix) ** 2 + (ys - iy) ** 2))
        seed = lab[ys[k], xs[k]]
    centres = (lab == seed).astype(np.uint8)
    k = 2 * int(math.ceil(radius / res)) + 3
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    near = cv2.dilate(centres, ker).astype(bool)
    return free & near


def largest_rectangle(mask: np.ndarray):
    """Largest axis-aligned all-True rectangle -> (area, r0, r1, c0, c1) (inclusive)."""
    h, w = mask.shape
    heights = np.zeros(w, np.int64)
    best = (0, 0, -1, 0, -1)
    for r in range(h):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []
        for c in range(w + 1):
            hc = heights[c] if c < w else 0
            start = c
            while stack and stack[-1][1] >= hc:
                s, hs = stack.pop()
                area = hs * (c - s)
                if area > best[0]:
                    best = (int(area), r - hs + 1, r, s, c - 1)
                start = s
            stack.append((start, hc))
    return best


def decompose_rectangles(mask: np.ndarray, min_cells: int = 4, max_rects: int = 40,
                         min_coverage: float = 0.985) -> list[tuple]:
    """Greedy cover of a binary mask by maximal rectangles (row/col inclusive).
    Stops at max_rects, when the next rectangle is smaller than min_cells or
    when min_coverage of the mask is covered."""
    m = mask.copy()
    total = max(1, int(mask.sum()))
    out = []
    while m.any() and len(out) < max_rects:
        area, r0, r1, c0, c1 = largest_rectangle(m)
        if area < min_cells:
            break
        out.append((r0, r1, c0, c1))
        m[r0:r1 + 1, c0:c1 + 1] = False
        if 1.0 - m.sum() / total >= min_coverage:
            break
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes of a binary mask (flood fill from the border)."""
    import cv2
    m = mask.astype(np.uint8)
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    h, w = ff.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    holes = ff[1:-1, 1:-1] == 0
    return mask | holes


# =============================================================================
# Mesh loading / measurements
# =============================================================================
@dataclass
class PlacedMesh:
    name: str
    tris: np.ndarray          # (F,3,3) world frame, single sided
    mats: np.ndarray          # (F,) material names
    normals: np.ndarray       # (F,3)
    areas: np.ndarray         # (F,)
    pose: tuple
    scale: float


def load_placed_mesh(name: str, pose, scale: float = 1.0, models_dir: str = MODELS_DIR) -> PlacedMesh:
    """Read simulation/models/<name>/meshes/<name>.obj and place it in the world
    (layout pose x, y, z, yaw and uniform scale).  Duplicate back faces
    (double-sided rendering) are dropped."""
    m = read_obj(os.path.join(models_dir, name, "meshes", name + ".obj"))
    key = np.sort(m.fv, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    x, y, z, yaw = (list(pose) + [0.0] * 4)[:4]
    R = rot_rpy(0, 0, yaw) * scale
    v = m.v @ R.T + np.array([x, y, z])
    tris = v[m.fv[first]]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    a = np.linalg.norm(n, axis=1)
    return PlacedMesh(name, tris, m.material_names()[first], n / np.maximum(a, 1e-12)[:, None],
                      0.5 * a, tuple(pose), float(scale))


def floor_levels(pm: PlacedMesh, min_area=5.0, tol=0.01) -> list[dict]:
    """Large horizontal up-facing surfaces -> [{'z', 'area', 'x0','x1','y0','y1'}]."""
    up = pm.normals[:, 2] > 0.98
    z = pm.tris[up, :, 2].mean(1)
    a = pm.areas[up]
    t = pm.tris[up]
    order = np.argsort(z)
    out, cur = [], [order[0]] if len(order) else []
    for i, j in zip(order[:-1], order[1:]):
        if z[j] - z[i] > tol:
            out.append(cur)
            cur = []
        cur.append(j)
    if cur:
        out.append(cur)
    res = []
    for idx in out:
        idx = np.asarray(idx)
        area = float(a[idx].sum())
        if area < min_area:
            continue
        tt = t[idx]
        res.append({"z": float(np.average(z[idx], weights=a[idx])), "area": area,
                    "x0": float(tt[:, :, 0].min()), "x1": float(tt[:, :, 0].max()),
                    "y0": float(tt[:, :, 1].min()), "y1": float(tt[:, :, 1].max())})
    return res


def wall_profile(occ: np.ndarray, grid: Grid, axis: str, approx: float, inner_sign: int,
                 s_range: tuple, search: float = 0.5) -> tuple[float, np.ndarray, np.ndarray]:
    """Innermost occupied coordinate along a wall.

    axis 'x': wall of constant x (scan rows y in s_range); inner_sign = +1 if the
    room lies at larger x (take the max occupied x), -1 otherwise.  axis 'y'
    likewise.  Returns (median face coordinate, scan positions, per-scan face or NaN)."""
    s = np.arange(s_range[0], s_range[1], grid.res)
    vals = np.full(len(s), np.nan)
    if axis == "x":
        c0, c1 = grid.ix(approx - search), grid.ix(approx + search)
        c0, c1 = max(c0, 0), min(c1, grid.w - 1)
        coords = grid.x0 + (np.arange(c0, c1 + 1) + 0.5) * grid.res
        for k, yy in enumerate(s):
            r = grid.iy(yy)
            if not 0 <= r < grid.h:
                continue
            hit = np.nonzero(occ[r, c0:c1 + 1])[0]
            if len(hit):
                vals[k] = coords[hit.max()] if inner_sign > 0 else coords[hit.min()]
    else:
        r0, r1 = grid.iy(approx - search), grid.iy(approx + search)
        r0, r1 = max(r0, 0), min(r1, grid.h - 1)
        coords = grid.y0 + (np.arange(r0, r1 + 1) + 0.5) * grid.res
        for k, xx in enumerate(s):
            c = grid.ix(xx)
            if not 0 <= c < grid.w:
                continue
            hit = np.nonzero(occ[r0:r1 + 1, c])[0]
            if len(hit):
                vals[k] = coords[hit.max()] if inner_sign > 0 else coords[hit.min()]
    # the face lies inside the marked cell: its centre is the unbiased estimate (+-res/2)
    face = float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else float("nan")
    return face, s, vals


def gaps_in_profile(s: np.ndarray, vals: np.ndarray, min_len: float = 0.4, res: float = SLAB_RES) -> list[tuple]:
    """Runs of NaN (no wall) in a wall profile -> [(start, end)] (cell edges)."""
    nan = ~np.isfinite(vals)
    d = np.diff(np.r_[0, nan.astype(int), 0])
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    out = []
    for a, b in zip(st, en):
        lo, hi = s[a] - res / 2, s[b - 1] + res / 2
        if hi - lo >= min_len:
            out.append((float(lo), float(hi)))
    return out


def _spatial_groups(boxes: np.ndarray, margin: float) -> list[np.ndarray]:
    """Union-find of axis-aligned 2-D boxes (N,4: x0,x1,y0,y1) that overlap
    after growing by `margin` -> list of index arrays."""
    n = len(boxes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    order = np.argsort(boxes[:, 0])
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if boxes[j, 0] > boxes[i, 1] + margin:
                break
            if boxes[j, 2] <= boxes[i, 3] + margin and boxes[i, 2] <= boxes[j, 3] + margin:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    return [np.nonzero(roots == r)[0] for r in np.unique(roots)]


def detect_treads(pm: PlacedMesh, region, z_lo: float, z_hi: float, exclude_mat: str | None = None,
                  min_area=0.12, min_side=0.20, compact=0.55, tol=0.012, margin=0.03,
                  min_rise=0.08, max_area=4.0) -> list[dict]:
    """Stair treads = horizontal up-facing faces at heights strictly between
    z_lo and z_hi inside region (x0, x1, y0, y1), clustered in z (tread faces
    are exactly flat) and then split into spatially connected groups (so that
    a tread is never merged with a trim strip at a similar height elsewhere).
    Groups must be compact (area / bbox area) and at least min_side wide."""
    x0, x1, y0, y1 = region
    c = pm.tris.mean(1)
    sel = (pm.normals[:, 2] > 0.95) & (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1) \
        & (c[:, 2] > z_lo + min_rise) & (c[:, 2] < z_hi - min_rise)
    # glass shards / decals lying on the floor are not treads
    sel &= np.array([not any(k in str(m).lower() for k in ("window", "glass")) for m in pm.mats])
    if exclude_mat:
        sel &= np.array([exclude_mat not in str(m).lower() for m in pm.mats])
    idx = np.nonzero(sel)[0]
    if len(idx) == 0:
        return []
    z = c[idx, 2]
    order = np.argsort(z)
    zgroups, cur = [], [idx[order[0]]]
    for a, b in zip(order[:-1], order[1:]):
        if z[b] - z[a] > tol:
            zgroups.append(cur)
            cur = []
        cur.append(idx[b])
    zgroups.append(cur)
    out = []
    for zg in zgroups:
        zg = np.asarray(zg)
        t = pm.tris[zg]
        bx = np.stack([t[:, :, 0].min(1), t[:, :, 0].max(1), t[:, :, 1].min(1), t[:, :, 1].max(1)], 1)
        for sg in _spatial_groups(bx, margin):
            g = zg[sg]
            tt = pm.tris[g]
            area = float(pm.areas[g].sum())
            bx0, bx1 = float(tt[:, :, 0].min()), float(tt[:, :, 0].max())
            by0, by1 = float(tt[:, :, 1].min()), float(tt[:, :, 1].max())
            w, h = bx1 - bx0, by1 - by0
            if area < min_area or area > max_area or min(w, h) < min_side or area / max(w * h, 1e-9) < compact:
                continue
            out.append({"z": float(np.average(c[g, 2], weights=pm.areas[g])), "x0": bx0, "x1": bx1,
                        "y0": by0, "y1": by1, "area": area})
    return stair_flights(out)


def stair_flights(treads: list[dict], rise=(0.08, 0.26), run=(0.15, 0.70), min_len=3) -> list[dict]:
    """Keep only treads that form a flight: a chain of treads where each next
    tread is higher by a plausible riser, offset horizontally by a plausible
    run and overlapping sideways (floor planks of a mezzanine, shelves etc.
    at a similar height are rejected)."""
    tr = sorted(treads, key=lambda q: q["z"])
    n = len(tr)
    cen = np.array([[(q["x0"] + q["x1"]) / 2, (q["y0"] + q["y1"]) / 2] for q in tr]).reshape(-1, 2)
    nxt = [-1] * n
    for i in range(n):
        best = None
        for j in range(i + 1, n):
            dz = tr[j]["z"] - tr[i]["z"]
            if dz < rise[0]:
                continue
            if dz > rise[1]:
                break
            d = float(np.hypot(*(cen[j] - cen[i])))
            ox = min(tr[i]["x1"], tr[j]["x1"]) - max(tr[i]["x0"], tr[j]["x0"])
            oy = min(tr[i]["y1"], tr[j]["y1"]) - max(tr[i]["y0"], tr[j]["y0"])
            if run[0] <= d <= run[1] and max(ox, oy) > 0.15 and (best is None or d < best[0]):
                best = (d, j)
        if best is not None:
            nxt[i] = best[1]
    has_prev = set(j for j in nxt if j >= 0)
    keep = set()
    for i in range(n):
        if i in has_prev:
            continue
        chain = [i]
        while nxt[chain[-1]] >= 0:
            chain.append(nxt[chain[-1]])
        if len(chain) >= min_len:
            keep.update(chain)
    return [tr[i] for i in sorted(keep)]


def fit_ramp(pm: PlacedMesh, material_key: str = "uphill") -> dict | None:
    """Plane z = a x + b through the up-facing faces of the ramp material."""
    sel = np.array([material_key in str(m).lower() for m in pm.mats]) & (pm.normals[:, 2] > 0.5)
    if sel.sum() < 3:
        return None
    P = pm.tris[sel].reshape(-1, 3)
    A = np.c_[P[:, 0], np.ones(len(P))]
    (a, b), *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    t = pm.tris[np.array([material_key in str(m).lower() for m in pm.mats])]
    up = pm.tris[sel]
    return {"a": float(a), "b": float(b), "x0": float(up[:, :, 0].min()), "x1": float(up[:, :, 0].max()),
            "y0": float(up[:, :, 1].min()), "y1": float(up[:, :, 1].max()),
            "slope_deg": float(math.degrees(math.atan(abs(a)))),
            "residual_max": float(np.abs(A @ (a, b) - P[:, 2]).max()), "n_faces": int(sel.sum()),
            "bbox_z": [float(t[:, :, 2].min()), float(t[:, :, 2].max())]}


def light_clusters(pm: PlacedMesh, key: str = "light", link: float = 0.6) -> list[np.ndarray]:
    """Centroids of connected groups of emissive 'Light' faces (lamp positions)."""
    sel = np.array([key in str(m).lower() for m in pm.mats])
    if not sel.any():
        return []
    c = pm.tris[sel].mean(1)
    a = pm.areas[sel]
    n = len(c)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    order = np.argsort(c[:, 0])
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if c[j, 0] - c[i, 0] > link:
                break
            if np.linalg.norm(c[j] - c[i]) < link:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    out = []
    for r in np.unique(roots):
        m = roots == r
        out.append(np.average(c[m], axis=0, weights=a[m] + 1e-9))
    return out


# =============================================================================
# World builder
# =============================================================================
class WorldBuilder:
    def __init__(self, layout: Layout, models_dir: str = MODELS_DIR, log=print):
        self.L = layout
        self.models_dir = models_dir
        self.log = log
        self.prims: list[Prim] = []
        self.models: dict[str, dict] = {}       # SDF model name -> {'pose','static','prims','extra'}
        self.includes: list[dict] = []           # <include> entries
        self.lights: list[dict] = []
        self.geometry: dict = {}                 # measured architecture (summary yaml)
        self.notes: list[str] = []
        self.meshes: dict[str, PlacedMesh] = {}
        self.machine_records: list[dict] = []
        self.gauge_frames: list[dict] = []
        nb = self.L["nav_map"]["bounds"]
        self.grid = Grid(nb[0], nb[1], nb[2], nb[3], VOX_RES)

    # ------------------------------------------------------------------ utils
    def add(self, prim: Prim, model: str | None = None, model_pose=(0, 0, 0, 0, 0, 0)) -> Prim:
        """Register a primitive inside an SDF model (pose relative to the model)."""
        model = model or prim.group
        rec = self.models.setdefault(model, {"pose": tuple(model_pose), "static": True, "prims": [],
                                             "extra": ""})
        mp = rec["pose"]
        if any(abs(v) > 1e-12 for v in mp):
            Rm = rot_rpy(*mp[3:6])
            c = Rm @ np.asarray(prim.pose[:3]) + np.asarray(mp[:3])
            Rw = Rm @ rot_rpy(*prim.pose[3:6])
            pitch = -math.asin(max(-1.0, min(1.0, Rw[2, 0])))
            roll = math.atan2(Rw[2, 1], Rw[2, 2])
            yaw = math.atan2(Rw[1, 0], Rw[0, 0])
            prim.world_pose = (c[0], c[1], c[2], roll, pitch, yaw)
        rec["prims"].append(prim)
        self.prims.append(prim)
        return prim

    def walkable(self) -> list[Prim]:
        return [p for p in self.prims if p.nav == NAV_WALK]

    def floor_at(self, x: float, y: float, default: float | None = None) -> float:
        """Walkable surface height at (x, y) from the architecture primitives."""
        import cv2
        best = None
        for p in self.walkable():
            if p.kind not in ("floor", "tread", "ramp", "landing"):
                continue
            fp = p.footprint().astype(np.float32)
            if cv2.pointPolygonTest(fp.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0:
                z = float(p.top_z_at(x, y))
                best = z if best is None else max(best, z)
        if best is None:
            best = default if default is not None else float(self.L.floor_z(x, y))
        return best

    # ----------------------------------------------------------- architecture
    def build_architecture(self):
        L = self.L
        fac, cor = L["factory"], L["corridor"]
        self.log("== meshes")
        F = self.meshes["factory_hall"] = load_placed_mesh(fac["mesh_model"], fac["pose"], 1.0, self.models_dir)
        C = self.meshes["corridor"] = load_placed_mesh(cor["mesh_model"], cor["pose"], 1.0, self.models_dir)
        self.log(f"    factory_hall {len(F.tris)} tris, corridor {len(C.tris)} tris")
        self._build_factory(F)
        self._build_corridor(C)
        self._build_junction()

    def _build_factory(self, F: PlacedMesh):
        L, g = self.L, "plant_structure"
        fi = L["factory"]["interior"]
        lv = [f for f in floor_levels(F) if f["area"] > 50.0]
        floor_z = min(lv, key=lambda f: abs(f["z"] - L["factory"]["floor_z"]))["z"] if lv \
            else L["factory"]["floor_z"]
        floor_z = round(floor_z, 4)
        grid = Grid(fi["x_min"] - 1.0, fi["y_min"] - 1.0, fi["x_max"] + 1.0, fi["y_max"] + 1.0, SLAB_RES)
        occ = rasterise_slab(F.tris, grid, floor_z + 0.05, floor_z + 1.0)
        ys = (fi["y_min"] + 0.3, fi["y_max"] - 0.3)
        xs = (fi["x_min"] + 0.3, fi["x_max"] - 0.3)
        west, _, _ = wall_profile(occ, grid, "x", fi["x_min"], +1, ys)
        east, s_e, v_e = wall_profile(occ, grid, "x", fi["x_max"], -1, ys)
        south, _, _ = wall_profile(occ, grid, "y", fi["y_min"], +1, xs)
        north, _, _ = wall_profile(occ, grid, "y", fi["y_max"], -1, xs)
        for nm, v, ref in (("west", west, fi["x_min"]), ("east", east, fi["x_max"]),
                           ("south", south, fi["y_min"]), ("north", north, fi["y_max"])):
            if not np.isfinite(v):
                self.notes.append(f"factory {nm} wall face not measurable -> layout value {ref}")
        west = west if np.isfinite(west) else fi["x_min"]
        east = east if np.isfinite(east) else fi["x_max"]
        south = south if np.isfinite(south) else fi["y_min"]
        north = north if np.isfinite(north) else fi["y_max"]
        eo = L["factory"]["east_opening"]
        gaps = [gp for gp in gaps_in_profile(s_e, v_e) if gp[0] < eo["y_max"] and gp[1] > eo["y_min"]]
        if gaps:
            gp = max(gaps, key=lambda q: q[1] - q[0])
            open_y0, open_y1 = gp
        else:
            open_y0, open_y1 = eo["y_min"], eo["y_max"]
            self.notes.append("east opening not found in the mesh -> layout values")
        # opening height: lowest geometry above the opening
        c = F.tris.mean(1)
        sel = (np.abs(c[:, 0] - east) < 0.4) & (c[:, 1] > open_y0 + 0.1) & (c[:, 1] < open_y1 - 0.1) \
            & (c[:, 2] > floor_z + 1.5)
        open_top = float(F.tris[sel][:, :, 2].min()) if sel.any() else eo["height"]
        # interior walls: raster columns occupied over > 75 % of the room depth
        r0, r1 = grid.iy(south + 0.3), grid.iy(north - 0.3)
        frac = occ[r0:r1 + 1].mean(0)
        partitions = []
        cx = grid.x0 + (np.arange(grid.w) + 0.5) * grid.res
        cand = np.nonzero((frac > 0.75) & (cx > west + 1.0) & (cx < east - 1.0))[0]
        if len(cand):
            runs = np.split(cand, np.nonzero(np.diff(cand) > 2)[0] + 1)
            for run in runs:
                xw0, xw1 = cx[run[0]] - grid.res / 2, cx[run[-1]] + grid.res / 2
                xm = (xw0 + xw1) / 2
                _, s_p, v_p = wall_profile(occ, grid, "x", xm, -1, ys, search=0.15)
                doors = gaps_in_profile(s_p, v_p, min_len=0.6)
                csel = (np.abs(c[:, 0] - xm) < 0.3) & (c[:, 2] > floor_z + 1.0)
                ptop = []
                for d0, d1 in doors:
                    s2 = csel & (c[:, 1] > d0 + 0.1) & (c[:, 1] < d1 - 0.1)
                    ptop.append(float(F.tris[s2][:, :, 2].min()) if s2.any() else 2.0)
                partitions.append({"x0": float(xw0), "x1": float(xw1), "doors": doors, "door_top": ptop})
        # stairs (factory: SW staircase up to the mezzanine) - treads at intermediate heights
        # (upper bound above the mezzanine landing; the large mezzanine floor
        #  itself is rejected by max_area)
        treads = detect_treads(F, (west, east, south, north), floor_z, floor_z + 2.7)
        self.geometry["factory"] = {
            "floor_z": floor_z, "wall_faces": {"west": round(west, 4), "east": round(east, 4),
                                                "south": round(south, 4), "north": round(north, 4)},
            "east_opening": {"y_min": round(open_y0, 4), "y_max": round(open_y1, 4), "top": round(open_top, 3)},
            "interior_walls": [{"x0": round(p["x0"], 4), "x1": round(p["x1"], 4),
                                "doors": [[round(a, 4), round(b, 4)] for a, b in p["doors"]],
                                "door_top": [round(t, 3) for t in p["door_top"]]} for p in partitions],
            "stair_treads": [{k: round(v, 4) for k, v in t.items()} for t in treads],
            "interior_size": [round(east - west, 3), round(north - south, 3)],
            "mesh_bbox": [[round(float(v), 3) for v in F.tris.reshape(-1, 3).min(0)],
                          [round(float(v), 3) for v in F.tris.reshape(-1, 3).max(0)]],
        }
        self.log(f"    factory floor {floor_z:+.3f}  walls W {west:.3f} E {east:.3f} S {south:.3f} N {north:.3f}"
                 f"  east opening y [{open_y0:.2f}, {open_y1:.2f}] top {open_top:.2f}")
        for p in partitions:
            self.log(f"    interior wall x [{p['x0']:.3f}, {p['x1']:.3f}] doors {p['doors']}")
        self.log(f"    factory stair treads: {len(treads)}")
        # --- primitives
        t, H = WALL_THICKNESS, FACTORY_WALL_HEIGHT
        z0 = floor_z
        x_out = self.L["corridor"]["interior"]["x_min"]          # factory outer east face = corridor start
        self.add(box_from_bounds("factory_floor", west, east, south, north, z0 - FLOOR_THICKNESS, z0, g,
                                 nav=NAV_WALK, visual=False, mu=FLOOR_MU, kind="floor"), "plant_structure")
        walls = [
            ("factory_wall_west", west - t, west, south - t, north + t),
            ("factory_wall_south", west - t, east + t, south - t, south),
            ("factory_wall_north", west - t, east + t, north, north + t),
            ("factory_wall_east_s", east, x_out, south - t, open_y0),
            ("factory_wall_east_n", east, x_out, open_y1, north + t),
        ]
        for nm, ax0, ax1, ay0, ay1 in walls:
            self.add(box_from_bounds(nm, ax0, ax1, ay0, ay1, z0, z0 + H, g, nav=FORCED_OCCUPIED,
                                     visual=False, kind="wall"), "plant_structure")
        self.add(box_from_bounds("factory_wall_east_lintel", east, x_out, open_y0, open_y1, z0 + open_top,
                                 z0 + max(H, open_top + 0.3), g, nav=NAV_IGNORE, visual=False, kind="wall"),
                 "plant_structure")
        for k, p in enumerate(partitions):
            xa, xb = p["x0"], p["x1"]
            if xb - xa < 0.15:                      # thin mesh wall -> 15 cm collision slab
                xm = (xa + xb) / 2
                xa, xb = xm - 0.075, xm + 0.075
            edges = [south] + [v for d in p["doors"] for v in d] + [north]
            for j in range(0, len(edges), 2):
                if edges[j + 1] - edges[j] > 0.02:
                    self.add(box_from_bounds(f"factory_partition{k}_wall{j // 2}", xa, xb, edges[j], edges[j + 1],
                                             z0, z0 + H, g, nav=FORCED_OCCUPIED, visual=False, kind="wall"),
                             "plant_structure")
            for j, (d0, d1) in enumerate(p["doors"]):
                self.add(box_from_bounds(f"factory_partition{k}_lintel{j}", xa, xb, d0, d1, z0 + p["door_top"][j],
                                         z0 + H, g, nav=NAV_IGNORE, visual=False, kind="wall"), "plant_structure")
        self._add_treads(treads, z0, "factory_stair", FORCED_OCCUPIED, (south, north))

    def _add_treads(self, treads, z_floor, prefix, nav, y_clip=None, x_end=None, y_span=None):
        """One solid box per tread: from z_floor up to the tread, x from the
        tread's riser to the next riser (treads overlap by their nosing)."""
        if not treads:
            return
        # order the flight along the direction of rising z
        tr = sorted(treads, key=lambda q: q["z"])
        xs = [(q["x0"] + q["x1"]) / 2 for q in tr]
        along_x = abs(xs[-1] - xs[0]) >= abs(((tr[-1]["y0"] + tr[-1]["y1"]) - (tr[0]["y0"] + tr[0]["y1"])) / 2)
        for k, q in enumerate(tr):
            x0, x1, y0, y1 = q["x0"], q["x1"], q["y0"], q["y1"]
            if y_span is not None:
                y0, y1 = y_span
            nxt = tr[k + 1] if k + 1 < len(tr) else None
            if along_x and nxt is not None:      # cut the nosing overlap at the next riser
                if nxt["x0"] > x0:
                    x1 = min(x1, nxt["x0"])
                else:
                    x0 = max(x0, nxt["x1"])
            self.add(box_from_bounds(f"{prefix}_{k + 1:02d}", x0, x1, y0, y1, z_floor - 0.05, q["z"],
                                     "plant_structure", nav=nav,
                                     visual=False, mu=FLOOR_MU, kind="tread"), "plant_structure")

    def _build_corridor(self, C: PlacedMesh):
        L, g = self.L, "plant_structure"
        ci, cor = L["corridor"]["interior"], L["corridor"]
        lv = floor_levels(C, min_area=20.0)
        z_up = min(lv, key=lambda f: abs(f["z"] - cor["upper_floor"]["z"]))["z"]
        z_lo = min(lv, key=lambda f: abs(f["z"] - cor["lower_floor"]["z"]))["z"]
        z_up, z_lo = round(z_up, 4), round(z_lo, 4)
        ramp = fit_ramp(C)
        de = cor["descent"]
        st = cor["stairs"]
        y_ramp0 = ramp["y0"] if ramp else cor["ramp"]["y_min"]
        treads = detect_treads(C, (de["x_min"] - 0.6, de["x_max"] + 0.6, ci["y_min"] - 0.1, y_ramp0 + 0.05),
                               z_lo, z_up, exclude_mat="uphill", min_side=0.15)
        grid = Grid(ci["x_min"] - 0.5, ci["y_min"] - 0.8, ci["x_max"] + 0.8, ci["y_max"] + 0.8, SLAB_RES)
        occ_up = rasterise_slab(C.tris, grid, z_up + 0.05, z_up + 1.0)
        occ_lo = rasterise_slab(C.tris, grid, z_lo + 0.05, z_lo + 1.0)
        up_x = (ci["x_min"] + 1.0, de["x_min"] - 1.0)
        lo_x = (de["x_max"] + 1.0, ci["x_max"] - 1.0)
        n_up, _, _ = wall_profile(occ_up, grid, "y", ci["y_max"], -1, up_x, search=0.4)
        s_up, _, _ = wall_profile(occ_up, grid, "y", ci["y_min"], +1, up_x, search=0.4)
        n_lo, _, _ = wall_profile(occ_lo, grid, "y", ci["y_max"], -1, lo_x, search=0.4)
        s_lo, _, _ = wall_profile(occ_lo, grid, "y", ci["y_min"], +1, lo_x, search=0.4)
        end_x, _, _ = wall_profile(occ_lo, grid, "x", ci["x_max"], -1, (ci["y_min"] + 0.3, ci["y_max"] - 0.3),
                                   search=0.5)
        vals = {"north_upper": n_up, "south_upper": s_up, "north_lower": n_lo, "south_lower": s_lo,
                "end": end_x}
        refs = {"north_upper": ci["y_max"], "south_upper": ci["y_min"], "north_lower": ci["y_max"],
                "south_lower": ci["y_min"], "end": ci["x_max"]}
        for k in vals:
            if not np.isfinite(vals[k]):
                self.notes.append(f"corridor {k} wall face not measurable -> layout value")
                vals[k] = refs[k]
        x_start = ci["x_min"]
        # the upper floor ends where the ramp / first tread starts
        x_desc0 = ramp["x0"] if ramp else de["x_min"]
        x_desc1 = ramp["x1"] if ramp else de["x_max"]
        stair_y = (min(t["y0"] for t in treads), max(t["y1"] for t in treads)) if treads else (st["y_min"], st["y_max"])
        self.geometry["corridor"] = {
            "upper_floor_z": z_up, "lower_floor_z": z_lo,
            "wall_faces": {k: round(v, 4) for k, v in vals.items()},
            "descent": {"x_start": round(x_desc0, 4), "x_end": round(x_desc1, 4)},
            "ramp": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in (ramp or {}).items()},
            "stairs": {"y_min": round(stair_y[0], 4), "y_max": round(stair_y[1], 4),
                       "treads": [{k: round(v, 4) for k, v in t.items()} for t in sorted(treads, key=lambda q: -q["z"])]},
            "interior_size": [round(vals["end"] - x_start, 3), round(min(n_up - s_up, n_lo - s_lo), 3)],
        }
        self.log(f"    corridor floors upper {z_up:+.3f} lower {z_lo:+.3f}; walls N {n_up:.3f}/{n_lo:.3f} "
                 f"S {s_up:.3f}/{s_lo:.3f} end x {vals['end']:.3f}")
        if ramp:
            self.log(f"    ramp z = {ramp['a']:.5f} x + {ramp['b']:.4f} ({ramp['slope_deg']:.2f} deg) "
                     f"x [{ramp['x0']:.3f}, {ramp['x1']:.3f}] y [{ramp['y0']:.3f}, {ramp['y1']:.3f}] "
                     f"resid {ramp['residual_max'] * 1000:.1f} mm")
        self.log(f"    stairs y [{stair_y[0]:.3f}, {stair_y[1]:.3f}] treads " +
                 ", ".join(f"z {t['z']:+.3f} x [{t['x0']:.2f},{t['x1']:.2f}]" for t in sorted(treads, key=lambda q: -q['z'])))
        t = WALL_THICKNESS
        ceiling = z_up + L["corridor"]["height"]
        nu, su, nl, sl, ex = vals["north_upper"], vals["south_upper"], vals["north_lower"], vals["south_lower"], vals["end"]
        # floors
        self.add(box_from_bounds("corridor_floor_upper", x_start, x_desc0, su, nu, z_up - FLOOR_THICKNESS, z_up, g,
                                 nav=NAV_WALK, visual=False, mu=FLOOR_MU, kind="floor"), "plant_structure")
        self.add(box_from_bounds("corridor_floor_lower", x_desc0, ex, sl, nl, z_lo - FLOOR_THICKNESS, z_lo, g,
                                 nav=NAV_WALK, visual=False, mu=FLOOR_MU, kind="floor"), "plant_structure")
        # landing between the end of the upper floor and the first riser (stair side)
        if treads:
            first = max(treads, key=lambda q: q["z"])
            if first["x0"] - x_desc0 > 0.02:
                self.add(box_from_bounds("corridor_stair_landing", x_desc0, first["x0"], stair_y[0], stair_y[1],
                                         z_lo - 0.05, z_up, g, nav=NAV_WALK, visual=False, mu=FLOOR_MU,
                                         kind="landing"), "plant_structure")
            # treads (walkable), spanning the measured stair width
            tr = sorted(treads, key=lambda q: -q["z"])           # top tread first (west -> east)
            for k, q in enumerate(tr):
                x0 = q["x0"]
                x1 = tr[k + 1]["x0"] if k + 1 < len(tr) else q["x1"]
                self.add(box_from_bounds(f"corridor_stair_{k + 1:02d}", x0, max(x1, x0 + 0.02), stair_y[0],
                                         stair_y[1], z_lo - 0.05, q["z"], g, nav=NAV_WALK, visual=False,
                                         mu=FLOOR_MU, kind="tread"), "plant_structure")
        # ramp: rotated box whose top face lies on the fitted plane
        if ramp:
            a, b = ramp["a"], ramp["b"]
            xa, xb = ramp["x0"], ramp["x1"]
            za, zb = a * xa + b, a * xb + b
            Lr = math.hypot(xb - xa, zb - za)
            pitch = math.atan2(za - zb, xb - xa)                 # descending towards +x -> positive pitch
            n = np.array([math.sin(pitch), 0.0, math.cos(pitch)])
            top_c = np.array([(xa + xb) / 2, (ramp["y0"] + ramp["y1"]) / 2, (za + zb) / 2])
            cen = top_c - n * RAMP_THICKNESS / 2
            self.add(box("corridor_ramp", cen, (Lr, ramp["y1"] - ramp["y0"], RAMP_THICKNESS), g, pitch=pitch,
                         nav=NAV_WALK, visual=False, mu=FLOOR_MU, kind="ramp"), "plant_structure")
            # the ramp mesh is a perforated grating: a solid visual deck just
            # below it keeps the depth camera from seeing the floor 0.9 m down
            deck = cen - n * RAMP_DECK_DROP
            self.add(box("corridor_ramp_deck", deck, (Lr, ramp["y1"] - ramp["y0"] - 0.01, RAMP_THICKNESS), g,
                         pitch=pitch, nav=NAV_IGNORE, collision=False, material="dock_board", kind="ramp"),
                     "plant_structure")
        # walls (upper level up to the ceiling; lower level from the lower floor)
        x_mid = x_desc0
        for nm, x0, x1, y0, y1, zb_, zt in [
            ("corridor_wall_north_upper", x_start, x_mid, nu, nu + t, z_up, ceiling),
            ("corridor_wall_south_upper", x_start, x_mid, su - t, su, z_up, ceiling),
            ("corridor_wall_north_lower", x_mid, ex + t, nl, nl + t, z_lo, ceiling),
            ("corridor_wall_south_lower", x_mid, ex + t, sl - t, sl, z_lo, ceiling),
            ("corridor_wall_end", ex, ex + t, sl - t, nl + t, z_lo, ceiling),
        ]:
            self.add(box_from_bounds(nm, x0, x1, y0, y1, zb_, zt, g, nav=FORCED_OCCUPIED, visual=False,
                                     kind="wall"), "plant_structure")

    def _build_junction(self):
        """Close the factory east opening around the corridor (the opening is
        wider than the corridor: visible holes into the void) and bridge the
        floor gap through the wall thickness."""
        fg, cg = self.geometry["factory"], self.geometry["corridor"]
        east = fg["wall_faces"]["east"]
        oy0, oy1, otop = fg["east_opening"]["y_min"], fg["east_opening"]["y_max"], fg["east_opening"]["top"]
        x_c = self.L["corridor"]["interior"]["x_min"]
        nu, su = cg["wall_faces"]["north_upper"], cg["wall_faces"]["south_upper"]
        z0 = fg["floor_z"]
        g = "plant_structure"
        self.add(box_from_bounds("junction_floor", east - 0.05, x_c + 0.05, max(oy0, su), min(oy1, nu),
                                 z0 - FLOOR_THICKNESS, z0, g, nav=NAV_WALK, visual=False, mu=FLOOR_MU,
                                 kind="floor"), "plant_structure")
        # The two meshes leave a 12 cm visual gap in the doorway floor (and the
        # corridor door frame has a 4 cm deep sill): a steel threshold plate,
        # 4 mm proud of the floors (no z-fighting), closes it for the cameras.
        self.add(box_from_bounds("junction_threshold", east - 0.04, x_c + 0.10, max(oy0, su), min(oy1, nu),
                                 z0 + THRESHOLD_H - 0.02, z0 + THRESHOLD_H, g, nav=NAV_WALK, material="steel_plate",
                                 mu=FLOOR_MU, kind="plate"), "plant_structure")
        x1 = x_c + 0.06
        if oy1 > nu + 0.01:
            self.add(box_from_bounds("junction_filler_north", east - 0.02, x1, nu, oy1 + 0.02, z0, z0 + otop + 0.02,
                                     g, nav=FORCED_OCCUPIED, material="filler_wall", kind="wall"), "plant_structure")
        if oy0 < su - 0.01:
            self.add(box_from_bounds("junction_filler_south", east - 0.02, x1, oy0 - 0.02, su, z0, z0 + otop + 0.02,
                                     g, nav=FORCED_OCCUPIED, material="filler_wall", kind="wall"), "plant_structure")
        self.geometry["junction"] = {"filler_north_y": [round(nu, 4), round(oy1, 4)],
                                     "filler_south_y": [round(oy0, 4), round(su, 4)]}

    # ---------------------------------------------------------- floor raster
    def floor_raster(self, kinds=("floor", "tread", "ramp", "landing", "platform", "plate", "pad")):
        """(height map, walkable mask) on the working grid from the walkable prims."""
        g = self.grid
        hm = np.full(g.shape, np.nan)
        X, Y = g.centres()
        for p in self.walkable():
            if p.kind not in kinds:
                continue
            m = np.zeros(g.shape, np.uint8)
            g.fill(m, p.footprint(), 1, outline=False)
            if not m.any():
                continue
            sel = m.astype(bool)
            z = p.top_z_at(X[sel], Y[sel])
            cur = hm[sel]
            hm[sel] = np.where(np.isnan(cur) | (z > cur), z, cur)
        return hm, np.isfinite(hm)

    # ------------------------------------------------------------- voxel test
    def mesh_obstacles(self):
        """Robot-height obstacle masks of the building meshes and per machine."""
        t0 = time.time()
        g = self.grid
        hm, walk = self.floor_raster(kinds=("floor", "tread", "ramp", "landing"))
        self._arch_floor = hm
        vox = None
        for nm in ("factory_hall", "corridor"):
            vox = voxelise(self.meshes[nm].tris, g, vox=vox)
        self._vox_building = vox
        obst_b = band_occupied(vox, hm, CLEAR_LO_BUILDING, CLEAR_HI)
        self.log(f"    building voxels [{time.time() - t0:.1f} s], blocked cells {int(obst_b.sum())}")
        self._obst_building = obst_b
        return obst_b

    def residual_features(self):
        """Building geometry at robot height not covered by explicit
        primitives -> extra collision boxes (columns, door jambs, wall pipes)."""
        import cv2
        g = self.grid
        cover = np.zeros(g.shape, np.uint8)
        for p in self.prims:
            if p.collision and p.nav == FORCED_OCCUPIED:
                g.fill(cover, p.footprint(), 1)
        cover = cv2.dilate(cover, np.ones((5, 5), np.uint8)).astype(bool)
        hm = self._arch_floor
        resid = self._obst_building & ~cover & np.isfinite(hm)
        # bridge the 1-cell gaps between the flanges of an I-beam etc.
        merged = cv2.morphologyEx(resid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        count = {"feature": 0, "block": 0}
        recs = []
        for i in range(1, n):
            comp = lab == i
            if comp.sum() < 2:
                continue
            x, y, w, h = stats[i, :4]
            comp = fill_holes(comp)
            sub = comp[y:y + h, x:x + w]
            bw, bh = w * g.res, h * g.res
            if bw <= 0.8 and bh <= 0.8:
                rects = [(0, h - 1, 0, w - 1)]               # compact detail -> its bounding box
            else:
                rects = decompose_rectangles(sub, min_cells=2, max_rects=8, min_coverage=0.95)
            fill_ratio = float(sub.mean())
            kind = "block" if bw * bh > 2.0 and fill_ratio > 0.85 else "feature"
            for (r0, r1, c0, c1) in rects:
                m = np.zeros(g.shape, bool)
                m[y + r0:y + r1 + 1, x + c0:x + c1 + 1] = True
                zf = float(np.nanmin(hm[m])) if np.any(np.isfinite(hm[m])) else 0.0
                top = voxel_top(self._vox_building, m)
                top = min(top if np.isfinite(top) else zf + 1.0, zf + 2.0)
                x0 = g.x0 + (x + c0) * g.res
                x1 = g.x0 + (x + c1 + 1) * g.res
                y0 = g.y0 + (y + r0) * g.res
                y1 = g.y0 + (y + r1 + 1) * g.res
                bname = "factory" if x0 < self.L["corridor"]["interior"]["x_min"] - 0.2 else "corridor"
                count[kind] += 1
                nm = f"{bname}_{kind}_{count[kind]:02d}"
                self.add(box_from_bounds(nm, x0, x1, y0, y1, zf, top, "plant_structure", nav=FORCED_OCCUPIED,
                                         visual=False, kind="wall" if kind == "block" else "feature"),
                         "plant_structure")
                recs.append({"name": nm, "bounds": [round(x0, 3), round(x1, 3), round(y0, 3), round(y1, 3),
                                                    round(zf, 3), round(top, 3)]})
        self.geometry["residual_features"] = recs
        self.log(f"    residual building geometry -> {count['block']} blocks + {count['feature']} detail boxes")

    # ------------------------------------------------------------------ machines
    def build_machines(self):
        import cv2
        hm = self._arch_floor
        g = self.grid
        self._obst_machine = np.zeros(g.shape, bool)
        for m in self.L.get("machines", []):
            name = m["model"]
            pose = m["pose"]
            scale = float(m.get("scale", 1.0))
            pm = load_placed_mesh(name, pose, scale, self.models_dir)
            self.meshes[m["id"]] = pm
            lo, hi = pm.tris.reshape(-1, 3).min(0), pm.tris.reshape(-1, 3).max(0)
            pad = 0.3
            sub = Grid(math.floor((lo[0] - pad) / g.res) * g.res, math.floor((lo[1] - pad) / g.res) * g.res,
                       math.ceil((hi[0] + pad) / g.res) * g.res, math.ceil((hi[1] + pad) / g.res) * g.res, g.res)
            vox = voxelise(pm.tris, sub)
            # floor under the machine (sampled from the global raster)
            ox, oy = int(round((sub.x0 - g.x0) / g.res)), int(round((sub.y0 - g.y0) / g.res))
            fl = np.full(sub.shape, np.nan)
            ys0, xs0 = max(0, oy), max(0, ox)
            ys1, xs1 = min(g.h, oy + sub.h), min(g.w, ox + sub.w)
            fl[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = hm[ys0:ys1, xs0:xs1]
            fl = np.where(np.isfinite(fl), fl, float(self.L.floor_z(pose[0], pose[1])))
            # collision: everything the robot body can touch (floor+0.03 .. +0.70)
            fp = band_occupied(vox, fl, CLEAR_LO_MACHINE, CLEAR_HI)
            fp_solid = fill_holes(cv2.morphologyEx(fp.astype(np.uint8), cv2.MORPH_CLOSE,
                                                   np.ones((3, 3), np.uint8)).astype(bool))
            # nav map: safety envelope - no planning under machine parts lower
            # than MACHINE_NAV_HI; gaps narrower than the robot and enclosed
            # pockets (e.g. the courtyard of the gas skid piping) are closed
            fp_hi = band_occupied(vox, fl, CLEAR_LO_MACHINE, MACHINE_NAV_HI) | fp_solid
            kd = int(round((2 * ROBOT_RADIUS + 0.1) / g.res)) | 1
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
            pad = kd
            big = cv2.copyMakeBorder(fp_hi.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            fp_nav = fill_holes(cv2.morphologyEx(big, cv2.MORPH_CLOSE, ker).astype(bool))[pad:-pad, pad:-pad]
            fp_nav |= fp_hi
            self._obst_machine[ys0:ys1, xs0:xs1] |= fp_nav[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
            rects = decompose_rectangles(fp_solid, min_cells=6, max_rects=40, min_coverage=0.97)
            mid = m["id"].lower().replace("-", "_")
            model = f"machine_{mid}"
            yaw = float(pose[3])
            Rm = rot_rpy(0, 0, yaw)
            self.models[model] = {"pose": (pose[0], pose[1], pose[2], 0.0, 0.0, yaw), "static": True,
                                  "prims": [], "extra": self._machine_visual_xml(name, scale, m)}
            n_box = 0
            for (r0, r1, c0, c1) in rects:
                mm = np.zeros(sub.shape, bool)
                mm[r0:r1 + 1, c0:c1 + 1] = True
                zf = float(np.nanmin(fl[mm]))
                top = voxel_top(vox, mm)
                if not np.isfinite(top):
                    continue
                top = min(top, float(hi[2]))
                x0, x1 = sub.x0 + c0 * sub.res, sub.x0 + (c1 + 1) * sub.res
                y0, y1 = sub.y0 + r0 * sub.res, sub.y0 + (r1 + 1) * sub.res
                cw = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (zf + top) / 2])
                cl = Rm.T @ (cw - np.array(pose[:3]))
                # world axis-aligned box expressed in the machine frame (yaw -yaw)
                n_box += 1
                p = box(f"{mid}_box_{n_box:02d}", cl, (x1 - x0, y1 - y0, top - zf), model, yaw=-yaw,
                        nav=FORCED_OCCUPIED, visual=False, kind="machine")
                self.add(p, model, self.models[model]["pose"])
            area = float(fp_solid.sum() * g.res ** 2)
            area_nav = float(fp_nav.sum() * g.res ** 2)
            fpl = m.get("footprint")
            self.machine_records.append({
                "id": m["id"], "model": name, "scale": scale, "pose": [float(v) for v in pose],
                "mesh_bbox": [[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]],
                "dense_footprint_area_m2": round(area, 2),
                "nav_footprint_area_m2": round(area_nav, 2),
                "layout_footprint_area_m2": round((fpl["x_max"] - fpl["x_min"]) * (fpl["y_max"] - fpl["y_min"]), 2)
                if fpl else None,
                "collision_boxes": n_box})
            self.log(f"    {m['id']}: mesh bbox x [{lo[0]:.2f},{hi[0]:.2f}] y [{lo[1]:.2f},{hi[1]:.2f}] z top "
                     f"{hi[2]:.2f}; dense low footprint {area:.1f} m2 -> {n_box} boxes; nav footprint "
                     f"{area_nav:.1f} m2")
        self.geometry["machines"] = self.machine_records

    def _machine_visual_xml(self, name: str, scale: float, m: dict) -> str:
        th = thermal_plugin_xml(MACHINE_TEMP_K, "          ") if MACHINE_TEMP_K else ""
        return f"""        <visual name="mesh_visual">
          <geometry>
            <mesh>
              <uri>model://{name}/meshes/{name}.obj</uri>
              <scale>{fmt(scale)} {fmt(scale)} {fmt(scale)}</scale>
            </mesh>
          </geometry>
{th}        </visual>
"""

    # ----------------------------------------------------------------- obstacles
    def build_obstacles(self):
        for ob in self.L.obstacles:
            t = ob["type"]
            oid = ob["id"]
            mid = "obstacle_" + oid.lower().replace("-", "_")
            if t == "pipe":
                cx, cy = ob["center"]
                d, Lp, yaw = float(ob["diameter"]), float(ob["length"]), float(ob.get("yaw", 0.0))
                z0 = self.floor_at(cx, cy, ob.get("floor_z"))
                self.add(cylinder(f"{mid}_pipe", (cx, cy, z0 + d / 2), d / 2, Lp, mid, pitch=math.pi / 2, yaw=yaw,
                                  nav=NAV_FREE, material="steel_pipe", kind="pipe"), mid)
            elif t == "pallet":
                cx, cy = ob["center"]
                yaw = float(ob.get("yaw", 0.0))
                z0 = self.floor_at(cx, cy, ob.get("floor_z"))
                lo, hi = self._model_bbox("pallet")
                hgt = hi[2] - lo[2]
                for k in range(int(ob.get("stack", 1))):
                    self.includes.append({"uri": "model://pallet", "name": f"{mid}_{k}",
                                          "pose": (cx, cy, z0 + k * hgt, 0.0, 0.0, yaw)})
                # footprint-only primitive for the nav map / figures (collision is in the model)
                p = box(f"{mid}_footprint", (cx, cy, z0 + hgt * ob.get("stack", 1) / 2),
                        (hi[0] - lo[0], hi[1] - lo[1], hgt * ob.get("stack", 1)), mid, yaw=yaw,
                        nav=FORCED_OCCUPIED, collision=False, visual=False, kind="pallet")
                self.prims.append(p)
            elif t == "rubble":
                self._rubble(ob, mid)
            elif t == "plate":
                x0, y0, x1, y1 = ob["rect"]
                th = float(ob.get("thickness", 0.01))
                z0 = self.floor_at((x0 + x1) / 2, (y0 + y1) / 2, ob.get("floor_z"))
                self.add(box_from_bounds(f"{mid}_plate", x0, x1, y0, y1, z0, z0 + th, mid, nav=NAV_WALK,
                                         material="steel_plate", mu=float(ob.get("mu", 0.2)), kind="plate"), mid)
            elif t == "drum":
                cx, cy = ob["center"]
                r, hgt = float(ob["radius"]), float(ob["height"])
                z0 = self.floor_at(cx, cy, ob.get("floor_z"))
                digits = "".join(ch for ch in oid if ch.isdigit())
                mat = "drum_blue" if (int(digits) if digits else len(oid)) % 2 == 0 else "drum_red"
                self.add(cylinder(f"{mid}_drum", (cx, cy, z0 + hgt / 2), r, hgt, mid, nav=FORCED_OCCUPIED,
                                  material=mat, kind="drum"), mid)
                for k, zz in enumerate((0.3, 0.62)):    # rolling hoops
                    self.add(cylinder(f"{mid}_hoop{k}", (cx, cy, z0 + zz * hgt / 0.9), r + 0.008, 0.02, mid,
                                      nav=NAV_IGNORE, collision=False, material="steel_pipe", kind="drum"), mid)
            elif t == "platform":
                x0, y0, x1, y1 = ob["rect"]
                hgt = float(ob.get("height", 0.1))
                z0 = self.floor_at((x0 + x1) / 2, (y0 + y1) / 2, ob.get("floor_z"))
                self.add(box_from_bounds(f"{mid}_platform", x0, x1, y0, y1, z0, z0 + hgt, mid, nav=NAV_WALK,
                                         material="platform", mu=FLOOR_MU, kind="platform"), mid)
            else:
                self.notes.append(f"unknown obstacle type {t} ({oid}) skipped")

    def _rubble(self, ob, mid):
        """Seeded random concrete chunks: boxes (visual == collision)."""
        x0, y0, x1, y1 = ob["rect"]
        rng = np.random.RandomState(int(ob.get("seed", 0)))
        hmin, hmax = float(ob.get("h_min", 0.02)), float(ob.get("h_max", 0.06))
        z0 = self.floor_at((x0 + x1) / 2, (y0 + y1) / 2, ob.get("floor_z"))
        for k in range(int(ob.get("count", 20))):
            sx, sy = rng.uniform(0.06, 0.28, 2)
            h = rng.uniform(hmin, hmax)
            yaw = rng.uniform(-math.pi, math.pi)
            r = 0.5 * math.hypot(sx, sy)
            cx = rng.uniform(x0 + r, x1 - r) if x1 - x0 > 2 * r else (x0 + x1) / 2
            cy = rng.uniform(y0 + r, y1 - r) if y1 - y0 > 2 * r else (y0 + y1) / 2
            roll, pitch = rng.uniform(-0.12, 0.12, 2)
            # lift so the lowest corner touches the floor despite the tilt
            Rk = rot_rpy(roll, pitch, yaw)
            corners = np.array([[a * sx / 2, b * sy / 2, d * h / 2] for a in (-1, 1) for b in (-1, 1)
                                for d in (-1, 1)]) @ Rk.T
            cz = z0 - corners[:, 2].min() - 0.005
            shade = RUBBLE_SHADES[k % len(RUBBLE_SHADES)]
            self.add(box(f"{mid}_chunk{k:02d}", (cx, cy, cz), (sx, sy, h), mid, roll=roll, pitch=pitch, yaw=yaw,
                         nav=NAV_FREE, colour=shade, kind="rubble"), mid)

    def _model_bbox(self, name):
        pm = read_obj(os.path.join(self.models_dir, name, "meshes", name + ".obj"))
        return pm.v.min(0), pm.v.max(0)

    # ------------------------------------------------------- pipe rack, hot spots
    def build_pipe_rack(self):
        L = self.L
        th = L.thermal
        pr = L["pipe_rack"]
        mid = "pipe_rack"
        p0, p1 = np.asarray(pr["pipe"]["start"], float), np.asarray(pr["pipe"]["end"], float)
        r = float(pr["pipe"]["radius"])
        axis = p1 - p0
        roll, pitch, yaw = axis_rpy(axis)
        self.add(cylinder("rack_pipe", (p0 + p1) / 2, r, float(np.linalg.norm(axis)), mid, roll, pitch, yaw,
                          nav=NAV_AUTO, material="rack_pipe", temperature=th.get("pipe_body_k"), kind="rack_pipe"),
                 mid)
        sx, sy, sz = pr.get("support_size", [0.12, 0.5, 1.5])
        y_c = float(p0[1])
        for k, xs in enumerate(pr["supports_x"]):
            z0 = self.floor_at(xs, y_c)
            self.add(box(f"rack_support_{k}", (xs, y_c, z0 + sz / 2), (sx, sy, sz), mid, nav=FORCED_OCCUPIED,
                         material="rack_support", kind="rack_support"), mid)
            # pipe saddle on the support
            self.add(box(f"rack_saddle_{k}", (xs, y_c, float(p0[2]) - r - 0.03), (sx + 0.04, sy * 0.7, 0.06), mid,
                         nav=NAV_IGNORE, material="rack_support", kind="rack_support"), mid)
        for h in L.hotspots:
            pos, ax = np.asarray(h["position"], float), np.asarray(h.get("axis", [1, 0, 0]), float)
            roll, pitch, yaw = axis_rpy(ax)
            on_rack = (h["kind"] == "flange" and abs(pos[1] - y_c) < 0.05 and abs(pos[2] - p0[2]) < 0.05
                       and min(p0[0], p1[0]) - 0.1 <= pos[0] <= max(p0[0], p1[0]) + 0.1)
            if on_rack:
                model = mid
            else:
                model = "hotspot_" + h["id"].lower()
            mat = "bearing" if h["kind"] == "bearing" else "flange"
            nav = NAV_AUTO
            self.add(cylinder(f"hotspot_{h['id'].lower()}_{h['kind']}", pos, float(h["radius"]), float(h["length"]),
                              model, roll, pitch, yaw, nav=nav, material=mat, temperature=float(h["temp_k"]),
                              kind="hotspot"), model)
            if h["kind"] == "flange":                 # bolt ring look: second thinner disc
                self.add(cylinder(f"hotspot_{h['id'].lower()}_gasket", pos, float(h["radius"]) * 0.82,
                                  float(h["length"]) * 1.25, model, roll, pitch, yaw, nav=NAV_IGNORE,
                                  collision=False, material="flange", temperature=float(h["temp_k"]),
                                  kind="hotspot"), model)
            if h["kind"] == "bearing":                # pedestal under the bearing housing
                zf = self.floor_at(pos[0], pos[1])
                hgt = pos[2] - float(h["radius"]) - zf
                if hgt > 0.05:
                    self.add(box(f"hotspot_{h['id'].lower()}_pedestal", (pos[0], pos[1], zf + hgt / 2),
                                 (float(h["length"]) * 0.8, float(h["radius"]) * 1.6, hgt), model,
                                 nav=FORCED_OCCUPIED, material="bearing", temperature=0.5 * (float(h["temp_k"])
                                                                                            + th.get("ambient_k", 298.15)),
                                 kind="hotspot"), model)
        st = L.get("corridor_pipe_stub")
        if st:
            x, y = st["position"]
            zf = self.floor_at(x, y)
            hgt = float(st["height"])
            self.add(cylinder("corridor_pipe_stub", (x, y, zf + hgt / 2), float(st["radius"]), hgt,
                              "corridor_pipe_stub", nav=FORCED_OCCUPIED, material="steel_pipe",
                              temperature=th.get("pipe_body_k"), kind="rack_support"), "corridor_pipe_stub")
            # wall bracket at the top
            wall = self.geometry["corridor"]["wall_faces"]["south_upper" if y < 0 else "north_upper"]
            blen = abs(wall - y)
            self.add(box("corridor_pipe_stub_bracket", (x, (y + wall) / 2, zf + hgt - 0.1), (0.05, blen, 0.04),
                         "corridor_pipe_stub", nav=NAV_IGNORE, collision=False, material="rack_support",
                         kind="rack_support"), "corridor_pipe_stub")

    # ------------------------------------------------------------------- gauges
    def build_gauges(self, write_models: bool = True):
        import cv2
        occupied_now = [p for p in self.prims if p.nav == FORCED_OCCUPIED and p.collision]
        for gdef in self.L.gauges:
            gid = gdef["id"]
            name = f"gauge_{gid.lower()}"
            pos = np.asarray(gdef["position"], float)
            n = np.asarray(gdef["normal"], float)
            n = n / np.linalg.norm(n)
            d = float(gdef.get("dial_diameter", 0.30))
            yaw = math.atan2(n[1], n[0])
            pitch = -math.asin(max(-1.0, min(1.0, n[2])))
            R = rot_rpy(0.0, pitch, yaw)
            # stand geometry (model frame: face normal +X, up +Z, origin = dial centre)
            house_r, house_d = d / 2 + 0.012, 0.040
            plate_t, plate_s = 0.012, d + 0.10
            x_plate = -(house_d + 0.0015) - plate_t / 2
            post_r = 0.03
            x_post = x_plate - plate_t / 2 - post_r
            post_world = R @ np.array([x_post, 0.0, 0.0]) + pos
            zf = self.floor_at(post_world[0], post_world[1])
            # mounted on an existing structure (e.g. G06 on a rack support) -> no post
            mounted = None
            for p in occupied_now:
                fp = p.footprint().astype(np.float32).reshape(-1, 1, 2)
                if cv2.pointPolygonTest(fp, (float(post_world[0]), float(post_world[1])), True) > -0.01:
                    zr = p.z_range()
                    if zr[0] <= pos[2] <= zr[1] + 0.05:
                        mounted = p.name
                        break
            z_post_bot = zf - pos[2]              # local z of the floor below (horizontal normal)
            z_post_top = d / 2 + 0.06
            frame = {"id": gid, "model": name, "position": pos.tolist(), "normal": n.tolist(),
                     "yaw": yaw, "pitch": pitch, "dial_diameter": d, "floor_z": zf, "mounted_on": mounted}
            self.gauge_frames.append(frame)
            mp = (pos[0], pos[1], pos[2], 0.0, pitch, yaw)
            # nav / figure primitives (world frame) mirror the model content
            stand = [box(f"{name}_plate", (x_plate, 0, 0), (plate_t, plate_s, plate_s), name,
                         nav=FORCED_OCCUPIED, kind="gauge"),
                     cylinder(f"{name}_housing", (-house_d / 2 - 0.0015, 0, 0), house_r, house_d, name,
                              pitch=math.pi / 2, nav=FORCED_OCCUPIED, kind="gauge")]
            if mounted is None:
                stand.append(cylinder(f"{name}_post", (x_post, 0, (z_post_bot + z_post_top) / 2), post_r,
                                      z_post_top - z_post_bot, name, nav=FORCED_OCCUPIED, kind="gauge"))
            for p in stand:
                Rw = R @ rot_rpy(*p.pose[3:6])
                c = R @ np.asarray(p.pose[:3]) + pos
                p.world_pose = (c[0], c[1], c[2], math.atan2(Rw[2, 1], Rw[2, 2]),
                                -math.asin(max(-1, min(1, Rw[2, 0]))), math.atan2(Rw[1, 0], Rw[0, 0]))
                p.collision = False                # collision lives in the gauge model
                p.visual = False
                self.prims.append(p)
            self.includes.append({"uri": f"model://{name}", "name": f"gauge_{gid}", "pose": mp})
            if write_models:
                self._write_gauge_model(name, gdef, d, house_r, house_d, x_plate, plate_t, plate_s,
                                        None if mounted else (x_post, post_r, z_post_bot, z_post_top))

    def _write_gauge_model(self, name, gdef, d, house_r, house_d, x_plate, plate_t, plate_s, post):
        import cv2
        mdir = os.path.join(self.models_dir, name)
        mesh_dir = os.path.join(mdir, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        tex_name = f"{name}_dial.png"
        img = gauge_design.render_gauge_face(gdef["value"], gdef["min"], gdef["max"], gdef.get("unit", ""),
                                             gdef.get("tag", ""), size=1024)
        cv2.imwrite(os.path.join(mesh_dir, tex_name), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        write_dial_obj(os.path.join(mesh_dir, "dial.obj"), d / 2.0, tex_name, f"{name}_dial")
        post_xml = ""
        if post is not None:
            x_post, post_r, zb, zt = post
            post_xml = f"""      <visual name="post_visual">
        <pose>{fmt_vec((x_post, 0, (zb + zt) / 2))} 0 0 0</pose>
        <geometry><cylinder><radius>{fmt(post_r)}</radius><length>{fmt(zt - zb)}</length></cylinder></geometry>
{material_xml("gauge_stand", indent="        ")}      </visual>
      <collision name="post_collision">
        <pose>{fmt_vec((x_post, 0, (zb + zt) / 2))} 0 0 0</pose>
        <geometry><cylinder><radius>{fmt(post_r)}</radius><length>{fmt(zt - zb)}</length></cylinder></geometry>
      </collision>
      <visual name="foot_visual">
        <pose>{fmt_vec((x_post, 0, zb + 0.01))} 0 0 0</pose>
        <geometry><cylinder><radius>{fmt(post_r * 3.5)}</radius><length>0.02</length></cylinder></geometry>
{material_xml("gauge_stand", indent="        ")}      </visual>
"""
        desc = (f"Analog pressure gauge {gdef['id']} ({gdef.get('tag', '')}), {gdef['min']}-{gdef['max']} "
                f"{gdef.get('unit', '')}, showing {gdef['value']}. Origin = dial centre, face normal +X, "
                f"texture up +Z. Generated by tools/generate_world.py.")
        sdf = f"""<?xml version="1.0"?>
<sdf version="{SDF_VERSION}">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="dial">
        <geometry>
          <mesh>
            <uri>model://{name}/meshes/dial.obj</uri>
          </mesh>
        </geometry>
      </visual>
      <visual name="housing">
        <pose>{fmt_vec((-house_d / 2 - 0.0015, 0, 0))} 0 1.570796 0</pose>
        <geometry><cylinder><radius>{fmt(house_r)}</radius><length>{fmt(house_d)}</length></cylinder></geometry>
{material_xml("gauge_housing", indent="        ")}      </visual>
      <visual name="plate_visual">
        <pose>{fmt_vec((x_plate, 0, 0))} 0 0 0</pose>
        <geometry><box><size>{fmt_vec((plate_t, plate_s, plate_s))}</size></box></geometry>
{material_xml("gauge_plate", indent="        ")}      </visual>
      <collision name="plate_collision">
        <pose>{fmt_vec(((x_plate - plate_t / 2) / 2, 0, 0))} 0 0 0</pose>
        <geometry><box><size>{fmt_vec((plate_t / 2 - x_plate, plate_s, plate_s))}</size></box></geometry>
      </collision>
{post_xml}    </link>
  </model>
</sdf>
"""
        with open(os.path.join(mdir, "model.sdf"), "w", encoding="utf-8", newline="\n") as f:
            f.write(sdf)
        with open(os.path.join(mdir, "model.config"), "w", encoding="utf-8", newline="\n") as f:
            f.write(model_config_xml(name, desc))

    # ----------------------------------------------------------------- gas leak
    def build_gas_leak(self, write_models: bool = True):
        gl = self.L.gas_leak
        vp = np.asarray(gl["valve_prop"], float)
        src = np.asarray(gl["source"], float)
        zf = self.floor_at(vp[0], vp[1])
        wind = gl.get("wind_base", [1.0, 0.0])
        wyaw = math.atan2(wind[1], wind[0])
        name = "gas_leak_valve"
        # model frame = valve prop on the floor; outlet pipe ends at the leak source
        off = src - np.array([vp[0], vp[1], zf])
        stub_h = max(0.30, off[2] + 0.15)
        prims = [
            cylinder("base_flange", (0, 0, 0.015), 0.09, 0.03, name, material="flange"),
            cylinder("stub", (0, 0, stub_h / 2), 0.05, stub_h, name, material="steel_pipe"),
            box("valve_body", (0, 0, stub_h + 0.05), (0.13, 0.11, 0.10), name, material="valve_body"),
            cylinder("stem", (0, 0, stub_h + 0.13), 0.012, 0.08, name, material="steel_pipe", collision=False),
            cylinder("hand_wheel", (0, 0, stub_h + 0.17), 0.11, 0.02, name, material="valve_red", collision=False),
            box("wheel_spoke_a", (0, 0, stub_h + 0.17), (0.22, 0.015, 0.012), name, material="valve_red",
                collision=False),
            box("wheel_spoke_b", (0, 0, stub_h + 0.17), (0.015, 0.22, 0.012), name, material="valve_red",
                collision=False),
        ]
        horiz = math.hypot(off[0], off[1])
        if horiz > 0.03:
            a = math.atan2(off[1], off[0])
            mid = np.array([off[0] / 2, off[1] / 2, off[2]])
            prims.append(cylinder("outlet", mid, 0.035, horiz, name, pitch=math.pi / 2, yaw=a, material="steel_pipe"))
            prims.append(cylinder("outlet_tee", (0, 0, off[2]), 0.055, 0.12, name, material="steel_pipe"))
            prims.append(cylinder("nozzle", (off[0], off[1], off[2]), 0.045, 0.04, name, pitch=math.pi / 2, yaw=a,
                                  material="valve_red", collision=False))
        mp = (vp[0], vp[1], zf, 0, 0, 0)
        for p in prims:
            p.world_pose = (p.pose[0] + mp[0], p.pose[1] + mp[1], p.pose[2] + mp[2]) + tuple(p.pose[3:])
            p.nav = FORCED_OCCUPIED if p.collision else NAV_IGNORE
            p.kind = "valve"
            self.prims.append(p)
        self.includes.append({"uri": f"model://{name}", "name": "gas_leak_valve", "pose": mp})
        self.leak = {"source": src.tolist(), "valve_prop": vp.tolist(), "emitter_yaw": wyaw}
        if write_models:
            emitter = particle_emitter_xml(off, wyaw, name)
            write_prim_model(os.path.join(self.models_dir, name), name, prims,
                             "Gas leak relief valve prop (stub, valve, hand wheel, outlet) with a green gas "
                             "ParticleEmitter at the leak source pointing downwind. Visual plume only: "
                             "particle_scatter_ratio 0 keeps it out of depth images.", extra=emitter)
            write_smoke_texture(os.path.join(self.models_dir, name, "materials", "textures", "smoke.png"))

    # ----------------------------------------------------------- charging station
    def build_dock(self, write_models: bool = True):
        cs = self.L["charging_station"]
        x, y = cs["pad"][0], cs["pad"][1]
        zf = self.floor_at(x, y)
        sx, sy, sz = cs.get("pad_size", [1.0, 1.0, 0.02])
        dock_yaw = float(cs["dock_pose"][2])
        name = "charging_station"
        # model frame: pad centre on the floor, +X = direction the docked robot faces
        bx = -(sx / 2 + 0.05)                                   # backboard behind the robot
        # The pad is flush with the floor (layout: pad top z == lower floor z) so
        # that the robot spawned at floor + 0.56 m does not start inside it:
        # the textured marking decal lies 2 mm above the floor (visual only).
        prims = [
            box("backboard", (bx - 0.04, 0, 0.36), (0.08, sy, 0.72), name, material="dock_board",
                nav=FORCED_OCCUPIED, kind="dock"),
            box("backboard_foot", (bx - 0.10, 0, 0.02), (0.20, sy, 0.04), name, material="dock_board",
                nav=FORCED_OCCUPIED, kind="dock"),
            box("contact_left", (bx + 0.005, 0.12, 0.30), (0.01, 0.06, 0.04), name, material="copper",
                collision=False, nav=NAV_IGNORE, kind="dock"),
            box("contact_right", (bx + 0.005, -0.12, 0.30), (0.01, 0.06, 0.04), name, material="copper",
                collision=False, nav=NAV_IGNORE, kind="dock"),
            box("status_led", (bx + 0.005, 0.0, 0.62), (0.01, 0.20, 0.03), name, material="led_green",
                collision=False, nav=NAV_IGNORE, kind="dock"),
        ]
        mp = (x, y, zf, 0.0, 0.0, dock_yaw)
        R = rot_rpy(0, 0, dock_yaw)
        for p in prims:
            c = R @ np.asarray(p.pose[:3]) + np.array(mp[:3])
            p.world_pose = (c[0], c[1], c[2], 0.0, 0.0, p.pose[5] + dock_yaw)
            self.prims.append(p)
        self.includes.append({"uri": f"model://{name}", "name": "charging_station", "pose": mp})
        self.dock = {"pad_centre": [x, y], "pad_size": [sx, sy], "pad_top_z": zf + 0.002, "floor_z": zf,
                     "dock_yaw": dock_yaw, "backboard_x_local": bx - 0.04}
        # footprint of the pad for the figures (no collision: flush with the floor)
        pad = box("charging_pad", (x, y, zf + 0.001), (sx, sy, 0.002), name, yaw=dock_yaw, nav=NAV_IGNORE,
                  collision=False, visual=False, kind="pad")
        self.prims.append(pad)
        if write_models:
            mdir = os.path.join(self.models_dir, name)
            os.makedirs(os.path.join(mdir, "meshes"), exist_ok=True)
            write_dock_marking(os.path.join(mdir, "meshes"), sx, sy, 0.002, name)
            mark = f"""      <visual name="marking">
        <geometry>
          <mesh>
            <uri>model://{name}/meshes/dock_marking.obj</uri>
          </mesh>
        </geometry>
      </visual>
"""
            write_prim_model(mdir, name, prims,
                             "Robot charging dock: pad marking flush with the floor (1 x 1 m decal), backboard "
                             "with charging contacts and status LED. Origin = pad centre on the floor, +X = "
                             "heading of the docked robot.", extra=mark)

    # ------------------------------------------------------------- access points
    def build_access_points(self, write_models: bool = True):
        name = "access_point"
        prims = [
            box("body", (0, 0, 0), (0.22, 0.16, 0.045), name, material="ap_white", collision=False),
            cylinder("antenna_l", (0.08, 0.05, 0.10), 0.006, 0.16, name, material="antenna", collision=False),
            cylinder("antenna_r", (0.08, -0.05, 0.10), 0.006, 0.16, name, material="antenna", collision=False),
            box("led", (0.111, 0.0, 0.0), (0.002, 0.04, 0.008), name, material="led_blue", collision=False),
        ]
        for ap in self.L.comm.get("access_points", []):
            px, py, pz = ap["position"]
            # mount on the nearer side wall or hang from the ceiling: face the room centre
            yaw = 0.0 if px < self.L["corridor"]["interior"]["x_min"] else (-math.pi / 2 if py > 0 else math.pi / 2)
            self.includes.append({"uri": f"model://{name}", "name": f"access_point_{ap['id']}",
                                  "pose": (px, py, pz, 0.0, 0.0, yaw)})
        if write_models:
            write_prim_model(os.path.join(self.models_dir, name), name, prims,
                             "Wireless access point prop (visual only, no collision). Position = "
                             "layout comm.access_points.")

    # -------------------------------------------------------------------- lights
    def build_lights(self):
        """Lamps where the meshes have light fixtures (emissive 'Light' faces,
        tubes of one fixture merged), a lamp every ~6 m along the corridor,
        one shadow-casting spot over the hall and a shadowless directional
        fill (the roofs would block a shadowed sun completely)."""
        fg, cg = self.geometry["factory"], self.geometry["corridor"]
        wf = fg["wall_faces"]
        z0 = fg["floor_z"]
        parts = fg["interior_walls"]
        x_hall1 = parts[0]["x0"] if parts else wf["east"]
        fixtures = light_clusters(self.meshes["factory_hall"], link=0.8)
        if len(fixtures) >= 4:
            for j, c in enumerate(sorted(fixtures, key=lambda q: (round(q[0], 1), q[1]))):
                high = c[2] > z0 + 4.0
                self.lights.append(point_light(f"hall_fixture_{j + 1:02d}", (c[0], c[1], c[2] - 0.15),
                                               rng=9.0 if high else 5.0, diffuse=0.80 if high else 0.55))
        else:                                          # fallback grid
            k = 0
            for x in np.linspace(wf["west"] + 3.5, x_hall1 - 3.0, 3):
                for y in np.linspace(wf["south"] + 4.5, wf["north"] - 4.5, 2):
                    k += 1
                    self.lights.append(point_light(f"hall_lamp_{k}", (x, y, z0 + 6.3), rng=18.0, diffuse=0.85))
        if parts:
            self.lights.append(point_light("backroom_lamp", ((x_hall1 + wf["east"]) / 2, 0.0, z0 + 4.2),
                                           rng=10.0, diffuse=0.7))
        self.lights.append(spot_light("hall_spot_shadow", (1.0, -1.0, z0 + 6.8), (0.0, -0.25, -1.0), rng=16.0,
                                      shadows=True))
        ci = self.L["corridor"]["interior"]
        x0c, x1c = ci["x_min"], cg["wall_faces"]["end"]
        n = max(2, int(round((x1c - x0c) / 6.0)))
        z_c = cg["upper_floor_z"] + self.L["corridor"]["height"] - 0.55
        for k in range(n):
            x = x0c + (k + 0.5) * (x1c - x0c) / n
            self.lights.append(point_light(f"corridor_lamp_{k + 1:02d}", (x, 0.0, z_c), rng=9.0,
                                           diffuse=0.75 if x < cg["descent"]["x_start"] else 0.85))
        self.lights.append(directional_light("sun_fill", (-0.35, 0.25, -1.0), diffuse=0.30))
        self.log(f"    lights: {len(self.lights)} ({len(fixtures)} hall fixtures from the mesh)")

    # =====================================================================
    # Nav map
    # =====================================================================
    def build_nav_map(self):
        import cv2
        g = self.grid
        hm, walk = self.floor_raster()
        # drop-offs: neighbouring walkable cells with a height jump > STEP_MAX
        drop = dropoff_mask(hm, STEP_MAX)
        forced = np.zeros(g.shape, np.uint8)
        auto = np.zeros(g.shape, bool)
        for p in self.prims:
            if p.nav == FORCED_OCCUPIED:
                g.fill(forced, p.footprint(), 1)
            elif p.nav == NAV_AUTO:
                m = np.zeros(g.shape, np.uint8)
                g.fill(m, p.footprint(), 1)
                m = m.astype(bool)
                zr = p.z_range()
                f = np.where(np.isfinite(hm), hm, 0.0)
                auto |= m & (zr[0] < f + CLEAR_HI) & (zr[1] > f + CLEAR_LO_BUILDING)
        occ = ~walk | drop | forced.astype(bool) | auto | self._obst_building | self._obst_machine
        free = ~occ
        # keep the free space connected to the dock
        n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
        dx_, dy_ = self.L.dock_pose[0], self.L.dock_pose[1]
        seed = lab[g.iy(dy_), g.ix(dx_)]
        if seed == 0:
            raise RuntimeError("dock position is not free in the nav map")
        free = lab == seed
        self._fine = {"free": free, "walk": walk, "drop": drop, "hm": hm, "forced": forced.astype(bool),
                      "auto": auto}
        # max-pool to the map resolution (a coarse cell is free only if all sub-cells are)
        res = float(self.L["nav_map"]["resolution"])
        f = int(round(res / g.res))
        H, W = g.h // f, g.w // f
        blk = free[:H * f, :W * f].reshape(H, f, W, f)
        coarse_free = blk.all(axis=(1, 3))
        coarse_free = reachable_free(coarse_free, res, g.x0, g.y0, (dx_, dy_), ROBOT_RADIUS)
        occ_c = np.where(coarse_free, 0, 100).astype(np.int8)
        self.nav = OccupancyGridMap(occ_c, res, g.x0, g.y0)
        self.log(f"    nav map {W} x {H} @ {res} m: free {coarse_free.mean() * 100:.1f} %, "
                 f"drop-off cells {int(drop.sum())}")
        return self.nav

    # =====================================================================
    # Output
    # =====================================================================
    def write_world(self, path: str = WORLD_PATH):
        L = self.L
        out = [f"""<?xml version="1.0"?>
<!--
  Virtual plant world '{L.get('world_name', 'plant')}' - GENERATED by tools/generate_world.py from
  simulation/config/world_layout.yaml and the converted Unity meshes.  Do not edit by hand:
  change the layout and re-run the generator.  The robot is spawned by the launch file.
  Model URIs (model://...) require GZ_SIM_RESOURCE_PATH to contain <share>/plant_simulation/models.
-->
<sdf version="{SDF_VERSION}">
  <world name="{L.get('world_name', 'plant')}">
    <physics name="1ms" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.80665</gravity>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>
    <plugin filename="gz-sim-particle-emitter-system" name="gz::sim::systems::ParticleEmitter"/>

    <scene>
      <ambient>0.42 0.42 0.42 1</ambient>
      <background>0.55 0.60 0.68 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- thermal camera: objects without a Thermal plugin appear at this temperature -->
    <atmosphere type="adiabatic">
      <temperature>{L.thermal.get('ambient_k', 298.15):.2f}</temperature>
      <pressure>101325</pressure>
      <temperature_gradient>-0.0065</temperature_gradient>
    </atmosphere>

    <!-- ============================ lights ============================ -->
"""]
        for li in self.lights:
            out.append(li["xml"])
        out.append("\n    <!-- ===================== building shells (visual meshes) ===================== -->\n")
        for key in ("factory", "corridor"):
            sec = L[key]
            px, py, pz, pyaw = (list(sec["pose"]) + [0, 0, 0, 0])[:4]
            out.append(include_xml(f"model://{sec['mesh_model']}", sec["mesh_model"], (px, py, pz, 0, 0, pyaw)))
        # models made of primitives
        order = ["plant_structure"] + sorted(k for k in self.models if k.startswith("machine_")) + \
            sorted(k for k in self.models if k not in ("plant_structure",) and not k.startswith("machine_"))
        titles = {"plant_structure": "static collision structure (measured from the meshes)"}
        for mname in order:
            rec = self.models[mname]
            out.append(f"\n    <!-- {titles.get(mname, mname)} -->\n")
            out.append(model_xml(mname, rec))
        out.append("\n    <!-- ====================== included models ====================== -->\n")
        for inc in self.includes:
            out.append(include_xml(inc["uri"], inc["name"], inc["pose"]))
        out.append("  </world>\n</sdf>\n")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(out))
        self.log(f"    wrote {os.path.relpath(path, REPO)} ({os.path.getsize(path) / 1024:.0f} kB, "
                 f"{len(self.prims)} primitives, {len(self.includes)} includes)")

    def write_summary(self, path: str):
        import yaml
        prims = []
        for p in self.prims:
            prims.append({"name": p.name, "model": p.group, "shape": p.shape,
                          "pose": [round(float(v), 4) for v in p.wpose()],
                          "size": [round(float(v), 4) for v in p.size], "kind": p.kind, "nav": p.nav,
                          "collision": bool(p.collision), "visual": bool(p.visual),
                          **({"temperature_k": p.temperature} if p.temperature else {}),
                          **({"mu": p.mu} if p.mu is not None else {})})
        data = {
            "generated_by": "tools/generate_world.py",
            "layout": os.path.relpath(self.L.path, REPO).replace("\\", "/") if self.L.path else None,
            "note": "Measured architecture and every primitive placed in plant.sdf (world frame). "
                    "nav: occupied = forced obstacle in the nav map, free = crossable debris, walkable = "
                    "floor surface, auto = obstacle only inside the robot height band.",
            "parameters": {"voxel_res": VOX_RES, "band_dz": VOX_DZ, "clear_lo_building": CLEAR_LO_BUILDING,
                           "clear_lo_machine": CLEAR_LO_MACHINE, "clear_hi": CLEAR_HI,
                           "machine_nav_hi": MACHINE_NAV_HI, "step_max": STEP_MAX, "robot_radius": ROBOT_RADIUS},
            "architecture": self.geometry,
            "gauges": self.gauge_frames,
            "gas_leak": getattr(self, "leak", {}),
            "dock": getattr(self, "dock", {}),
            "notes": self.notes,
            "primitives": prims,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(_plain(data), f, sort_keys=False, width=140)


MACHINE_TEMP_K = 303.15


def _plain(o):
    """numpy -> python for yaml."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float):
        return round(o, 5)
    return o


# =============================================================================
# SDF snippets
# =============================================================================
def material_xml(key: str | None, colour=None, indent: str = "          ") -> str:
    m = dict(MATERIALS.get(key, {})) if key else {}
    if colour is not None:
        m = {"diffuse": tuple(colour), "specular": 0.15, "pbr": (0.0, 0.85)}
    if not m:
        return ""
    d = m["diffuse"]
    s = m.get("specular", 0.2)
    lines = [f"{indent}<material>",
             f"{indent}  <ambient>{fmt_vec(d, 3)} 1</ambient>",
             f"{indent}  <diffuse>{fmt_vec(d, 3)} 1</diffuse>",
             f"{indent}  <specular>{fmt(s, 3)} {fmt(s, 3)} {fmt(s, 3)} 1</specular>"]
    if "emissive" in m:
        lines.append(f"{indent}  <emissive>{fmt_vec(m['emissive'], 3)} 1</emissive>")
    if "pbr" in m:
        mt, rg = m["pbr"]
        lines += [f"{indent}  <pbr>", f"{indent}    <metal>",
                  f"{indent}      <metalness>{fmt(mt, 3)}</metalness>",
                  f"{indent}      <roughness>{fmt(rg, 3)}</roughness>",
                  f"{indent}    </metal>", f"{indent}  </pbr>"]
    lines.append(f"{indent}</material>")
    return "\n".join(lines) + "\n"


def geometry_xml(p: Prim, indent: str) -> str:
    if p.shape == "box":
        return f"{indent}<geometry><box><size>{fmt_vec(p.size, 4)}</size></box></geometry>\n"
    r, Lc = p.size
    return (f"{indent}<geometry><cylinder><radius>{fmt(r, 4)}</radius><length>{fmt(Lc, 4)}</length>"
            f"</cylinder></geometry>\n")


def prim_xml(p: Prim, indent: str = "        ") -> str:
    pose = f"{indent}  <pose>{fmt_vec(p.pose[:3], 4)} {fmt_vec(p.pose[3:], 5)}</pose>\n"
    out = ""
    if p.collision:
        out += f'{indent}<collision name="{p.name}_collision">\n{pose}{geometry_xml(p, indent + "  ")}'
        if p.mu is not None:
            out += (f"{indent}  <surface><friction><ode><mu>{fmt(p.mu, 3)}</mu><mu2>{fmt(p.mu, 3)}</mu2></ode>"
                    f"</friction></surface>\n")
        out += f"{indent}</collision>\n"
    if p.visual:
        out += f'{indent}<visual name="{p.name}_visual">\n{pose}{geometry_xml(p, indent + "  ")}'
        out += material_xml(p.material, p.colour, indent + "  ")
        if p.temperature:
            out += thermal_plugin_xml(p.temperature, indent + "  ")
        out += f"{indent}</visual>\n"
    return out


def model_xml(name: str, rec: dict) -> str:
    body = "".join(prim_xml(p) for p in rec["prims"])
    return (f'    <model name="{name}">\n      <static>{"true" if rec.get("static", True) else "false"}</static>\n'
            f"      <pose>{fmt_vec(rec['pose'][:3], 4)} {fmt_vec(rec['pose'][3:], 5)}</pose>\n"
            f'      <link name="link">\n{rec.get("extra", "")}{body}      </link>\n    </model>\n')


def include_xml(uri: str, name: str, pose) -> str:
    p = list(pose) + [0.0] * (6 - len(pose))
    return (f"    <include>\n      <uri>{uri}</uri>\n      <name>{name}</name>\n"
            f"      <pose>{fmt_vec(p[:3], 4)} {fmt_vec(p[3:], 5)}</pose>\n    </include>\n")


def point_light(name, pos, rng=15.0, diffuse=0.8, shadows=False) -> dict:
    d = diffuse
    xml = f"""    <light type="point" name="{name}">
      <pose>{fmt_vec(pos, 3)} 0 0 0</pose>
      <cast_shadows>{'true' if shadows else 'false'}</cast_shadows>
      <diffuse>{fmt(d, 3)} {fmt(d * 0.97, 3)} {fmt(d * 0.92, 3)} 1</diffuse>
      <specular>0.15 0.15 0.15 1</specular>
      <attenuation>
        <range>{fmt(rng, 2)}</range>
        <constant>0.35</constant>
        <linear>0.04</linear>
        <quadratic>0.006</quadratic>
      </attenuation>
    </light>
"""
    return {"name": name, "type": "point", "pos": list(pos), "xml": xml}


def spot_light(name, pos, direction, rng=15.0, shadows=False) -> dict:
    xml = f"""    <light type="spot" name="{name}">
      <pose>{fmt_vec(pos, 3)} 0 0 0</pose>
      <cast_shadows>{'true' if shadows else 'false'}</cast_shadows>
      <diffuse>0.65 0.64 0.60 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>{fmt_vec(direction, 3)}</direction>
      <attenuation>
        <range>{fmt(rng, 2)}</range>
        <constant>0.4</constant>
        <linear>0.02</linear>
        <quadratic>0.002</quadratic>
      </attenuation>
      <spot>
        <inner_angle>0.6</inner_angle>
        <outer_angle>1.3</outer_angle>
        <falloff>1.0</falloff>
      </spot>
    </light>
"""
    return {"name": name, "type": "spot", "pos": list(pos), "xml": xml}


def directional_light(name, direction, diffuse=0.3) -> dict:
    xml = f"""    <light type="directional" name="{name}">
      <pose>0 0 30 0 0 0</pose>
      <cast_shadows>false</cast_shadows>
      <diffuse>{fmt(diffuse, 3)} {fmt(diffuse, 3)} {fmt(diffuse * 0.95, 3)} 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>{fmt_vec(direction, 3)}</direction>
    </light>
"""
    return {"name": name, "type": "directional", "pos": [0, 0, 30], "xml": xml}


def particle_emitter_xml(offset, yaw: float, model: str) -> str:
    """Green translucent gas plume at the leak source (like the reference video),
    emitting along +X (downwind).  Only the colours differ from the former grey
    smoke.  The ogre2 engine of gz-rendering 8 ignores color_start / color_end
    ("ParticleEmitter SetColorRange is currently disabled"), so the green comes
    from the material diffuse times the near-white, soft-edged smoke.png
    (checked in the Harmonic GUI); the green colour range is kept for render
    engines that apply it."""
    return f"""      <particle_emitter name="gas_plume" type="point">
        <pose>{fmt_vec(offset, 4)} 0 0 {fmt(yaw, 5)}</pose>
        <emitting>true</emitting>
        <size>0.05 0.05 0.05</size>
        <particle_size>0.10 0.10 0.10</particle_size>
        <lifetime>4</lifetime>
        <rate>12</rate>
        <min_velocity>0.15</min_velocity>
        <max_velocity>0.35</max_velocity>
        <scale_rate>0.45</scale_rate>
        <color_start>0.30 0.95 0.35 0.60</color_start>
        <color_end>0.60 1.00 0.55 0.0</color_end>
        <particle_scatter_ratio>0.0</particle_scatter_ratio>
        <material>
          <diffuse>0.55 1.0 0.55 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <pbr>
            <metal>
              <albedo_map>model://{model}/materials/textures/smoke.png</albedo_map>
            </metal>
          </pbr>
        </material>
      </particle_emitter>
"""


def write_prim_model(mdir: str, name: str, prims: list[Prim], description: str, extra: str = "") -> None:
    """model.sdf + model.config of a static model made of primitives (model frame poses)."""
    os.makedirs(mdir, exist_ok=True)
    body = "".join(prim_xml(p, "      ") for p in prims)
    sdf = (f'<?xml version="1.0"?>\n<sdf version="{SDF_VERSION}">\n  <model name="{name}">\n'
           f"    <static>true</static>\n    <link name=\"link\">\n{extra}{body}    </link>\n  </model>\n</sdf>\n")
    with open(os.path.join(mdir, "model.sdf"), "w", encoding="utf-8", newline="\n") as f:
        f.write(sdf)
    with open(os.path.join(mdir, "model.config"), "w", encoding="utf-8", newline="\n") as f:
        f.write(model_config_xml(name, description + " Generated by tools/generate_world.py."))


# =============================================================================
# Generated meshes / textures
# =============================================================================
DIAL_UV_RADIUS = 0.5      # texture radius (fraction of the side) at the disc rim: side == dial_diameter


def dial_disc_vertices(radius: float, segments: int = 72):
    """Disc in the local Y-Z plane facing +X (see module docstring) ->
    (vertices (N,3), uvs (N,2), triangles (M,3) CCW seen from +X).
    The rim (radius = dial_diameter / 2) maps onto the inscribed circle of the
    texture, i.e. the full texture side corresponds to dial_diameter."""
    R_t = DIAL_UV_RADIUS
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    y, z = radius * np.cos(ang), radius * np.sin(ang)
    v = np.vstack([[0.0, 0.0, 0.0], np.c_[np.zeros(segments), y, z]])
    uv = np.c_[0.5 + R_t * v[:, 1] / radius, 0.5 + R_t * v[:, 2] / radius]
    tris = np.array([[0, 1 + k, 1 + (k + 1) % segments] for k in range(segments)])
    return v, uv, tris


def write_dial_obj(path: str, radius: float, texture: str, material: str, segments: int = 72) -> None:
    v, uv, tris = dial_disc_vertices(radius, segments)
    mtl = os.path.splitext(os.path.basename(path))[0] + ".mtl"
    with open(os.path.join(os.path.dirname(path), mtl), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# gauge dial face, generated by tools/generate_world.py\nnewmtl {material}\n"
                "Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.25 0.25 0.25\nNs 40.0\nd 1.0\nillum 2\n"
                f"map_Kd {texture}\n")
    lines = ["# gauge dial: disc in the Y-Z plane, normal +X, texture up +Z, viewer right +Y",
             f"mtllib {mtl}", f"o {material}"]
    lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in v]
    lines += [f"vt {a:.6f} {b:.6f}" for a, b in uv]
    lines.append("vn 1 0 0")
    lines.append(f"usemtl {material}")
    lines += [f"f {a + 1}/{a + 1}/1 {b + 1}/{b + 1}/1 {c + 1}/{c + 1}/1" for a, b, c in tris]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def dock_marking_image(size: int = 1024) -> np.ndarray:
    """Landing marking: hazard border, 'H', docking arrow and label (BGR)."""
    import cv2
    S = size
    img = np.full((S, S, 3), (38, 38, 40), np.uint8)
    b = int(0.07 * S)
    # yellow / black hazard stripes on the border
    stripes = np.zeros((S, S), np.uint8)
    yy, xx = np.mgrid[0:S, 0:S]
    stripes[((xx + yy) // int(0.05 * S)) % 2 == 0] = 1
    border = np.zeros((S, S), bool)
    border[:b, :] = border[-b:, :] = True
    border[:, :b] = border[:, -b:] = True
    img[border & (stripes == 1)] = (0, 200, 255)
    img[border & (stripes == 0)] = (20, 20, 20)
    cv2.rectangle(img, (b, b), (S - b, S - b), (230, 230, 230), max(2, S // 200))
    # big H
    w, h = int(0.36 * S), int(0.42 * S)
    cx, cy = S // 2, int(0.47 * S)
    t = int(0.07 * S)
    col = (235, 235, 235)
    cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx - w // 2 + t, cy + h // 2), col, -1)
    cv2.rectangle(img, (cx + w // 2 - t, cy - h // 2), (cx + w // 2, cy + h // 2), col, -1)
    cv2.rectangle(img, (cx - w // 2, cy - t // 2), (cx + w // 2, cy + t // 2), col, -1)
    # docking direction arrow (image up = heading of the docked robot)
    pts = np.array([[cx, int(0.13 * S)], [cx - int(0.07 * S), int(0.22 * S)], [cx + int(0.07 * S), int(0.22 * S)]])
    cv2.fillPoly(img, [pts], (0, 200, 255), cv2.LINE_AA)
    text = "DOCK"
    sc = S / 1024 * 2.2
    th = max(2, int(S / 1024 * 6))
    (tw, tth), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
    cv2.putText(img, text, (cx - tw // 2, int(0.84 * S) + tth // 2), cv2.FONT_HERSHEY_SIMPLEX, sc,
                (0, 200, 255), th, cv2.LINE_AA)
    return img


def write_dock_marking(mesh_dir: str, sx: float, sy: float, z: float, prefix: str) -> None:
    """Textured quad on top of the pad.  Model frame: +X = heading of the docked
    robot = image up; viewer standing behind the robot looking along +X has his
    right hand along -Y = image right.  OBJ v = 0 is the image bottom."""
    import cv2
    tex = f"{prefix}_marking.png"
    cv2.imwrite(os.path.join(mesh_dir, tex), dock_marking_image(1024), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    hx, hy = sx / 2, sy / 2
    # corners: (x, y) -> (u, v):  u grows towards -Y, v grows towards +X
    corners = [(-hx, hy), (-hx, -hy), (hx, -hy), (hx, hy)]
    uvs = [(0.5 - y / sy, 0.5 + x / sx) for x, y in corners]
    with open(os.path.join(mesh_dir, "dock_marking.mtl"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"newmtl {prefix}_marking\nKa 1 1 1\nKd 1 1 1\nKs 0.1 0.1 0.1\nNs 10\nd 1.0\nillum 2\n"
                f"map_Kd {tex}\n")
    lines = ["# charging pad marking (top face, normal +Z)", "mtllib dock_marking.mtl", "o marking"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y in corners]
    lines += [f"vt {u:.5f} {v:.5f}" for u, v in uvs]
    lines += ["vn 0 0 1", f"usemtl {prefix}_marking", "f 1/1/1 2/2/1 3/3/1", "f 1/1/1 3/3/1 4/4/1"]
    with open(os.path.join(mesh_dir, "dock_marking.obj"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_smoke_texture(path: str, size: int = 128) -> None:
    """Soft radial puff (RGBA) for the particle emitter."""
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) / (size / 2)
    rng = np.random.RandomState(5)
    noise = cv2.GaussianBlur(rng.rand(size, size).astype(np.float32), (0, 0), size / 12)
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-9)
    alpha = np.clip(1.0 - r, 0, 1) ** 1.6 * (0.65 + 0.35 * noise)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = (220 + 30 * noise[..., None]).clip(0, 255).astype(np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    cv2.imwrite(path, rgba)


# =============================================================================
# Figures
# =============================================================================
KIND_COLOURS = {
    "wall": "#3a4250", "feature": "#59606e", "machine": "#8c5a3c", "tread": "#9aa6b8", "ramp": "#b8c4d6",
    "pipe": "#2f7fbf", "pallet": "#2e7d32", "rubble": "#a1887f", "plate": "#90a4ae", "drum": "#c62828",
    "platform": "#f9a825", "rack_support": "#1a3d7c", "rack_pipe": "#c8a24a", "hotspot": "#ff5722",
    "gauge": "#1565c0", "valve": "#ad1457", "dock": "#2e7d32", "pad": "#43a047",
}


def render_figures(wb: WorldBuilder, layout_png: str, nav_png: str, test_paths: dict | None = None) -> list[str]:
    """docs/images/world_layout.png and nav_map.png (matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly, Rectangle
    except Exception as e:  # pragma: no cover
        wb.log(f"    matplotlib not available ({e}); figures skipped")
        return []
    L = wb.L
    g = wb.grid
    fine = wb._fine
    ext = (g.x0, g.x0 + g.w * g.res, g.y0, g.y0 + g.h * g.res)

    def draw_world(ax, xlim, ylim, labels=True):
        hm = fine["hm"]
        base = np.ones(g.shape + (3,)) * 0.97
        base[fine["walk"]] = (0.86, 0.87, 0.85)
        lowfl = fine["walk"] & (hm < -0.4)
        base[lowfl] = (0.80, 0.84, 0.88)
        base[wb._obst_building] = (0.55, 0.57, 0.60)
        base[wb._obst_machine] = (0.80, 0.66, 0.55)
        base[fine["drop"]] = (0.85, 0.15, 0.15)
        ax.imshow(base, origin="lower", extent=ext, interpolation="nearest")
        for p in wb.prims:
            if p.kind in ("floor", "landing") or p.nav == NAV_IGNORE and p.kind not in ("hotspot", "pad"):
                continue
            col = KIND_COLOURS.get(p.kind)
            if col is None:
                continue
            zorder = 3 if p.kind in ("gauge", "hotspot", "valve") else 2
            alpha = 0.55 if p.kind in ("tread", "ramp", "plate", "platform", "pad") else 0.9
            ax.add_patch(MplPoly(p.footprint(), closed=True, fc=col, ec="none", alpha=alpha, zorder=zorder))
        # gauges with normals
        for gf in wb.gauge_frames:
            x, y, _ = gf["position"]
            nx, ny, _ = gf["normal"]
            ax.annotate("", xy=(x + 1.2 * nx, y + 1.2 * ny), xytext=(x, y), annotation_clip=True,
                        arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.4), zorder=5)
            ax.plot(x, y, "o", ms=5, mfc="#64b5f6", mec="#0d47a1", zorder=6)
            if labels:
                ax.text(x - 0.55 * nx + 0.15, y - 0.55 * ny + 0.15, gf["id"], fontsize=7, color="#0d47a1",
                        zorder=6, weight="bold", clip_on=True)
        for h in L.hotspots:
            x, y, _ = h["position"]
            st = L.temperature_status(h["temp_k"])
            col = {"alarm": "#d50000", "warning": "#ff9100", "normal": "#ffd600"}[st]
            ax.plot(x, y, marker="o", ms=9, mfc="none", mec=col, mew=2, zorder=6)
            if labels:
                ax.text(x + 0.2, y - 0.45, f"{h['id']} {h['temp_k'] - 273.15:.0f}C", fontsize=6.5, color=col,
                        zorder=6, weight="bold", clip_on=True)
        gl = L.gas_leak
        sx, sy = gl["source"][:2]
        ax.plot(sx, sy, marker="*", ms=13, mfc="#7b1fa2", mec="k", zorder=7)
        w = np.asarray(gl.get("wind_base", [0.4, -0.1]), float)
        ax.annotate("", xy=(sx + 4 * w[0], sy + 4 * w[1]), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color="#7b1fa2", lw=1.5, ls="--"), zorder=7)
        if labels:
            ax.text(sx + 0.2, sy + 0.3, "LEAK + wind", fontsize=7, color="#7b1fa2", weight="bold", zorder=7, clip_on=True)
        dx, dy, dyaw = L.dock_pose
        ax.annotate("", xy=(dx + 0.9 * math.cos(dyaw), dy + 0.9 * math.sin(dyaw)), xytext=(dx, dy),
                    arrowprops=dict(arrowstyle="-|>", color="#1b5e20", lw=2), zorder=7)
        if labels:
            ax.text(dx - 0.4, dy + 0.7, "DOCK / spawn", fontsize=7, color="#1b5e20", weight="bold", zorder=7, clip_on=True)
        for ap in L.comm.get("access_points", []):
            x, y, _ = ap["position"]
            ax.plot(x, y, marker="D", ms=6, mfc="#212121", mec="w", zorder=7)
            if labels:
                ax.text(x + 0.2, y + 0.2, ap["id"], fontsize=7, zorder=7, clip_on=True)
        for z in L.get("zones", []):
            x0, y0, x1, y1 = z["rect"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls=":", lw=0.8, ec="#555", zorder=1))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, lw=0.3, alpha=0.5)

    files = []
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.6, 1.1])
    ax0 = fig.add_subplot(gs[0, :])
    draw_world(ax0, (ext[0], ext[1]), (ext[2], ext[3]), labels=False)
    ax0.set_title("Virtual plant - overview (grey = mesh obstacles at robot height, red = drop-offs, "
                  "light blue = lower corridor level)", fontsize=10)
    ax1 = fig.add_subplot(gs[1, 0])
    fw = wb.geometry["factory"]["wall_faces"]
    draw_world(ax1, (fw["west"] - 0.6, fw["east"] + 1.2), (fw["south"] - 0.6, fw["north"] + 0.6))
    ax1.set_title("Turbine hall: machines, gauges (normals), pipe rack hot spots, leak, obstacles", fontsize=10)
    ax2 = fig.add_subplot(gs[1, 1])
    draw_world(ax2, (14.0, 32.0), (-3.0, 3.0))
    ax2.set_title("Corridor west: door, pipes, plate, rubble, G07", fontsize=10)
    ax3 = fig.add_subplot(gs[2, :])
    draw_world(ax3, (36.0, 64.2), (-2.6, 2.6))
    ax3.set_title("Corridor east: H05, rubble, descent (stairs south / ramp north), lower level, G08, dock",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(layout_png, dpi=110)
    plt.close(fig)
    files.append(layout_png)

    # nav map figure
    nav = wb.nav
    fig, axes = plt.subplots(3, 1, figsize=(17, 13), gridspec_kw={"height_ratios": [0.9, 1.5, 0.8]})
    occ = nav.occ
    infl = nav.inflated_obstacles(ROBOT_RADIUS)
    rgb = np.ones(occ.shape + (3,))
    rgb[infl & (occ == 0)] = (0.93, 0.80, 0.80)
    rgb[occ != 0] = (0.15, 0.15, 0.15)
    ext2 = (nav.origin_x, nav.origin_x + nav.width * nav.resolution, nav.origin_y,
            nav.origin_y + nav.height * nav.resolution)
    for ax, (xl, yl, title) in zip(axes, [
            ((ext2[0], ext2[1]), (ext2[2], ext2[3]), f"plant_map ({nav.width} x {nav.height} @ {nav.resolution} m)"
             f" - black occupied, pink = within robot radius {ROBOT_RADIUS} m, lines = A* test paths from the dock"
             " (terrain cost: around pipes / rubble where possible)"),
            ((-10.8, 16.0), (-9.8, 9.8), "turbine hall"), ((36.0, 64.2), (-2.4, 2.4), "corridor descent / lower")]):
        ax.imshow(rgb, origin="lower", extent=ext2, interpolation="nearest")
        if test_paths:
            for k, (lab, path) in enumerate(test_paths.items()):
                if path:
                    p = np.asarray(path)
                    ax.plot(p[:, 0], p[:, 1], "-", lw=1.1, color=plt.cm.tab10(k % 10))
                    ax.plot(p[-1, 0], p[-1, 1], "o", ms=4, color=plt.cm.tab10(k % 10))
        for gf in wb.gauge_frames:
            ax.plot(gf["position"][0], gf["position"][1], "^", ms=5, color="#1565c0")
        ax.plot(*L.gas_leak["source"][:2], "*", ms=10, color="#7b1fa2")
        ax.plot(L.dock_pose[0], L.dock_pose[1], "s", ms=6, color="#2e7d32")
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(nav_png, dpi=110)
    plt.close(fig)
    files.append(nav_png)
    return files


# =============================================================================
# Test goals (shared with tools/test/test_world.py)
# =============================================================================
def test_goals(layout: Layout) -> dict:
    """Goal points of the reachability requirements (from the dock)."""
    goals = {"factory_centre": (0.0, 0.0)}
    for gdef in layout.gauges:
        p = np.asarray(gdef["position"], float)
        n = np.asarray(gdef["normal"], float)
        n2 = n[:2] / max(np.linalg.norm(n[:2]), 1e-9)
        q = p[:2] + 1.5 * n2
        goals[f"gauge_{gdef['id']}"] = (float(q[0]), float(q[1]))
    src = np.asarray(layout.gas_leak["source"][:2], float)
    w = np.asarray(layout.gas_leak.get("wind_base", [1.0, 0.0]), float)
    d = w / max(np.linalg.norm(w), 1e-9)
    goals["gas_source_0.8m"] = (float(src[0] + 0.8 * d[0]), float(src[1] + 0.8 * d[1]))
    return goals


def plan_test_paths(nav: OccupancyGridMap, layout: Layout) -> dict:
    """A* paths dock -> test goals with the planners' terrain cost layer (detours around
    pipes / rubble where the map leaves room, as mission / RTH / patrol plan them)."""
    from dusan_common.planning import plan
    from dusan_common.terrain_cost import terrain_cost_grid
    start = layout.dock_pose[:2]
    cost = terrain_cost_grid(nav, layout)
    return {k: plan(nav, start, g, robot_radius=ROBOT_RADIUS, extra_cost=cost) for k, g in test_goals(layout).items()}


# =============================================================================
# main
# =============================================================================
def generate(layout_path: str = LAYOUT_PATH, models_dir: str = MODELS_DIR, world_path: str = WORLD_PATH,
             map_yaml: str | None = None, images: bool = True, write_models: bool = True, log=print) -> WorldBuilder:
    t0 = time.time()
    L = Layout.load(layout_path)
    wb = WorldBuilder(L, models_dir, log)
    wb.build_architecture()
    log("== robot-height analysis")
    wb.mesh_obstacles()
    wb.residual_features()
    wb.build_machines()
    log("== world content")
    wb.build_obstacles()
    wb.build_pipe_rack()
    wb.build_gauges(write_models)
    wb.build_gas_leak(write_models)
    wb.build_dock(write_models)
    wb.build_access_points(write_models)
    wb.build_lights()
    log("== nav map")
    nav = wb.build_nav_map()
    map_yaml = map_yaml or L.nav_map_yaml()
    os.makedirs(os.path.dirname(map_yaml), exist_ok=True)
    nav.save(map_yaml)
    log(f"    wrote {os.path.relpath(map_yaml, REPO)}")
    log("== world")
    wb.write_world(world_path)
    wb.write_summary(os.path.join(GENERATED_DIR, "world_geometry.yaml"))
    if images:
        log("== figures")
        paths = plan_test_paths(nav, L)
        for k, p in paths.items():
            log(f"    A* dock -> {k}: {'OK' if p else 'FAILED'}")
        os.makedirs(IMAGES_DIR, exist_ok=True)
        for f in render_figures(wb, os.path.join(IMAGES_DIR, "world_layout.png"),
                                os.path.join(IMAGES_DIR, "nav_map.png"), paths):
            log(f"    wrote {os.path.relpath(f, REPO)}")
    for n in wb.notes:
        log(f"    NOTE: {n}")
    log(f"done [{time.time() - t0:.1f} s]")
    return wb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default=LAYOUT_PATH)
    ap.add_argument("--no-images", action="store_true", help="skip docs/images figures")
    args = ap.parse_args(argv)
    generate(args.layout, images=not args.no_images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
