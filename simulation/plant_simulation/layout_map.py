"""
Fallback 2-D navigation map rasterised directly from world_layout.yaml.

The real map (simulation/maps/plant_map.yaml/.pgm) is produced by
tools/generate_world.py.  If it is missing (fresh checkout, generator not run)
map_server_lite rasterises this coarse equivalent so that planning, the
mission manager and the dashboard keep working.  Semantics follow
DESIGN_SPEC section 2: walls (outer walls, the hall partition with its
doorway, the corridor entrance frame), machines, drums, pallets, pipe-rack
supports, gauge stands and the ramp drop-off are OCCUPIED; stairs/ramps/low
debris (pipes, rubble, plates, low platforms) are FREE.
"""
import math

import numpy as np

from dusan_common.grid_map import OccupancyGridMap

PALLET_SIZE = (1.22, 1.40)      # x, y of one pallet footprint (m)

# map_server_lite parameters (kept here so that they can be checked without ROS)
MAP_SERVER_DEFAULTS = {
    "map_yaml": "",
    "layout_file": "",
    "frame_id": "map",
    "topic": "/map",
    "republish_period": 5.0,
    "fallback_to_layout": True,
}


def _rect_mask(xx, yy, x0, y0, x1, y1):
    return (xx >= min(x0, x1)) & (xx <= max(x0, x1)) & (yy >= min(y0, y1)) & (yy <= max(y0, y1))


def _oriented_rect_mask(xx, yy, cx, cy, sx, sy, yaw):
    """Cells whose centre lies inside a rectangle of size (sx, sy) centred at
    (cx, cy) rotated by yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = xx - cx, yy - cy
    lx = c * dx + s * dy        # rotate world offset into the rectangle frame
    ly = -s * dx + c * dy
    return (np.abs(lx) <= sx / 2.0) & (np.abs(ly) <= sy / 2.0)


def _disc_mask(xx, yy, cx, cy, r):
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def rasterize_layout(layout, resolution=None, bounds=None):
    """Return an OccupancyGridMap (0 free / 100 occupied) built from a Layout."""
    nav = layout.get("nav_map", {}) or {}
    res = float(resolution or nav.get("resolution", 0.10))
    x_min, y_min, x_max, y_max = bounds or nav.get("bounds", [-10.6, -9.6, 64.2, 9.6])
    w = int(math.ceil((x_max - x_min) / res))
    h = int(math.ceil((y_max - y_min) / res))
    xs = x_min + (np.arange(w) + 0.5) * res
    ys = y_min + (np.arange(h) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys)          # row = y, col = x (OccupancyGrid convention)

    free = np.zeros((h, w), dtype=bool)
    fac = layout.get("factory", {}) or {}
    if "interior" in fac:
        i = fac["interior"]
        free |= _rect_mask(xx, yy, i["x_min"], i["y_min"], i["x_max"], i["y_max"])
    cor = layout.get("corridor", {}) or {}
    if "interior" in cor:
        i = cor["interior"]
        free |= _rect_mask(xx, yy, i["x_min"], i["y_min"], i["x_max"], i["y_max"])
    op = fac.get("east_opening")
    if op and "interior" in cor:
        free |= _rect_mask(xx, yy, op["x"] - 0.3, op["y_min"], cor["interior"]["x_min"] + 0.3, op["y_max"])

    occ = ~free
    # interior wall of the hall: full room depth except the doorway
    part = fac.get("partition")
    if part and "interior" in fac:
        i = fac["interior"]
        wall = _rect_mask(xx, yy, part["x_min"], i["y_min"], part["x_max"], i["y_max"])
        d0, d1 = part.get("door_y", (0.0, 0.0))[:2]
        occ |= wall & ~((yy >= min(d0, d1)) & (yy <= max(d0, d1)))
        # a wall thinner than a cell must still cut the raster: mark the column the wall lies in
        xc = 0.5 * (part["x_min"] + part["x_max"])
        col = np.abs(xx - xc) <= 0.5 * res
        occ |= col & (yy >= i["y_min"]) & (yy <= i["y_max"]) & ~((yy >= min(d0, d1)) & (yy <= max(d0, d1)))
    door = cor.get("door_frame")
    if door and "interior" in cor:
        i = cor["interior"]
        frame = _rect_mask(xx, yy, door["x_min"], i["y_min"], door["x_max"], i["y_max"])
        occ |= frame & ((yy < door["y_min"]) | (yy > door["y_max"]))
    drop = cor.get("dropoff")
    if drop:
        occ |= _rect_mask(xx, yy, drop["x_min"], drop["y_min"], drop["x_max"], drop["y_max"])
    for m in layout.get("machines", []) or []:
        fp = m.get("footprint")
        if fp:
            occ |= _rect_mask(xx, yy, fp["x_min"], fp["y_min"], fp["x_max"], fp["y_max"])
    for ob in layout.obstacles:
        t = ob.get("type")
        if t == "drum":
            occ |= _disc_mask(xx, yy, ob["center"][0], ob["center"][1], float(ob.get("radius", 0.3)))
        elif t == "pallet":
            occ |= _oriented_rect_mask(xx, yy, ob["center"][0], ob["center"][1], PALLET_SIZE[0], PALLET_SIZE[1],
                                       float(ob.get("yaw", 0.0)))
        # pipe / rubble / plate / platform: traversable debris -> free
    rack = layout.get("pipe_rack")
    if rack:
        py = float(rack["pipe"]["start"][1])
        sx, sy = rack.get("support_size", [0.12, 0.5, 1.5])[:2]
        for x in rack.get("supports_x", []):
            occ |= _rect_mask(xx, yy, x - sx / 2.0, py - sy / 2.0, x + sx / 2.0, py + sy / 2.0)
    for g in layout.gauges:
        # gauge stand / mounting post just behind the dial face
        px, py, _ = g["position"]
        nx, ny = g.get("normal", [0, 1, 0])[:2]
        occ |= _disc_mask(xx, yy, px - 0.08 * nx, py - 0.08 * ny, 0.12)
    stub = layout.get("corridor_pipe_stub")
    if stub:
        occ |= _disc_mask(xx, yy, stub["position"][0], stub["position"][1], float(stub.get("radius", 0.06)) + 0.04)
    leak = layout.gas_leak
    if leak.get("valve_prop"):
        vx, vy = leak["valve_prop"][:2]
        occ |= _disc_mask(xx, yy, vx, vy, 0.10)

    grid = np.where(occ, 100, 0).astype(np.int8)
    return OccupancyGridMap(grid, res, x_min, y_min)
