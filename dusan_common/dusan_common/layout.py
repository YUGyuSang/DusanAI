"""
Access to simulation/config/world_layout.yaml (the single source of truth for
object positions, gauge ground truth, hot spots, gas leak, access points ...).
"""
import os
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))


def default_layout_path():
    """Installed share dir of plant_simulation, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("plant_simulation"), "config", "world_layout.yaml")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    # source tree fallback: <repo>/dusan_common/dusan_common/layout.py -> <repo>/simulation/...
    p = os.path.normpath(os.path.join(_HERE, "..", "..", "simulation", "config", "world_layout.yaml"))
    return p


def plant_share_dir():
    """Directory that contains config/, maps/, worlds/ of plant_simulation."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory("plant_simulation")
    except Exception:
        return os.path.normpath(os.path.join(_HERE, "..", "..", "simulation"))


def _tread_x(t, key_layout, key_geometry):
    return float(t[key_layout] if key_layout in t else t[key_geometry])


def _stair_dict(treads, stairs_y, ramp_y):
    # tread x extent: {x_min, x_max} (world_layout.yaml) or {x0, x1} (world_geometry.yaml)
    tr = sorted((_tread_x(t, "x_min", "x0"), _tread_x(t, "x_max", "x1"), float(t["z"])) for t in treads)
    sy = (float(min(stairs_y)), float(max(stairs_y)))
    ry = None if ramp_y is None else (float(min(ramp_y)), float(max(ramp_y)))
    return {"treads": tr, "stairs_y": sy, "ramp_y": ry}


def _band_dist(v, band):
    return max(band[0] - v, 0.0, v - band[1])


def _on_stairs_side(y, stairs_y, ramp_y):
    """y belongs to the stairs half of the descent: nearer to the stairs band
    than to the ramp band (ties -> ramp); inside the stairs band without a
    ramp band."""
    if ramp_y is None:
        return stairs_y[0] <= y <= stairs_y[1]
    return _band_dist(y, stairs_y) < _band_dist(y, ramp_y)


def _tread_z(x, treads, up_z, lo_z):
    """Step-function floor along the stairs: upper floor before the first
    tread, the (highest) tread under x, lower floor after the last tread."""
    under = [z for x0, x1, z in treads if x0 <= x <= x1]
    if under:
        return max(under)
    if x < treads[0][0]:
        return up_z
    if x > max(t[1] for t in treads):
        return lo_z
    return [z for x0, _x1, z in treads if x0 <= x][-1]     # between treads: last one passed


class Layout:
    def __init__(self, data, path=None):
        self.data = data
        self.path = path
        self._stairs = None          # lazily resolved corridor stair profile (False = none)

    @classmethod
    def load(cls, path=None):
        path = path or default_layout_path()
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f), path)

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    # ------------------------------------------------------------ helpers
    @property
    def gauges(self):
        defaults = self.data.get("gauge_defaults", {})
        out = []
        for g in self.data.get("gauges", []):
            d = dict(defaults)
            d.update(g)
            out.append(d)
        return out

    def gauge(self, gauge_id):
        for g in self.gauges:
            if g["id"] == gauge_id:
                return g
        return None

    @property
    def hotspots(self):
        return list(self.data.get("hotspots", []))

    @property
    def obstacles(self):
        return list(self.data.get("obstacles", []))

    @property
    def thermal(self):
        return dict(self.data.get("thermal", {}))

    @property
    def gas_leak(self):
        return dict(self.data.get("gas_leak", {}))

    @property
    def comm(self):
        return dict(self.data.get("comm", {}))

    @property
    def dock_pose(self):
        """(x, y, yaw) of the charging dock."""
        return tuple(self.data["charging_station"]["dock_pose"])

    @property
    def spawn(self):
        return tuple(self.data["robot_spawn"])

    def nav_map_yaml(self):
        rel = self.data.get("nav_map", {}).get("yaml", "maps/plant_map.yaml")
        if os.path.isabs(rel):
            return rel
        base = os.path.dirname(os.path.dirname(os.path.abspath(self.path))) if self.path else plant_share_dir()
        return os.path.join(base, rel)

    def floor_z(self, x, y):
        """Floor height at (x, y): corridor levels, linear ramp, and a step
        function on the stairs side of the descent when tread data exist
        (see ``stair_profile``)."""
        c = self.data.get("corridor", {})
        if c and x >= c["interior"]["x_min"] - 0.05:
            up = c["upper_floor"]
            lo = c["lower_floor"]
            de = c["descent"]
            st = self.stair_profile()
            if st is not None and _on_stairs_side(y, st["stairs_y"], st["ramp_y"]):
                return _tread_z(x, st["treads"], float(up["z"]), float(lo["z"]))
            if x <= de["x_min"]:
                return up["z"]
            if x >= de["x_max"]:
                return lo["z"]
            t = (x - de["x_min"]) / (de["x_max"] - de["x_min"])
            return up["z"] + t * (lo["z"] - up["z"])
        return self.data.get("factory", {}).get("floor_z", 0.0)

    def stair_profile(self):
        """Corridor stairs as {"treads": [(x0, x1, z), ...] sorted by x0,
        "stairs_y": (y_min, y_max), "ramp_y": (y_min, y_max) or None}, or None.

        Sources, first match wins: ``corridor.stairs.treads`` in the layout
        (list of {x_min, x_max, z}; {x0, x1, z} also accepted); else the
        measured architecture in
        ``config/generated/world_geometry.yaml`` next to the layout file
        (written by tools/generate_world.py from the corridor mesh).  Without
        either, floor_z interpolates linearly over the whole descent."""
        if getattr(self, "_stairs", None) is None:
            try:
                self._stairs = self._resolve_stairs() or False
            except Exception:  # malformed optional data: keep the linear model
                self._stairs = False
        return self._stairs or None

    def _resolve_stairs(self):
        c = self.data.get("corridor") or {}
        st = c.get("stairs") or {}
        ramp = c.get("ramp") or {}
        if st.get("treads"):
            return _stair_dict(st["treads"], (st["y_min"], st["y_max"]),
                               (ramp["y_min"], ramp["y_max"]) if "y_min" in ramp and "y_max" in ramp else None)
        if not self.path:
            return None
        gp = os.path.join(os.path.dirname(os.path.abspath(self.path)), "generated", "world_geometry.yaml")
        if not os.path.isfile(gp):
            return None
        with open(gp, "r", encoding="utf-8") as f:
            geo = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or {}
        gc = (geo.get("architecture") or {}).get("corridor") or {}
        gst = gc.get("stairs") or {}
        gramp = gc.get("ramp") or {}
        if not gst.get("treads"):
            return None
        return _stair_dict(gst["treads"], (gst["y_min"], gst["y_max"]),
                           (gramp["y0"], gramp["y1"]) if "y0" in gramp and "y1" in gramp else None)

    def gauge_status(self, gauge_id, value):
        """'normal' | 'warning' | 'alarm' for a reading."""
        g = self.gauge(gauge_id)
        if g is None or value is None:
            return "unknown"
        if value < g.get("alarm_min", -1e18) or value > g.get("alarm_max", 1e18):
            return "alarm"
        if value < g.get("normal_min", -1e18) or value > g.get("normal_max", 1e18):
            return "warning"
        return "normal"

    def temperature_status(self, temp_k):
        t = self.thermal
        if temp_k >= t.get("alarm_k", 353.15):
            return "alarm"
        if temp_k >= t.get("warning_k", 333.15):
            return "warning"
        return "normal"
