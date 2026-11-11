"""Regression tests for the integration-audit fixes in dusan_common
(image fast path, non-finite poses, map I/O, smoothing clearance, pure
pursuit on sparse paths, sensor QoS depth, stair floor heights)."""
import array
import math
import os
import sys
import tempfile
import threading
import types

import numpy as np
import pytest

from dusan_common import image_utils, ros_helpers
from dusan_common.grid_map import OccupancyGridMap, read_pgm
from dusan_common.layout import Layout
from dusan_common.planning import astar, plan
from dusan_common.pure_pursuit import PurePursuit

NAN, INF = float("nan"), float("inf")


# --------------------------------------------------------------------------- image
class _RosidlLikeImage:
    """Mimics the rosidl-generated setter of a uint8[] field: array.array('B')
    is stored as is, anything else is validated element by element."""

    def __init__(self):
        self.header = types.SimpleNamespace(frame_id="", stamp=None)
        self.height = self.width = self.step = self.is_bigendian = 0
        self.encoding = ""
        self._data = array.array("B")
        self.slow_path = False

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, array.array):
            assert value.typecode == "B"
            self._data = value
            return
        self.slow_path = True
        assert all(isinstance(v, int) for v in value) and all(0 <= v < 256 for v in value)
        self._data = array.array("B", value)


@pytest.fixture()
def fake_sensor_msgs(monkeypatch):
    pkg = types.ModuleType("sensor_msgs")
    msg = types.ModuleType("sensor_msgs.msg")
    msg.Image = _RosidlLikeImage
    pkg.msg = msg
    monkeypatch.setitem(sys.modules, "sensor_msgs", pkg)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", msg)


def test_numpy_to_image_msg_uses_array_fast_path(fake_sensor_msgs):
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    m = image_utils.bgr_to_image_msg(bgr, frame_id="cam", stamp="t")
    assert isinstance(m.data, array.array) and m.data.typecode == "B" and not m.slow_path
    assert (m.height, m.width, m.step, m.encoding, m.header.frame_id) == (60, 80, 240, "bgr8", "cam")
    assert np.array_equal(image_utils.image_msg_to_bgr(m), bgr)
    depth = rng.random((5, 7)).astype(np.float32)
    m2 = image_utils.numpy_to_image_msg(depth, "32FC1")
    assert not m2.slow_path and m2.step == 7 * 4
    assert np.array_equal(image_utils.image_msg_to_numpy(m2), depth)


# --------------------------------------------------------------- non-finite poses
def _room():
    occ = np.zeros((60, 100), np.int8)
    occ[:, 0] = occ[:, -1] = occ[0, :] = occ[-1, :] = 100
    occ[0:45, 50] = 100
    return OccupancyGridMap(occ, 0.1, -5.0, -3.0)


@pytest.mark.parametrize("bad", [(NAN, 0.0), (0.0, NAN), (INF, 0.0), (0.0, -INF), (1e308, 0.0)])
def test_grid_queries_survive_non_finite_pose(bad):
    g = _room()
    x, y = bad
    ix, iy = g.world_to_grid(x, y)
    assert not g.in_bounds(ix, iy)
    assert g.is_free(x, y) is False and g.is_free(x, y, 0.3) is False
    assert g.clearance(x, y) == 0.0
    assert g.nearest_free(x, y, 0.3) is None
    assert g.line_of_sight(x, y, 1.0, 1.0) is False and g.line_of_sight(1.0, 1.0, x, y, 0.3) is False
    n, frac = g.raycast_crossings(x, y, 1.0, 1.0)
    assert isinstance(n, int) and n >= 0 and 0.0 <= frac <= 1.0
    assert astar(g, (x, y), (3.0, -1.0)) is None
    assert plan(g, (-3.0, -1.0), (x, y)) is None
    ixa, iya = g.world_to_grid_array(np.array([[x, y], [-3.0, -1.0]]))
    assert not g.in_bounds(int(ixa[0]), int(iya[0]))
    assert (int(ixa[1]), int(iya[1])) == g.world_to_grid(-3.0, -1.0)


def test_raycast_far_endpoint_is_clipped_not_exploded():
    g = _room()
    # robot flung 1e9 m away: same walls crossed as the segment to the map edge
    far = g.raycast_crossings(-3.0, -1.0, 1e9, -1.0)
    edge = g.raycast_crossings(-3.0, -1.0, 4.95, -1.0)
    assert far[0] == edge[0] == 2          # inner wall (x = 0.0) + east wall (x = 4.9)
    assert g.raycast_crossings(1e9, 1e9, 2e9, 1e9) == (0, 0.0)


def _old_line_of_sight(g, x0, y0, x1, y1, radius=0.0):
    step = g.resolution * 0.5
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    ix = np.floor((xs - g.origin_x) / g.resolution).astype(int)
    iy = np.floor((ys - g.origin_y) / g.resolution).astype(int)
    if np.any((ix < 0) | (ix >= g.width) | (iy < 0) | (iy >= g.height)):
        return False
    mask = g.inflated_obstacles(radius) if radius > 0 else g.obstacle_mask()
    return not bool(np.any(mask[iy, ix]))


def _old_raycast(g, x0, y0, x1, y1):
    step = g.resolution * 0.5
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    ix = np.floor((xs - g.origin_x) / g.resolution).astype(int)
    iy = np.floor((ys - g.origin_y) / g.resolution).astype(int)
    valid = (ix >= 0) & (ix < g.width) & (iy >= 0) & (iy < g.height)
    occ = np.zeros(n, dtype=bool)
    occ[valid] = g.occ[iy[valid], ix[valid]] >= 50
    return int(np.count_nonzero(occ[1:] & ~occ[:-1])), float(occ.mean())


def test_finite_ray_queries_unchanged():
    g = _room()
    rng = np.random.default_rng(1)
    pts = rng.uniform([-7.0, -5.0, -7.0, -5.0], [7.0, 5.0, 7.0, 5.0], (300, 4))   # partly off-map
    for x0, y0, x1, y1 in pts:
        for r in (0.0, 0.25):
            assert g.line_of_sight(x0, y0, x1, y1, r) == _old_line_of_sight(g, x0, y0, x1, y1, r)
        assert g.raycast_crossings(x0, y0, x1, y1) == _old_raycast(g, x0, y0, x1, y1)


# ------------------------------------------------------------------------ map I/O
def test_save_load_keeps_unknown_cells():
    occ = np.array([[-1, 0, 100], [100, -1, 0]], np.int8)
    g = OccupancyGridMap(occ, 0.05, 1.0, 2.0)
    with tempfile.TemporaryDirectory() as d:
        g.save(os.path.join(d, "m.yaml"))
        g2 = OccupancyGridMap.load(os.path.join(d, "m.yaml"))
    assert np.array_equal(g2.occ, occ)


def _write(path, header, payload):
    with open(path, "wb") as f:
        f.write(header + payload)


def test_load_uses_header_maxval():
    with tempfile.TemporaryDirectory() as d:
        y = os.path.join(d, "m.yaml")
        with open(y, "w") as f:
            f.write("image: m.pgm\nresolution: 0.1\norigin: [0, 0, 0]\nnegate: 0\n"
                    "occupied_thresh: 0.65\nfree_thresh: 0.196\n")
        # 16-bit file whose pixels are all small: every cell is occupied
        _write(os.path.join(d, "m.pgm"), b"P5\n3 1\n65535\n", np.array([0, 200, 250], ">u2").tobytes())
        assert OccupancyGridMap.load(y).occ.tolist() == [[100, 100, 100]]
        arr, maxval = read_pgm(os.path.join(d, "m.pgm"), return_maxval=True)
        assert maxval == 65535 and arr.tolist() == [[0, 200, 250]]
        # 8-bit file with maxval 100: 100 = free, 50 = unknown, 0 = occupied
        _write(os.path.join(d, "m.pgm"), b"P2\n3 1\n100\n", b"100 50 0\n")
        assert OccupancyGridMap.load(y).occ.tolist() == [[0, -1, 100]]


@pytest.mark.parametrize("content", [b"", b"P5", b"P5\n3", b"P5\n# comment without end", b"P5\n3 2\n"])
def test_truncated_pgm_raises_instead_of_hanging(content):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.pgm")
        _write(p, content, b"")
        result = {}

        def run():
            try:
                read_pgm(p)
                result["ok"] = True
            except ValueError as e:
                result["err"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(5.0)
        assert not t.is_alive(), "read_pgm hangs on a truncated file"
        assert "err" in result


# ------------------------------------------------------------- smoothing clearance
def _min_shortcut_clearance(g, path):
    d = g.distance_field()
    worst = math.inf
    for a, b in zip(path[:-1], path[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg <= 1.5 * g.resolution:            # A* neighbour step, not a shortcut
            continue
        for t in np.linspace(0.0, 1.0, int(seg / (0.25 * g.resolution)) + 2):
            ix, iy = g.world_to_grid(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            worst = min(worst, float(d[iy, ix]))
    return worst


def test_plan_smoothing_keeps_robot_radius():
    occ = np.zeros((80, 80), np.int8)
    occ[:, 0] = occ[:, -1] = occ[0, :] = occ[-1, :] = 100
    occ[0:50, 30:50] = 100                      # block whose corner the path wraps around
    g = OccupancyGridMap(occ, 0.1, 0.0, 0.0)
    p = plan(g, (1.5, 1.5), (7.0, 1.5), robot_radius=0.30)
    assert p is not None and _min_shortcut_clearance(g, p) >= 0.30 - 1e-9
    # the old 0.9 x radius smoothing cut the corner (~0.22 m from the block)
    p09 = plan(g, (1.5, 1.5), (7.0, 1.5), robot_radius=0.30, smooth_radius_scale=0.9)
    assert _min_shortcut_clearance(g, p09) < 0.27


# ------------------------------------------------------------------- pure pursuit
def test_pure_pursuit_sparse_path_drives_forward():
    pp = PurePursuit()
    pp.set_path([(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)])
    vx, vy, wz, done = pp.compute(0.7, 0.0, 0.0)
    assert not done and vx > 0.3 and abs(wz) < 1e-6 and abs(vy) < 1e-6


def test_pure_pursuit_follows_sparse_plan_without_spinning():
    pp = PurePursuit()
    pp.set_path([(0.0, 0.0), (1.5, 0.0), (3.0, 1.5), (4.5, 1.5)], goal_yaw=None)
    x = y = yaw = 0.0
    spins = 0
    for _ in range(3000):
        vx, vy, wz, done = pp.compute(x, y, yaw)
        if done:
            break
        spins += vx == 0.0 and abs(wz) > 0.0
        dt = 0.05
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        yaw += wz * dt
    assert done and math.hypot(x - 4.5, y - 1.5) < 0.3
    assert spins == 0


def test_pure_pursuit_dense_path_unchanged_and_clear_resets():
    dense = [(0.1 * k, 0.0) for k in range(31)]
    pp = PurePursuit()
    pp.set_path(dense)
    assert np.array_equal(pp.path, np.asarray(dense, float))
    pp.compute(1.0, 0.0, 0.0)
    assert pp._idx > 0
    pp.clear()
    assert pp._idx == 0 and not pp.active


def test_pure_pursuit_non_finite_pose_stands_still():
    pp = PurePursuit()
    pp.set_path([(0.0, 0.0), (3.0, 0.0)])
    assert pp.compute(NAN, 0.0, 0.0) == (0.0, 0.0, 0.0, False)
    assert pp.compute(0.0, 0.0, INF) == (0.0, 0.0, 0.0, False)
    assert pp.active


# ------------------------------------------------------------------------ QoS
def _fake_rclpy(monkeypatch, full=True):
    qos = types.ModuleType("rclpy.qos")

    class QoSProfile:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    qos.qos_profile_sensor_data = QoSProfile(depth=5, reliability="best_effort")
    if full:
        qos.QoSProfile = QoSProfile
        qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT="best_effort", RELIABLE="reliable")
        qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE="volatile", TRANSIENT_LOCAL="transient_local")
        qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="keep_last")
    rclpy = types.ModuleType("rclpy")
    rclpy.qos = qos
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos)
    return qos


def test_sensor_qos_honours_depth(monkeypatch):
    qos = _fake_rclpy(monkeypatch)
    a, b = ros_helpers.sensor_qos(), ros_helpers.sensor_qos(20)
    assert (a.depth, a.reliability, a.durability, a.history) == (5, "best_effort", "volatile", "keep_last")
    assert b.depth == 20 and b.reliability == "best_effort"
    assert a is not ros_helpers.sensor_qos() and a is not qos.qos_profile_sensor_data
    _fake_rclpy(monkeypatch, full=False)        # minimal stand-in: falls back to the stock profile
    assert ros_helpers.sensor_qos().reliability == "best_effort"


# ---------------------------------------------------------------------- floor_z
def test_floor_z_stairs_step_function_real_layout():
    lay = Layout.load()
    st = lay.stair_profile()
    assert st is not None and len(st["treads"]) == 4
    up, lo = lay["corridor"]["upper_floor"]["z"], lay["corridor"]["lower_floor"]["z"]
    y = -1.0                                     # stairs side
    assert lay.floor_z(47.2, y) == up
    zs = [lay.floor_z(0.5 * (t[0] + t[1]), y) for t in st["treads"]]
    assert zs == [t[2] for t in st["treads"]]
    assert all(a > b for a, b in zip([up] + zs, zs + [lo]))    # monotone descent
    assert lay.floor_z(48.6, y) == lo and lay.floor_z(50.0, y) == lo
    # ramp side keeps the linear interpolation over the descent
    de = lay["corridor"]["descent"]
    t = (49.0 - de["x_min"]) / (de["x_max"] - de["x_min"])
    assert lay.floor_z(49.0, 1.0) == pytest.approx(up + t * (lo - up))
    assert abs(lay.floor_z(60.0, 0.0) - lo) < 1e-9 and lay.floor_z(0.0, 0.0) == 0.0


def _corridor(**stairs):
    return {"factory": {"floor_z": 0.0},
            "corridor": {"interior": {"x_min": 10.0}, "upper_floor": {"z": 0.0}, "lower_floor": {"z": -0.8},
                         "descent": {"x_min": 20.0, "x_max": 24.0},
                         "stairs": dict(y_min=-1.5, y_max=-0.5, **stairs), "ramp": {"y_min": 0.0, "y_max": 1.5}}}


def test_floor_z_treads_from_layout_and_linear_fallback():
    treads = [{"x_min": 20.5, "x_max": 21.0, "z": -0.2}, {"x0": 21.0, "x1": 21.5, "z": -0.4},
              {"x_min": 21.5, "x_max": 22.0, "z": -0.6}]      # layout keys; geometry keys accepted too
    lay = Layout(_corridor(treads=treads))
    assert lay.floor_z(20.2, -1.0) == 0.0
    assert lay.floor_z(20.7, -1.0) == -0.2
    assert lay.floor_z(21.0, -1.0) == -0.2        # shared edge: the higher tread
    assert lay.floor_z(21.7, -0.3) == -0.6        # between the bands, nearer the stairs
    assert lay.floor_z(22.5, -1.0) == -0.8
    assert lay.floor_z(22.0, 0.5) == pytest.approx(-0.4)     # ramp: linear
    legacy = Layout(_corridor())                 # no tread data, no file -> linear everywhere
    assert legacy.stair_profile() is None
    assert legacy.floor_z(22.0, -1.0) == pytest.approx(-0.4)
    assert legacy.floor_z(NAN, -1.0) == 0.0


def test_floor_z_treads_from_generated_world_geometry(tmp_path):
    import yaml
    data = _corridor()                           # layout without tread data
    (tmp_path / "generated").mkdir()
    geo = {"architecture": {"corridor": {
        "stairs": {"y_min": -1.6, "y_max": -0.4,
                   "treads": [{"z": -0.3, "x0": 20.6, "x1": 21.2}, {"z": -0.55, "x0": 21.15, "x1": 21.8}]},
        "ramp": {"y0": -0.4, "y1": 1.6, "x0": 20.0, "x1": 24.0}}}}
    (tmp_path / "generated" / "world_geometry.yaml").write_text(yaml.safe_dump(geo), encoding="utf-8")
    (tmp_path / "world_layout.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    lay = Layout.load(str(tmp_path / "world_layout.yaml"))
    st = lay.stair_profile()
    assert st["stairs_y"] == (-1.6, -0.4) and st["ramp_y"] == (-0.4, 1.6)
    assert lay.floor_z(20.3, -1.0) == 0.0
    assert lay.floor_z(21.17, -1.0) == -0.3      # overlap (tread nosing): the higher tread
    assert lay.floor_z(21.5, -1.0) == -0.55
    assert lay.floor_z(21.9, -1.0) == -0.8
    assert lay.floor_z(22.0, 1.0) == pytest.approx(-0.4)
    # a broken geometry file never breaks floor_z: linear model
    (tmp_path / "generated" / "world_geometry.yaml").write_text("architecture: [unclosed", encoding="utf-8")
    assert Layout.load(str(tmp_path / "world_layout.yaml")).floor_z(22.0, -1.0) == pytest.approx(-0.4)
