"""Unit tests of module2_inspection.thermal_fusion (ROS free)."""
import json
import math

import numpy as np
import pytest

from dusan_common.camera import PinholeCamera, world_to_camera_link
from dusan_common.layout import Layout
from dusan_common.ros_helpers import to_json
from module2_inspection.synthetic_thermal import SyntheticThermalCamera, full_pose, render_visible_proxy
from module2_inspection.thermal_fusion import (
    R_LINK_OPT, CameraRig, FusionParams, HotspotRef, StampedBuffer, TemporalConfirmer, ThermalFusion, apply_homography,
    colorize_temperature, kelvin_to_l16, match_ray_to_hotspots, point_ray_distance, raw_to_kelvin,
    refine_homography_ecc, segment_hotspots, transform_bbox)

LAYOUT = Layout.load()
POSE_H02 = full_pose(LAYOUT, 5.0, 4.4, math.pi / 2)        # 2 m south of flange H02, facing north
POSE_H01 = full_pose(LAYOUT, 2.0, 4.6, math.pi / 2)        # in front of H01 (318 K, below warning)


@pytest.fixture(scope="module")
def thermal_cam():
    return SyntheticThermalCamera(LAYOUT, seed=3)


@pytest.fixture(scope="module")
def frame_h02(thermal_cam):
    K = thermal_cam.render_kelvin(POSE_H02)
    rgb, _ = render_visible_proxy(thermal_cam.scene, POSE_H02)
    return K, rgb


# --------------------------------------------------------------------------- radiometry
@pytest.mark.parametrize("enc", ["mono16", "16UC1", "L16", "l16"])
def test_l16_to_kelvin_all_encodings(enc):
    raw = np.array([[29815, 37215], [0, 65535]], dtype=np.uint16)
    k = raw_to_kelvin(raw, enc, 0.01)
    assert k.dtype == np.float32
    assert abs(k[0, 0] - 298.15) < 1e-3 and abs(k[0, 1] - 372.15) < 1e-3
    assert np.isnan(k[1, 0])                         # zero count = invalid
    assert abs(k[1, 1] - 655.35) < 1e-2


def test_l8_float_and_roundtrip():
    k8 = raw_to_kelvin(np.array([[0, 255]], np.uint8), "L8", l8_range=(250.0, 350.0))
    assert np.allclose(k8, [[250.0, 350.0]], atol=1e-4)
    kf = raw_to_kelvin(np.array([[300.5, -1.0]], np.float32), "32FC1")
    assert kf[0, 0] == pytest.approx(300.5) and np.isnan(kf[0, 1])
    T = np.array([[298.15, 333.157, np.nan]])
    back = raw_to_kelvin(kelvin_to_l16(T), "mono16")
    assert np.allclose(back[0, :2], T[0, :2], atol=0.006) and np.isnan(back[0, 2])
    with pytest.raises(ValueError):
        raw_to_kelvin(np.zeros((2, 2), np.int32), "rgb8")


# --------------------------------------------------------------------------- homography
def _project(cam: PinholeCamera, T_world_link: np.ndarray, pts: np.ndarray) -> np.ndarray:
    uv, _ = cam.project_link(world_to_camera_link(T_world_link, pts))
    return uv


def test_rig_relative_pose():
    rig = CameraRig.from_robot_model()
    R, t = rig.relative_optical()
    assert np.allclose(R, np.eye(3), atol=1e-12)
    assert np.allclose(t, [0.05, 0.0, 0.0], atol=1e-12)    # thermal 5 cm to the right (+x optical)
    assert np.allclose(rig.epipolar_direction(), [1.0, 0.0])
    # optical <-> link convention matches dusan_common.camera
    p = np.array([0.3, -0.2, 2.0])
    assert np.allclose(R_LINK_OPT @ p, PinholeCamera.optical_to_link(p))


@pytest.mark.parametrize("mode", ["fronto", "vertical"])
def test_plane_homography_exact_on_plane(mode):
    rig = CameraRig.from_robot_model()
    pose = (1.0, 2.0, 0.34, 0.02, -0.03, 0.7)
    T_th, T_rgb = rig.thermal_world(pose), rig.rgb_world(pose)
    depth = 2.3
    H, n, d = rig.plane_homography(depth, mode, pose)
    # build world points on the plane n.X + d = 0 (thermal optical frame)
    rng = np.random.default_rng(0)
    u = rng.uniform(20, 300, 40)
    v = rng.uniform(20, 220, 40)
    rays_opt = np.linalg.inv(rig.thermal.K) @ np.vstack([u, v, np.ones_like(u)])
    lam = -d / (n @ rays_opt)                       # ray-plane intersection
    X_opt = rays_opt * lam
    X_link = (R_LINK_OPT @ X_opt).T
    X_world = X_link @ T_th[:3, :3].T + T_th[:3, 3]
    uv_th = _project(rig.thermal, T_th, X_world)
    uv_rgb = _project(rig.rgb, T_rgb, X_world)
    assert np.allclose(uv_th, np.c_[u, v], atol=1e-6)
    assert np.max(np.linalg.norm(apply_homography(H, uv_th) - uv_rgb, axis=1)) < 1e-6


def test_parallax_error_off_plane():
    """Off-plane points are displaced horizontally by f_rgb * b * |1/Z - 1/d|."""
    rig = CameraRig.from_robot_model()
    pose = (0.0, 0.0, 0.34, 0.0, 0.0, 0.0)
    T_th, T_rgb = rig.thermal_world(pose), rig.rgb_world(pose)
    H, _, _ = rig.plane_homography(2.0, "fronto", pose)
    for Z in (1.0, 1.5, 4.0, 8.0):
        X_opt = np.array([0.1, -0.05, Z])
        Xw = T_th[:3, :3] @ (R_LINK_OPT @ X_opt) + T_th[:3, 3]
        uv_t = _project(rig.thermal, T_th, Xw[None])
        uv_r = _project(rig.rgb, T_rgb, Xw[None])
        err = apply_homography(H, uv_t)[0] - uv_r[0]
        expected = rig.rgb.fx * 0.05 * abs(1.0 / Z - 1.0 / 2.0)
        assert abs(abs(err[0]) - expected) < 1e-6 * max(1.0, expected)
        assert abs(err[1]) < 1e-9                    # epipolar: purely horizontal


def test_transform_bbox_identity_and_scale():
    assert transform_bbox(np.eye(3), [10, 20, 30, 40]) == [10, 20, 30, 40]
    assert transform_bbox(np.diag([2.0, 2.0, 1.0]), [10, 20, 30, 40]) == [20, 40, 60, 80]


# --------------------------------------------------------------------------- ECC
def _hotspot_registration_error(H, rig, pose, pos):
    uv_t = _project(rig.thermal, rig.thermal_world(pose), np.asarray(pos)[None])
    uv_r = _project(rig.rgb, rig.rgb_world(pose), np.asarray(pos)[None])
    return float(np.linalg.norm(apply_homography(H, uv_t) - uv_r))


def test_ecc_recovers_depth_error(frame_h02):
    K, rgb = frame_h02
    rig = CameraRig.from_robot_model()
    h02 = np.asarray(LAYOUT.hotspots[1]["position"])
    H_bad, _, _ = rig.plane_homography(5.0, "fronto", POSE_H02)      # true depth ~1.8 m
    e0 = _hotspot_registration_error(H_bad, rig, POSE_H02, h02)
    assert e0 > 8.0
    res = refine_homography_ecc(K, rgb, H_bad, epipolar_dir=rig.epipolar_direction())
    assert res.ok, res.reason
    e1 = _hotspot_registration_error(res.H, rig, POSE_H02, h02)
    assert e1 < 0.5 * e0 and e1 < 5.0
    assert abs(res.shift_px[1]) < 1e-9               # epipolar constraint (horizontal baseline)


def test_ecc_rejects_featureless():
    rgb = np.full((600, 800, 3), 120, np.uint8)
    K = np.full((240, 320), 298.15, np.float32)
    res = refine_homography_ecc(K, rgb, np.eye(3))
    assert not res.ok and np.allclose(res.H, np.eye(3))


# --------------------------------------------------------------------------- segmentation
def test_segment_hotspots_stats():
    T = np.full((60, 80), 300.0, np.float32)
    T[10:20, 10:25] = 340.0           # warning blob, 150 px
    T[15, 18] = 360.0                 # peak inside it
    T[40:50, 50:60] = 370.0           # alarm blob, 100 px
    T[5, 70] = 400.0                  # single hot pixel -> below min area
    T[30, 30] = np.nan
    blobs = segment_hotspots(T, 333.15, min_area_px=6)
    assert len(blobs) == 2
    assert blobs[0].max_temp_k == pytest.approx(370.0)       # sorted by max temperature
    assert blobs[0].bbox == (50, 40, 10, 10) and blobs[0].area_px == 100
    b = blobs[1]
    assert b.max_temp_k == pytest.approx(360.0) and b.peak_uv == (18, 15)
    assert b.mean_temp_k == pytest.approx((149 * 340.0 + 360.0) / 150, rel=1e-6)
    assert 10 <= b.centroid_uv[0] <= 25 and 10 <= b.centroid_uv[1] <= 20
    assert segment_hotspots(np.full((10, 10), 300.0, np.float32), 333.15) == []


# --------------------------------------------------------------------------- localisation
def test_point_ray_distance_and_matching():
    o, d = np.zeros(3), np.array([1.0, 0.0, 0.0])
    dist, t = point_ray_distance(np.array([3.0, 0.4, 0.0]), o, d)
    assert dist == pytest.approx(0.4) and t == pytest.approx(3.0)
    dist, t = point_ray_distance(np.array([-1.0, 0.0, 0.0]), o, d)
    assert t < 0
    refs = [HotspotRef("A", "a", np.array([3.0, 0.3, 0.0]), 0.1), HotspotRef("B", "b", np.array([3.0, 2.0, 0.0]), 0.1),
            HotspotRef("C", "c", np.array([-3.0, 0.0, 0.0]), 0.1)]
    ref, dist, rng = match_ray_to_hotspots(o, d, refs, tol_m=0.3)
    assert ref.id == "A" and rng == pytest.approx(3.0)
    ref, _, _ = match_ray_to_hotspots(o, np.array([0.0, 0.0, 1.0]), refs, tol_m=0.3)
    assert ref is None


# --------------------------------------------------------------------------- temporal
def test_temporal_confirmation():
    tc = TemporalConfirmer(confirm_frames=3, max_misses=1, max_gap_s=2.0)
    det = lambda: {"id": "H02", "position": [5.0, 6.4, 1.0], "centroid_px": [100, 100], "max_temp_k": 370.0}
    assert tc.update([det()], 0.0) == []
    assert tc.update([det()], 0.2) == []
    out = tc.update([det()], 0.4)
    assert len(out) == 1 and out[0]["confirmed_frames"] == 3
    tc.update([], 0.6)                                 # one miss is tolerated
    assert len(tc.update([det()], 0.8)) == 1
    tc.update([], 1.0)
    tc.update([], 1.2)                                 # second miss -> track dropped
    assert tc.update([det()], 1.4) == []
    # anonymous blobs associated by world position
    tc2 = TemporalConfirmer(confirm_frames=2)
    a = {"id": None, "position": [1.0, 1.0, 1.0], "centroid_px": [10, 10], "max_temp_k": 340.0}
    b = {"id": None, "position": [1.2, 1.0, 1.0], "centroid_px": [14, 10], "max_temp_k": 341.0}
    tc2.update([a], 0.0)
    assert len(tc2.update([b], 0.2)) == 1
    far = {"id": None, "position": [5.0, 1.0, 1.0], "centroid_px": [200, 10], "max_temp_k": 341.0}
    assert tc2.update([far], 0.4) == []


# --------------------------------------------------------------------------- full pipeline
def test_pipeline_detects_and_localises_h02(thermal_cam):
    fus = ThermalFusion(LAYOUT)
    rgb, _ = render_visible_proxy(thermal_cam.scene, POSE_H02)
    res = None
    for i in range(3):
        K = thermal_cam.render_kelvin(POSE_H02)
        res = fus.process(K, rgb, POSE_H02, stamp=0.2 * i)
        if i < 2:
            assert res.confirmed == []                  # temporal confirmation
    assert len(res.confirmed) >= 1
    h = res.confirmed[0]
    assert h["id"] == "H02" and h["tag"] == "PIPE-J2" and h["status"] == "alarm"
    assert abs(h["max_temp_k"] - 372.15) < 2.0
    assert np.allclose(h["position"], LAYOUT.hotspots[1]["position"])
    assert res.depth_source == "hotspot:H02" and 1.5 < res.plane_depth < 2.2
    assert res.fusion.shape == (600, 800, 3) and res.fusion.dtype == np.uint8
    alerts = fus.alerts_dict(res, 12.5)
    assert alerts["stamp"] == 12.5
    entry = alerts["hotspots"][0]
    for key in ("id", "tag", "max_temp_c", "mean_temp_c", "position", "status", "bbox"):
        assert key in entry
    assert entry["status"] in ("normal", "warning", "alarm") and len(entry["bbox"]) == 4
    assert abs(entry["max_temp_c"] - 99.0) < 2.0
    json.loads(to_json(alerts))                          # serialisable (NaN -> null)


def test_pipeline_without_pose_and_rgb(frame_h02):
    K, _ = frame_h02
    fus = ThermalFusion(LAYOUT, FusionParams(confirm_frames=1))
    res = fus.process(K, None, None, 0.0)
    assert len(res.confirmed) >= 1
    h = res.confirmed[0]
    assert h["id"] is None and h["position"] is None and h["status"] == "alarm"
    assert res.depth_source == "fixed"
    assert res.fusion.shape == (600, 800, 3)            # thermal-only canvas


def test_normal_hotspot_reported_from_roi(thermal_cam):
    fus = ThermalFusion(LAYOUT, FusionParams(confirm_frames=1))
    K = thermal_cam.render_kelvin(POSE_H01)
    res = fus.process(K, None, POSE_H01, 0.0, render=False)
    h01 = [d for d in res.confirmed if d["id"] == "H01"]
    assert len(h01) == 1
    assert h01[0]["source"] == "roi" and h01[0]["status"] == "normal"
    assert abs(h01[0]["max_temp_k"] - 318.15) < 2.0
    assert res.fusion is None


def test_threshold_override_and_colormap():
    fus = ThermalFusion(LAYOUT, FusionParams(warning_k=320.0, alarm_k=330.0))
    assert fus.status(325.0) == "warning" and fus.status(331.0) == "alarm" and fus.status(300.0) == "normal"
    fus2 = ThermalFusion(LAYOUT)
    assert fus2.status(372.15) == LAYOUT.temperature_status(372.15) == "alarm"
    img = colorize_temperature(np.array([[290.0, 400.0, np.nan]], np.float32), 293.15, 393.15)
    assert img.shape == (1, 3, 3) and img[0, 1].sum() > img[0, 0].sum()


# --------------------------------------------------------------------------- time sync
def test_stamped_buffer_nearest_and_interpolation():
    buf = StampedBuffer(maxlen=5)
    assert buf.nearest(1.0, 0.1) is None and buf.interpolate_pose(1.0, 0.1) is None
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        buf.add(t, (t, 2.0 * t, 0.3, 0.0, 0.0, math.pi - 0.05 + t))   # yaw wraps through +-pi
    assert len(buf) == 5 and buf.t[0] == pytest.approx(0.1)          # oldest dropped
    assert buf.nearest(0.33, 0.05)[0] == pytest.approx(0.3)
    assert buf.nearest(0.9, 0.1) is None
    p = buf.interpolate_pose(0.25, 0.1)
    assert p[0] == pytest.approx(0.25) and p[1] == pytest.approx(0.5)
    yaw_expected = math.remainder(math.pi - 0.05 + 0.25, 2 * math.pi)
    assert math.remainder(p[5] - yaw_expected, 2 * math.pi) == pytest.approx(0.0, abs=1e-9)
    assert buf.interpolate_pose(0.55, 0.1)[0] == pytest.approx(0.5)   # newest, no extrapolation
    assert buf.interpolate_pose(0.9, 0.1) is None
    buf.add(0.45, (9.0, 0, 0, 0, 0, 0))                                # out-of-order insert stays sorted
    assert buf.t == sorted(buf.t)
    buf.add(-5.0, (0, 0, 0, 0, 0, 0))                                  # clock reset -> start over
    assert len(buf) == 1


def test_per_detection_box_uses_own_plane(thermal_cam):
    """With a deliberately wrong global plane the matched hot spot's RGB box is
    still placed with the plane through its own layout position."""
    fus = ThermalFusion(LAYOUT, FusionParams(confirm_frames=1, depth_mode="fixed", plane_distance=6.0,
                                             ecc_enabled=False))
    K = thermal_cam.render_kelvin(POSE_H02)
    res = fus.process(K, np.zeros((600, 800, 3), np.uint8), POSE_H02, 0.0, render=False)
    d = [x for x in res.detections if x["id"] == "H02"][0]
    rig = CameraRig.from_robot_model()
    h02 = np.asarray(LAYOUT.hotspots[1]["position"])
    e_global = _hotspot_registration_error(res.H, rig, POSE_H02, h02)
    e_own = _hotspot_registration_error(d["_H"], rig, POSE_H02, h02)
    assert e_global > 10.0 and e_own < 0.5
    uv_true = _project(rig.rgb, rig.rgb_world(POSE_H02), h02[None])[0]
    u, v, w, h = d["bbox_rgb"]
    assert u <= uv_true[0] <= u + w and v - 5 <= uv_true[1] <= v + h + 5


# --------------------------------------------------------------------------- config contract
def _declared_params(node_file, dict_names, namespace):
    """Evaluate the module-level parameter dicts of a node file without importing
    rclpy (ROS 2 rejects YAML values whose type differs from the declared default)."""
    import ast
    import os
    src = open(os.path.join(os.path.dirname(__file__), "..", "module2_inspection", node_file), encoding="utf-8").read()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in dict_names for t in node.targets):
            out.update(eval(compile(ast.Expression(node.value), node_file, "eval"), dict(namespace)))
    return out


def _check_yaml_section(section, declared):
    for key, val in section.items():
        assert key in declared, f"{key} is not declared by the node"
        d = declared[key]
        if isinstance(d, (list, tuple)):
            assert isinstance(val, list) and val and all(type(v) is type(d[0]) for v in val), key
        else:
            assert type(val) is type(d), f"{key}: YAML {type(val).__name__} vs declared {type(d).__name__}"


def test_thermal_yaml_matches_declared_parameters():
    import os
    import yaml
    from dusan_common.robot_model import NOMINAL_BODY_HEIGHT
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config", "thermal.yaml"), encoding="utf-8"))
    ns = {"_FP": FusionParams(), "list": list, "NOMINAL_BODY_HEIGHT": NOMINAL_BODY_HEIGHT}
    fusion = _declared_params("thermal_fusion_node.py", {"NODE_PARAMS", "FUSION_PARAMS"}, ns)
    exposed = {k for k in fusion if k in FusionParams.__dataclass_fields__}
    # every algorithm tunable is a ROS parameter ('resolution' is exposed as thermal_resolution)
    assert set(FusionParams.__dataclass_fields__) - {"resolution"} <= exposed
    assert "thermal_resolution" in fusion
    _check_yaml_section(cfg["thermal_fusion"]["ros__parameters"], fusion)
    synth = _declared_params("synthetic_thermal_node.py", {"PARAMS"}, ns)
    _check_yaml_section(cfg["synthetic_thermal_camera"]["ros__parameters"], synth)


# --------------------------------------------------------------------------- packaging
def test_setup_tree_skips_bytecode(tmp_path, monkeypatch):
    """setup.py installs launch/ etc. with tree(): byte-code caches must not end up in share/."""
    import ast
    import os
    src = open(os.path.join(os.path.dirname(__file__), "..", "setup.py"), encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "tree")
    ns = {"os": os, "package_name": "module2_inspection"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "setup.py", "exec"), ns)
    (tmp_path / "launch" / "__pycache__").mkdir(parents=True)
    (tmp_path / "launch" / "sub").mkdir()
    (tmp_path / "launch" / "inspection.launch.py").write_text("x = 1\n")
    (tmp_path / "launch" / "stale.pyc").write_bytes(b"\0")
    (tmp_path / "launch" / "__pycache__" / "inspection.launch.cpython-312.pyc").write_bytes(b"\0")
    (tmp_path / "launch" / "sub" / "a.yaml").write_text("a: 1\n")
    monkeypatch.chdir(tmp_path)
    out = ns["tree"]("launch")
    assert out == [(os.path.join("share", "module2_inspection", "launch"),
                    [os.path.join("launch", "inspection.launch.py")]),
                   (os.path.join("share", "module2_inspection", "launch", "sub"),
                    [os.path.join("launch", "sub", "a.yaml")])]


def test_own_frame_filter():
    """synthetic_thermal_camera recognises frames of a second publisher on its output topic."""
    import array
    from module2_inspection.synthetic_thermal import OwnFrameFilter
    flt = OwnFrameFilter(history=3)
    rng = np.random.default_rng(0)
    frames = [array.array("B", rng.integers(0, 256, 320 * 240 * 2, dtype=np.uint8).tobytes()) for _ in range(5)]
    for f in frames[:4]:
        flt.add_own(f)
    assert not flt.is_foreign(frames[3]) and not flt.is_foreign(bytes(frames[2]))
    assert not flt.is_foreign(list(frames[1]))                  # plain int list payloads work too
    assert flt.is_foreign(frames[0])                            # dropped from the history
    assert flt.is_foreign(frames[4])                            # never published by us


# --------------------------------------------------------------------------- launch file
def test_inspection_launch_description(monkeypatch):
    """Dry run of launch/inspection.launch.py against minimal stand-ins of the
    launch APIs: executables exist in setup.py, YAML sections match the node
    names (a mismatch silently drops all parameters), launch arguments and the
    thermal_source condition."""
    import importlib.util
    import os
    import re
    import sys
    import types
    import yaml

    pkg_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    class Rec:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs

    class LaunchConfiguration(Rec):
        def perform(self, ctx):
            return ctx[self.args[0]]

    class PythonExpression(Rec):
        def perform(self, ctx):
            return str(eval("".join(p if isinstance(p, str) else p.perform(ctx) for p in self.args[0])))

    class IfCondition(Rec):
        def evaluate(self, ctx):
            return self.args[0].perform(ctx).strip().lower() in ("true", "1")

    stubs = {
        "launch": dict(LaunchDescription=type("LaunchDescription", (Rec,), {})),
        "launch.actions": dict(DeclareLaunchArgument=type("DeclareLaunchArgument", (Rec,), {}),
                               LogInfo=type("LogInfo", (Rec,), {})),
        "launch.conditions": dict(IfCondition=IfCondition),
        "launch.substitutions": dict(LaunchConfiguration=LaunchConfiguration, PythonExpression=PythonExpression),
        "launch_ros": {},
        "launch_ros.actions": dict(Node=type("Node", (Rec,), {})),
        "launch_ros.parameter_descriptions": dict(ParameterValue=type("ParameterValue", (Rec,), {})),
        "ament_index_python": {},
        "ament_index_python.packages": dict(get_package_share_directory=lambda pkg: pkg_dir),
    }
    for name, attrs in stubs.items():
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
    spec = importlib.util.spec_from_file_location("inspection_launch", os.path.join(pkg_dir, "launch", "inspection.launch.py"))
    mod = importlib.util.module_from_spec(spec)
    # no launch/__pycache__ (setup.py installs everything below launch/ into share/)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec.loader.exec_module(mod)
    ld = mod.generate_launch_description()
    entities = ld.args[0]

    args = {e.args[0]: e.kwargs.get("default_value") for e in entities if type(e).__name__ == "DeclareLaunchArgument"}
    assert args == {"use_sim_time": "true", "thermal_source": "gazebo", "map_yaml": "", "layout_file": ""}
    nodes = {e.kwargs["executable"]: e for e in entities if type(e).__name__ == "Node"}
    setup_src = open(os.path.join(pkg_dir, "setup.py"), encoding="utf-8").read()
    entry = set(re.findall(r'"\s*([A-Za-z0-9_]+)\s*=\s*module2_inspection\.[A-Za-z0-9_]+:main"', setup_src))
    assert set(nodes) == {"gauge_reader", "thermal_fusion", "patrol_planner", "synthetic_thermal_camera"} <= entry
    expected_yaml = {"gauge_reader": "gauge_reader.yaml", "thermal_fusion": "thermal.yaml",
                     "patrol_planner": "patrol.yaml", "synthetic_thermal_camera": "thermal.yaml"}
    for exe, node in nodes.items():
        params = node.kwargs["parameters"]
        files = [p for p in params if isinstance(p, str)]
        overrides = [p for p in params if isinstance(p, dict)]
        assert len(overrides) == 1 and params[-1] is overrides[0]          # launch overrides win
        assert overrides[0]["use_sim_time"].kwargs["value_type"] is bool
        assert overrides[0]["layout_file"].kwargs["value_type"] is str
        if exe != "gauge_reader":                                            # owned by another agent
            assert len(files) == 1 and os.path.basename(files[0]) == expected_yaml[exe]
            sections = yaml.safe_load(open(files[0], encoding="utf-8"))
            assert node.kwargs["name"] in sections, f"{files[0]} has no section for node {node.kwargs['name']}"
    assert "map_yaml" in nodes["patrol_planner"].kwargs["parameters"][-1]
    cond = nodes["synthetic_thermal_camera"].kwargs["condition"]
    assert cond.evaluate({"thermal_source": "synthetic"}) and not cond.evaluate({"thermal_source": "gazebo"})
    assert all("condition" not in n.kwargs for e, n in nodes.items() if e != "synthetic_thermal_camera")
