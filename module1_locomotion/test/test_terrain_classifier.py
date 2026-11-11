"""Unit tests: terrain classification and gait hysteresis."""
import math
import os

import numpy as np
import pytest
import yaml

from module1_locomotion.elevation_map import ElevationMap
from module1_locomotion.terrain_classifier import (ClassifierParams, TerrainClassifier, classify,
                                                   decide, detect_edges, height_profile, split_bumps)

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE = (0.0, 0.0, 0.34, 0.0, 0.0, 0.0)
KEYS = ("roughness", "max_step", "slope_deg", "class", "recommended_gait", "ahead_clear")


def grid_map(f, center=(0.0, 0.0), noise=0.0, seed=0):
    m = ElevationMap(center=center)
    X, Y = m.grid_xy()
    z = f(X, Y)
    if noise:
        z = z + np.random.default_rng(seed).normal(0.0, noise, z.shape)
    return ElevationMap.from_elevation(z, m.origin, m.resolution)


def rotated(f, yaw, cx=0.0, cy=0.0):
    """Height function f defined in a frame rotated by yaw about (cx, cy)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return lambda X, Y: f((X - cx) * c + (Y - cy) * s, -(X - cx) * s + (Y - cy) * c)


SCENES = {
    "flat": (lambda X, Y: 0.0 * X, "flat", "trot", True),
    "stairs": (lambda X, Y: np.clip(np.floor((X - 0.7) / 0.3) + 1, 0, 5) * 0.17, "stairs", "crawl", True),
    "stairs_down": (lambda X, Y: -np.clip(np.floor((X - 0.7) / 0.3) + 1, 0, 5) * 0.17, "stairs", "crawl", True),
    "ramp16": (lambda X, Y: np.clip(X - 0.5, 0, None) * math.tan(math.radians(16)), "slope", "crawl", True),
    "ramp8": (lambda X, Y: X * math.tan(math.radians(8)), "slope", "trot", True),
    "pipe": (lambda X, Y: np.where(np.abs(X - 1.0) < 0.05, 0.10, 0.0), "rough", "crawl", True),
    "wall": (lambda X, Y: np.where((X > 1.0) & (X < 1.4) & (np.abs(Y) < 0.5), 0.6, 0.0), "obstacle", "crawl", False),
    "platform": (lambda X, Y: np.where(X > 0.9, 0.10, 0.0), "stairs", "crawl", True),
}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_scene_classes(name):
    f, cls, gait, clear = SCENES[name]
    info = classify(grid_map(f, noise=0.002), POSE)
    for k in KEYS:
        assert k in info
    assert info["class"] == cls, info
    assert info["recommended_gait"] == gait, info
    assert info["ahead_clear"] is clear


@pytest.mark.parametrize("yaw", [0.0, 1.2, -2.5, math.pi])
def test_heading_frame_and_slope_sign(yaw):
    up = lambda X, Y: X * math.tan(math.radians(10))
    m = grid_map(rotated(up, yaw, 0.5, -0.4))
    pose = (0.5, -0.4, 0.34, 0.0, 0.0, yaw)
    info = classify(m, pose)
    assert info["class"] == "slope" and info["slope_deg"] == pytest.approx(10.0, abs=0.3)
    back = (0.5, -0.4, 0.34, 0.0, 0.0, yaw + math.pi)          # facing downhill
    assert classify(m, back)["slope_deg"] == pytest.approx(-10.0, abs=0.3)
    # stairs seen from a rotated robot
    st = rotated(SCENES["stairs"][0], yaw, 0.5, -0.4)
    assert classify(grid_map(st), pose)["class"] == "stairs"


def test_rubble_is_rough():
    rng = np.random.default_rng(7)
    H = rng.normal(0.0, 0.002, (150, 150))
    for _ in range(120):
        r, c = rng.integers(0, 148, 2)
        H[r:r + 3, c:c + 3] = rng.uniform(0.02, 0.07)
    m = ElevationMap.from_elevation(H, (-3.0, -3.0), 0.04)
    info = classify(m, POSE)
    assert info["class"] == "rough" and info["recommended_gait"] == "crawl"
    assert info["roughness"] > 0.012 and info["ahead_clear"]


def test_unknown_when_corridor_not_observed():
    e = np.full((150, 150), np.nan)
    e[:, :70] = 0.0                                    # only behind the robot is known
    m = ElevationMap.from_elevation(e, (-3.0, -3.0), 0.04, hole_fill_radius=0)
    info = classify(m, POSE)
    assert info["class"] == "unknown" and info["recommended_gait"] == "crawl"
    assert info["ahead_clear"] is False and info["roughness"] is None
    # corridor outside the map
    assert classify(grid_map(SCENES["flat"][0]), (10.0, 0.0, 0.3, 0, 0, 0))["class"] == "unknown"


def test_profile_and_edge_helpers():
    x = np.repeat(np.arange(0.31, 1.5, 0.04), 5)
    z = np.where(x > 0.9, 0.17, 0.0)
    prof = height_profile(x, z, 0.3, 1.5, 0.06)
    edges = detect_edges(prof, 0.3, 0.06, 0.02, 4)
    assert len(edges) == 1 and edges[0][1] == pytest.approx(0.17) and abs(edges[0][0] - 0.9) < 0.07
    ramp = np.arange(20) * 0.06 * math.tan(math.radians(25))   # steep ramp: no edges after detrending
    assert detect_edges(ramp, 0.3, 0.06, 0.02, 4) == []
    rem, bumps = split_bumps([(0.9, 0.1), (1.0, -0.1), (1.3, 0.17)], 0.3)
    assert bumps == 1 and rem == [(1.3, 0.17)]
    gap = np.array([0.0, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, 0.1])
    x2 = np.repeat(0.3 + 0.06 * (np.arange(8) + 0.5), 3)
    p2 = height_profile(x2, np.repeat(gap, 3), 0.3, 0.78, 0.06)
    assert np.isfinite(p2[1]) and np.isnan(p2[4])        # short gap filled, long gap kept


def test_gait_hysteresis_no_flicker():
    p = ClassifierParams(trot_hold_time=1.0, crawl_confirm_count=2)
    tc = TerrainClassifier(p)
    flat = grid_map(SCENES["flat"][0], noise=0.001)
    rough = grid_map(SCENES["pipe"][0], noise=0.001)
    t = 0.0
    assert tc.update(flat, POSE, t)["recommended_gait"] == "trot"
    # one rough frame is not enough (confirmation), two are
    t += 0.2
    out = tc.update(rough, POSE, t)
    assert out["recommended_gait"] == "trot" and out["raw_gait"] == "crawl"
    t += 0.2
    assert tc.update(rough, POSE, t)["recommended_gait"] == "crawl"
    # alternating terrain must not toggle the gait
    seq = []
    for k in range(10):
        t += 0.2
        seq.append(tc.update(flat if k % 2 else rough, POSE, t)["recommended_gait"])
    assert set(seq) == {"crawl"}
    # flat for less than the hold time -> still crawl, longer -> trot
    gaits = []
    for k in range(8):
        t += 0.2
        gaits.append(tc.update(flat, POSE, t)["recommended_gait"])
    assert gaits[0] == "crawl" and gaits[3] == "crawl" and gaits[-1] == "trot"


def _metrics(roughness=0.0, slope=0.0, edges=(), blocked=0, known=1.0):
    nan = float("nan")
    return {"known_fraction": known, "n_known": 400, "roughness": roughness, "max_step": 0.0,
            "slope_deg": slope, "incline_deg": slope, "blocked_cells": blocked, "edges": list(edges),
            "height_change": nan}


def test_decision_thresholds_and_hysteresis_band():
    p = ClassifierParams()
    border = _metrics(roughness=0.010)                 # between 0.75 * 0.012 and 0.012
    assert decide(border, p)[:2] == ("flat", "trot")
    assert decide(border, p, p.hysteresis_ratio)[:2] == ("rough", "crawl")
    assert decide(_metrics(slope=10.0), p)[:2] == ("slope", "trot")
    assert decide(_metrics(slope=10.0), p, p.hysteresis_ratio)[:2] == ("slope", "crawl")
    assert decide(_metrics(slope=-13.0), p)[:2] == ("slope", "crawl")
    assert decide(_metrics(edges=[(0.8, 0.07)]), p)[0] == "flat"          # below min rise
    assert decide(_metrics(edges=[(0.8, 0.07)]), p, 0.75)[0] == "stairs"  # strict
    assert decide(_metrics(edges=[(0.8, 0.30)]), p) == ("obstacle", "crawl", False)
    assert decide(_metrics(blocked=5), p) == ("obstacle", "crawl", False)
    assert decide(_metrics(known=0.1), p) == ("unknown", "crawl", False)
    # poor coverage (occluded treads / area behind an obstacle) does not hide positive evidence
    assert decide(_metrics(known=0.2, blocked=5), p) == ("obstacle", "crawl", False)
    assert decide(_metrics(known=0.2, edges=[(0.8, 0.30, False, False)]), p)[0] == "obstacle"
    assert decide(_metrics(known=0.2, edges=[(0.7, 0.17, True, False)]), p) == ("stairs", "crawl", True)
    # ... but weaker cues are not trusted on a poorly covered corridor
    assert decide(_metrics(known=0.2, roughness=0.03), p) == ("unknown", "crawl", False)
    assert decide(_metrics(known=0.2, edges=[(0.7, 0.17, False, False)]), p)[0] == "unknown"
    assert decide(_metrics(known=0.2, slope=14.0), p)[0] == "unknown"


def test_stairs_close_up_with_occluded_treads_is_not_unknown():
    """0.3 m before the first riser the upper treads (above the camera) are
    occluded: the corridor coverage drops below min_known_fraction, but the
    confirmed first step keeps the class at 'stairs'."""
    f = SCENES["stairs"][0]                                # first riser 0.7 m ahead
    m = ElevationMap(center=(0.0, 0.0))
    X, Y = m.grid_xy()
    z = f(X, Y)
    z[(X < 0.59) | (X > 0.90)] = np.nan                    # only the first riser + ~0.2 m of its tread
    emap = ElevationMap.from_elevation(z, m.origin, m.resolution, hole_fill_radius=0)
    info = classify(emap, POSE)
    assert info["known_fraction"] < ClassifierParams().min_known_fraction
    assert info["class"] == "stairs" and info["recommended_gait"] == "crawl" and info["ahead_clear"]
    # the same coverage on flat ground stays 'unknown'
    flat = ElevationMap.from_elevation(np.where(np.isfinite(z), 0.0, np.nan), m.origin, m.resolution,
                                       hole_fill_radius=0)
    assert classify(flat, POSE)["class"] == "unknown"


def test_classifier_state_machine_uses_strict_thresholds(monkeypatch):
    import module1_locomotion.terrain_classifier as tcmod
    script = []
    monkeypatch.setattr(tcmod, "compute_metrics", lambda emap, pose, p: script.pop(0))
    tc = TerrainClassifier(ClassifierParams(trot_hold_time=0.5, crawl_confirm_count=1))
    script.append(_metrics(roughness=0.03))
    assert tc.update(None, POSE, 0.0)["recommended_gait"] == "crawl"
    # borderline roughness: raw decision says trot, the strict exit test keeps crawl forever
    for k in range(10):
        script.append(_metrics(roughness=0.010))
        out = tc.update(None, POSE, 0.2 * (k + 1))
        assert out["raw_gait"] == "trot" and out["recommended_gait"] == "crawl"
    # clearly flat: back to trot after the hold time
    gaits = []
    for k in range(5):
        script.append(_metrics(roughness=0.002))
        gaits.append(tc.update(None, POSE, 3.0 + 0.2 * k)["recommended_gait"])
    assert gaits[:2] == ["crawl", "crawl"] and gaits[-1] == "trot"
    tc.reset()
    assert tc.gait == "trot"
    # simulation reset while waiting to return to trot: the clock jumps back,
    # the hold timer restarts instead of blocking forever
    script.append(_metrics(roughness=0.03))
    assert tc.update(None, POSE, 50.0)["recommended_gait"] == "crawl"
    script.append(_metrics(roughness=0.002))
    assert tc.update(None, POSE, 50.2)["recommended_gait"] == "crawl"
    gaits = []
    for k in range(5):                                   # sim restarted at t = 0
        script.append(_metrics(roughness=0.002))
        gaits.append(tc.update(None, POSE, 0.2 * k)["recommended_gait"])
    assert gaits[0] == "crawl" and gaits[-1] == "trot"


def test_yaml_classifier_section_matches_params():
    with open(os.path.join(PKG_DIR, "config", "terrain.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["terrain_mapper"]["ros__parameters"]["classifier"]
    d = ClassifierParams()
    for k, v in cfg.items():
        assert hasattr(d, k), k
        assert type(v) is type(getattr(d, k)), k
    assert ClassifierParams.from_dict(cfg) == d
    assert ClassifierParams.from_dict({"corridor_far": 2, "edge_max_bins": 3.0}).corridor_far == 2.0


# --------------------------------------------------------------- friction prior (slippery plates)
def _plate_map(mu=0.15, rect=(1.0, -1.2, 5.0, 1.2)):
    from module1_locomotion.friction_map import FrictionMap
    return FrictionMap.from_layout({"obstacles": [{"id": "F-OBS9", "type": "plate", "rect": list(rect), "mu": mu}]})


def test_slippery_plate_ahead_and_under_the_robot():
    """A plate is flat: without the prior it is 'flat'; with it 'slippery' as soon as the
    look-ahead corridor (1 m) reaches it, with the friction fields in /terrain/info."""
    flat = grid_map(lambda X, Y: 0.0 * X, noise=0.002)
    far = (-0.5, 0.0, 0.34, 0.0, 0.0, 0.0)                   # plate 1.5 m ahead: beyond the 1 m check
    near = (0.3, 0.0, 0.34, 0.0, 0.0, 0.0)                   # 0.7 m ahead
    on = (3.0, 0.0, 0.34, 0.0, 0.0, 0.0)
    fm = _plate_map()
    info = classify(flat, far, friction_map=fm)
    assert info["class"] == "flat" and info["slippery"] is False and info["friction_mu"] is None
    assert info["friction_zones"] == []
    info = classify(flat, near, friction_map=fm)
    assert info["class"] == "slippery" and info["slippery"] is True and info["recommended_gait"] == "trot"
    assert info["friction_mu"] == pytest.approx(0.15) and info["friction_zones"] == ["F-OBS9"]
    assert info["friction_distance"] == pytest.approx(0.7) and info["ahead_clear"] is True
    info = classify(grid_map(lambda X, Y: 0.0 * X, center=(3.0, 0.0), noise=0.002), on, friction_map=fm)
    assert info["class"] == "slippery" and info["friction_distance"] == 0.0
    # mu 0.4 plate is not slippery (slippery_mu 0.35); prior disabled -> ignored
    assert classify(flat, near, friction_map=_plate_map(mu=0.4))["slippery"] is False
    off = ClassifierParams(use_friction_prior=False)
    assert classify(flat, near, off, friction_map=fm)["class"] == "flat"
    assert classify(flat, near)["slippery"] is False         # no map at all: fields still present


def test_geometry_keeps_its_class_on_a_slippery_zone():
    """rough / stairs / obstacle keep their class and crawl; the flag is still reported."""
    fm = _plate_map(rect=(0.0, -1.2, 5.0, 1.2))
    pose = (0.0, 0.0, 0.34, 0.0, 0.0, 0.0)
    pipe = classify(grid_map(SCENES["pipe"][0], noise=0.002), pose, friction_map=fm)
    assert pipe["class"] == "rough" and pipe["recommended_gait"] == "crawl" and pipe["slippery"] is True
    wall = classify(grid_map(SCENES["wall"][0], noise=0.002), pose, friction_map=fm)
    assert wall["class"] == "obstacle" and wall["ahead_clear"] is False and wall["slippery"] is True
    unk = classify(ElevationMap(), pose, friction_map=fm)    # nothing observed: the layout knows it is flat
    assert unk["class"] == "slippery" and unk["recommended_gait"] == "trot"
    crawl = ClassifierParams(slippery_gait="crawl")
    assert classify(grid_map(SCENES["flat"][0]), pose, crawl, friction_map=fm)["recommended_gait"] == "crawl"


def test_slippery_hysteresis_and_gait_votes():
    flat = grid_map(lambda X, Y: 0.0 * X, noise=0.002)
    tc = TerrainClassifier(friction_map=_plate_map())
    on = tc.update(flat, (0.3, 0.0, 0.34, 0.0, 0.0, 0.0), 10.0)
    assert on["slippery"] is True and on["class"] == "slippery"
    # turned away from the plate: held for slippery_hold_time (1 s), then released
    away = (0.3, 0.0, 0.34, 0.0, 0.0, math.pi)
    held = tc.update(flat, away, 10.6)
    assert held["slippery"] is True and held["friction_mu"] == pytest.approx(0.15)
    assert held["friction_zones"] == ["F-OBS9"]
    done = tc.update(flat, away, 11.2)
    assert done["slippery"] is False and done["class"] == "flat" and done["friction_mu"] is None
    # simulation reset (clock back): no stale hold
    tc.update(flat, (0.3, 0.0, 0.34, 0.0, 0.0, 0.0), 20.0)
    assert tc.update(flat, away, 1.0)["slippery"] is False
    # slippery_gait crawl goes through the crawl confirmation of the gait hysteresis
    tc2 = TerrainClassifier(ClassifierParams(slippery_gait="crawl"), friction_map=_plate_map())
    g = [tc2.update(flat, (0.3, 0.0, 0.34, 0.0, 0.0, 0.0), 0.2 * k)["recommended_gait"] for k in range(3)]
    assert g == ["trot", "crawl", "crawl"]
    tc2.set_friction_map(None)
    assert tc2.update(flat, (0.3, 0.0, 0.34, 0.0, 0.0, 0.0), 0.8)["slippery"] is True    # still held
    assert tc2.update(flat, (0.3, 0.0, 0.34, 0.0, 0.0, 0.0), 2.0)["slippery"] is False


def test_friction_prior_with_scripted_metrics(monkeypatch):
    """The friction check needs only the pose (works with emap None / scripted metrics)."""
    from module1_locomotion import terrain_classifier as tcmod
    monkeypatch.setattr(tcmod, "compute_metrics", lambda emap, pose, p: _metrics(roughness=0.002))
    tc = TerrainClassifier(friction_map=_plate_map())
    out = tc.update(None, (0.5, 0.0, 0.0), 1.0)                # (x, y, yaw) pose form
    assert out["class"] == "slippery" and out["raw_gait"] == "trot"
