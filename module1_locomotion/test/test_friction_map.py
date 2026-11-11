"""Unit tests: friction prior (friction_map.py) from world_layout.yaml."""
import math
import os

import numpy as np
import pytest
import yaml

from module1_locomotion.friction_map import (DEFAULT_MU, FrictionMap, FrictionZone, convex_quads_overlap,
                                             corridor_corners)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAYOUT = os.path.join(REPO, "simulation", "config", "world_layout.yaml")


def test_plant_layout_plates_are_the_zones():
    """C-OBS4 (corridor, mu 0.2) and F-OBS9 (factory, mu 0.15) - the slippery steel plates."""
    fm = FrictionMap.load(LAYOUT)
    zones = {z.zone_id: z for z in fm.zones}
    assert set(zones) == {"C-OBS4", "F-OBS9"}
    assert zones["C-OBS4"].mu == pytest.approx(0.2)
    assert (zones["C-OBS4"].x_min, zones["C-OBS4"].y_min, zones["C-OBS4"].x_max, zones["C-OBS4"].y_max) == \
        pytest.approx((30.0, -1.2, 32.5, 1.2))
    assert zones["F-OBS9"].mu == pytest.approx(0.15)
    assert fm.mu_at(1.0, -1.5) == pytest.approx(0.15)          # on F-OBS9
    assert fm.mu_at(31.0, 0.0) == pytest.approx(0.2)           # on C-OBS4
    assert fm.mu_at(10.0, 0.0) == DEFAULT_MU                   # ordinary floor
    # the zones are exactly the plates of the layout (single source of truth)
    with open(LAYOUT, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    plates = [o for o in data["obstacles"] if o["type"] == "plate"]
    assert len(plates) == len(fm)


def test_friction_zones_list_threshold_and_validation():
    data = {"obstacles": [{"id": "P1", "type": "plate", "rect": [2.0, 1.0, 0.0, -1.0], "mu": 0.15},
                          {"id": "P2", "type": "plate", "rect": [5, 5, 6, 6], "mu": 0.9},      # grippy: dropped
                          {"id": "PIPE", "type": "pipe", "center": [0, 0]}],
            "friction_zones": [{"id": "OIL", "rect": [10, 0, 11, 1], "mu": 0.1}, {"rect": [20, 0, 21, 1], "mu": 0.3}]}
    fm = FrictionMap.from_layout(data, max_mu=0.6)
    assert [z.zone_id for z in fm.zones] == ["P1", "OIL", "zone1"]
    p1 = fm.zones[0]
    assert (p1.x_min, p1.y_min, p1.x_max, p1.y_max) == (0.0, -1.0, 2.0, 1.0)     # corners sorted
    assert fm.mu_at(10.5, 0.5) == pytest.approx(0.1)
    assert len(FrictionMap.from_layout(data, max_mu=0.2)) == 2
    assert len(FrictionMap.from_layout({})) == 0 and not FrictionMap.from_layout(None)
    with pytest.raises(ValueError):
        FrictionZone.from_rect("bad", [0, 0, 1], 0.2)
    with pytest.raises(ValueError):
        FrictionZone.from_rect("bad", [0, 0, 1, 1], 0.0)
    with pytest.raises(ValueError):
        FrictionMap.from_zone_dicts([{"rect": [0, 0, 1, 1]}])
    with pytest.raises(FileNotFoundError):
        FrictionMap.load(os.path.join(REPO, "no_such_layout.yaml"))
    # overlapping zones: the lowest mu wins
    both = FrictionMap.from_zone_dicts([{"id": "a", "rect": [0, 0, 2, 2], "mu": 0.3},
                                        {"id": "b", "rect": [1, 1, 3, 3], "mu": 0.12}])
    assert both.mu_at(1.5, 1.5) == pytest.approx(0.12) and both.mu_at(0.5, 0.5) == pytest.approx(0.3)


def test_separating_axis_overlap():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert convex_quads_overlap(sq, sq + 0.5)
    assert not convex_quads_overlap(sq, sq + [1.5, 0.0])
    assert convex_quads_overlap(sq, sq + [1.0, 0.0])                     # touching edge
    # a 45 deg diamond next to the square corner: the bounding boxes overlap, the shapes do not
    c = np.array([1.5, 1.5])
    diamond = c + 0.6 * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    assert not convex_quads_overlap(sq, diamond)
    assert convex_quads_overlap(sq, diamond - 0.3)


def test_corridor_query_in_the_heading_frame():
    fm = FrictionMap.from_zone_dicts([{"id": "plate", "rect": [1.0, -1.2, 5.0, 1.2], "mu": 0.15}])
    # 0.5 m before the plate, looking at it: inside a 1 m look-ahead, distance 0.5 m
    mu, ids, dist = fm.query(0.5, 0.0, 0.0, far=1.0, half_width=0.3, rear=0.35)
    assert mu == pytest.approx(0.15) and ids == ["plate"] and dist == pytest.approx(0.5)
    # same place, looking away (yaw pi): only the rear footprint could reach it -> no overlap
    mu, ids, dist = fm.query(0.5, 0.0, math.pi, far=1.0, half_width=0.3, rear=0.35)
    assert ids == [] and mu == DEFAULT_MU and dist == math.inf
    # walking backwards off the plate: the rear footprint still overlaps
    mu, ids, dist = fm.query(5.2, 0.0, 0.0, far=1.0, half_width=0.3, rear=0.35)
    assert ids == ["plate"] and dist == 0.0
    # beside the plate, heading along it: the corridor half width decides
    assert fm.query(3.0, 1.6, 0.0, far=1.0, half_width=0.3)[1] == []
    assert fm.query(3.0, 1.45, 0.0, far=1.0, half_width=0.3)[1] == ["plate"]
    # diagonal approach (yaw 45 deg) from the corner
    y = math.radians(45.0)
    assert fm.query(0.3, -1.9, y, far=1.0, half_width=0.3)[1] == ["plate"]
    assert fm.query(0.3, -1.9, y, far=0.4, half_width=0.3)[1] == []
    q = corridor_corners(0.0, 0.0, math.pi / 2, rear=0.5, far=1.0, half_width=0.2)
    assert q[:, 1].min() == pytest.approx(-0.5) and q[:, 1].max() == pytest.approx(1.0)
    assert q[:, 0].min() == pytest.approx(-0.2) and q[:, 0].max() == pytest.approx(0.2)
    assert FrictionMap().query(0, 0, 0, 1.0, 0.3) == (DEFAULT_MU, [], math.inf)
    assert fm.as_list() == [{"id": "plate", "rect": [1.0, -1.2, 5.0, 1.2], "mu": 0.15}]
