"""Fallback navigation map rasterised from world_layout.yaml."""
import pytest

from dusan_common.layout import Layout
from dusan_common.planning import path_length, plan

from plant_simulation.layout_map import rasterize_layout


@pytest.fixture(scope="module")
def lay_grid():
    lay = Layout.load()
    return lay, rasterize_layout(lay)


def test_extent_and_resolution(lay_grid):
    lay, g = lay_grid
    x0, y0, x1, y1 = lay["nav_map"]["bounds"]
    assert g.resolution == pytest.approx(0.10)
    assert g.origin_x == pytest.approx(x0) and g.origin_y == pytest.approx(y0)
    assert g.width * g.resolution == pytest.approx(x1 - x0, abs=0.11)
    assert g.height * g.resolution == pytest.approx(y1 - y0, abs=0.11)
    assert set(map(int, set(g.occ.flatten().tolist()))) <= {0, 100}


def test_semantics(lay_grid):
    lay, g = lay_grid
    sx, sy = lay.spawn[:2]
    assert g.is_free(sx, sy, 0.3)                                   # spawn / dock
    assert not g.is_free(-1.5, -4.5)                                 # turbine footprint
    assert not g.is_free(-6.5, 4.0)                                  # gas tank
    assert not g.is_free(-1.0, 4.0)                                  # drum
    assert not g.is_free(3.5, 6.4)                                   # pipe-rack support
    assert not g.is_free(8.0, 1.8)                                   # pallet
    assert g.is_free(22.0, 0.0)                                      # low pipe in the corridor: crossable
    assert g.is_free(4.5, 0.0)                                       # rubble: crossable
    assert g.is_free(49.0, 0.8)                                      # ramp
    assert g.is_free(15.0, 0.0)                                      # east opening hall <-> corridor
    assert not g.is_free(15.0, 5.0)                                  # wall next to the opening
    assert not g.is_free(30.0, 2.5)                                  # outside the corridor


def test_hall_to_dock_is_navigable(lay_grid):
    lay, g = lay_grid
    p = plan(g, (0.0, -1.5), lay.spawn[:2], robot_radius=0.35)
    assert p is not None
    assert path_length(p) > 60.0


def test_partition_door_frame_and_dropoff_follow_the_layout(lay_grid):
    """Walls the world generator measured in the meshes (layout factory.partition,
    corridor.door_frame, corridor.dropoff) cut the fallback map like the real one."""
    lay, g = lay_grid
    part = lay["factory"]["partition"]
    xw = 0.5 * (part["x_min"] + part["x_max"])
    d0, d1 = part["door_y"]
    for y in (-8.0, -4.0, d0 - 0.15, d1 + 0.15, 4.0, 8.0):
        assert not g.is_free(xw, y), ("partition", y)
    for y in (d0 + 0.15, 0.0, d1 - 0.15):
        assert g.is_free(xw, y), ("doorway", y)
    fr = lay["corridor"]["door_frame"]
    xf = 0.5 * (fr["x_min"] + fr["x_max"])
    assert not g.is_free(xf, 1.6) and not g.is_free(xf, -1.6) and g.is_free(xf, 0.0)
    dr = lay["corridor"]["dropoff"]
    assert not g.is_free(49.0, 0.5 * (dr["y_min"] + dr["y_max"]))
    assert g.is_free(48.4, -1.0) and g.is_free(48.0, 0.8)           # stairs and ramp themselves


def test_paths_through_the_partition_door_and_down_the_ramp(lay_grid):
    lay, g = lay_grid
    d0, d1 = lay["factory"]["partition"]["door_y"]
    p = plan(g, (5.0, 6.0), (12.0, 6.0), robot_radius=0.35)      # hall -> back room
    assert p is not None
    xw = 9.815
    crossing = [y0 + (xw - x0) / (x1 - x0) * (y1 - y0) for (x0, y0), (x1, y1) in zip(p[:-1], p[1:])
                if x0 != x1 and min(x0, x1) <= xw <= max(x0, x1)]
    assert crossing and all(d0 <= y <= d1 for y in crossing), crossing
    p = plan(g, (45.0, -1.0), (53.0, -1.0), robot_radius=0.30)    # stairs side down to the lower floor
    assert p is not None
    for x, y in p:
        assert g.is_free(x, y)
