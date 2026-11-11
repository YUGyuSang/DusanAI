"""Shared fixtures for the Module-3 tests (ROS free).

The unit tests use the map rasterised from world_layout.yaml so that their
numbers do not depend on the (separately generated) plant_map.pgm; the
binding map is exercised by ``test_real_map_sanity`` and tools/eval_gas.py.
"""
import pytest

from dusan_common.layout import Layout

from module3_gas_safety.gas_model import GasConfig, GasDispersion2D
from module3_gas_safety.layout_map import build_layout_map


@pytest.fixture(scope="session")
def layout():
    return Layout.load()


@pytest.fixture(scope="session")
def layout_grid(layout):
    return build_layout_map(layout)


@pytest.fixture(scope="session")
def plant_model(layout_grid):
    """Gas model on the layout map, spun up to steady state (base wind)."""
    m = GasDispersion2D.from_nav_map(layout_grid, GasConfig())
    m.spin_up(400.0, 0.5)
    return m
