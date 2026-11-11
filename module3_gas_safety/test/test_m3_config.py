"""Config / launch / node-file consistency (catches rclpy start-up errors on Windows)."""
import ast
import os
import py_compile
import re
import sys
import types

import pytest
import yaml

from module3_gas_safety import params as P

PKG = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
NODES = {
    "gas_dispersion_sim": ("gas_dispersion.yaml", P.GAS_PARAMS, "gas_dispersion_node"),
    "source_seeker": ("seeker.yaml", P.SEEKER_PARAMS, "source_seeker_node"),
    "battery_simulator": ("safety.yaml", P.BATTERY_PARAMS, "battery_node"),
    "comm_simulator": ("safety.yaml", P.COMM_PARAMS, "comm_node"),
    "return_to_home": ("safety.yaml", P.RTH_PARAMS, "return_to_home_node"),
}


def _yaml(name):
    with open(os.path.join(PKG, "config", name), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _kind(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        kinds = {_kind(x) for x in v}
        return "list[" + ",".join(sorted(kinds)) + "]"
    return type(v).__name__


@pytest.mark.parametrize("node", sorted(NODES))
def test_yaml_matches_declared_parameters(node):
    fname, defaults, _ = NODES[node]
    data = _yaml(fname)
    assert node in data, f"{fname} has no section {node}"
    rp = data[node]["ros__parameters"]
    assert set(rp) == set(defaults), (f"{node}: missing {sorted(set(defaults) - set(rp))}, "
                                      f"unknown {sorted(set(rp) - set(defaults))}")
    for k, v in rp.items():
        assert _kind(v) == _kind(defaults[k]), f"{node}.{k}: YAML {v!r} ({_kind(v)}) vs default {defaults[k]!r}"


@pytest.mark.parametrize("node", sorted(NODES))
def test_yaml_values_equal_code_defaults(node):
    """The YAML documents the defaults; keep both in sync (values may be tuned
    in YAML later, but then the README/eval must be regenerated)."""
    fname, defaults, _ = NODES[node]
    rp = _yaml(fname)[node]["ros__parameters"]
    for k, v in rp.items():
        d = defaults[k]
        if isinstance(d, float):
            assert v == pytest.approx(d), f"{node}.{k}"
        elif isinstance(d, list):
            assert v == pytest.approx(d), f"{node}.{k}"
        else:
            assert v == d, f"{node}.{k}"


def test_rth_energy_model_matches_battery_simulator():
    s = _yaml("safety.yaml")
    bat = s["battery_simulator"]["ros__parameters"]
    rth = s["return_to_home"]["ros__parameters"]
    for k in ("capacity_wh", "p_compute_w", "p_sensors_w", "p_stand_w", "p_sit_w", "p_per_mps_w", "p_per_radps_w"):
        assert rth[k] == bat[k], k


def test_rth_comm_model_matches_comm_simulator():
    s = _yaml("safety.yaml")
    assert (s["return_to_home"]["ros__parameters"]["rf_min_obstacle_area_m2"]
            == s["comm_simulator"]["ros__parameters"]["rf_min_obstacle_area_m2"])


def test_sensor_geometry_consistent_between_sim_and_seeker():
    g = _yaml("gas_dispersion.yaml")["gas_dispersion_sim"]["ros__parameters"]
    s = _yaml("seeker.yaml")["source_seeker"]["ros__parameters"]
    assert g["nose_offset"] == s["nose_offset"]
    assert g["array_offsets"] == s["array_offsets"]
    assert g["sensor_tau"] == s["sensor_lag"]
    assert g["sensor_noise_ppm"] == s["sensor_noise_ppm"]


def test_param_conversions_build_valid_configs(layout):
    gc = P.gas_config(P.GAS_PARAMS, layout)
    assert gc.source_xy == tuple(layout.gas_leak["source"][:2])
    assert gc.wind_base == tuple(layout.gas_leak["wind_base"])
    assert len(gc.inflow_rects) == 3 and len(gc.outflow_rects) == 4
    sc = P.seeker_config(P.SEEKER_PARAMS)
    assert len(sc.array_offsets) == 4 and sc.nose_offset == (0.42, 0.0)
    rc = P.rth_config(P.RTH_PARAMS, layout)
    assert rc.rssi_min_dbm == layout.comm["rssi_min_dbm"]
    bc = P.battery_config(P.BATTERY_PARAMS)
    assert bc.cells_series == 14 and isinstance(bc.cells_series, int)


def _compile(path, tmp_path):
    """py_compile into tmp_path: a launch/__pycache__ in the source tree would
    be installed by setup.py (and copied to the robot PC)."""
    py_compile.compile(path, cfile=os.path.join(str(tmp_path), os.path.basename(path) + "c"), doraise=True)


def test_node_files_compile_and_define_main(tmp_path):
    for _, (_, _, mod) in NODES.items():
        path = os.path.join(PKG, "module3_gas_safety", mod + ".py")
        _compile(path, tmp_path)
        tree = ast.parse(open(path, encoding="utf-8").read())
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert "main" in names, mod
    # ROS-free modules must not import rclpy at module level
    for mod in ("gas_model", "source_seeking", "battery_model", "comm_model", "rth_logic", "params",
                "layout_map", "sim_harness", "node_common"):
        tree = ast.parse(open(os.path.join(PKG, "module3_gas_safety", mod + ".py"), encoding="utf-8").read())
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in n.names] + ([n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
                assert not any(x and (x.startswith("rclpy") or x.endswith("_msgs") or ".msg" in x) for x in names), mod


def test_launch_file_references_existing_entry_points_and_configs(tmp_path):
    path = os.path.join(PKG, "launch", "gas_safety.launch.py")
    _compile(path, tmp_path)
    src = open(path, encoding="utf-8").read()
    setup_src = open(os.path.join(PKG, "setup.py"), encoding="utf-8").read()
    for node, (fname, _, mod) in NODES.items():
        assert f'executable="{node}"' in src and f'name="{node}"' in src
        assert f'"{fname}"' in src
        assert f"{node} = module3_gas_safety.{mod}:main" in setup_src
    for arg in ("use_sim_time", "map_yaml", "layout_file"):
        assert f'DeclareLaunchArgument("{arg}"' in src


def test_no_deprecated_logger_warn_alias():
    """RcutilsLogger.warn() is the deprecated alias of warning() (gone in newer
    rclpy); an AttributeError inside a timer would kill the node."""
    pat = re.compile(r"(get_logger\(\)|logger)\.warn\(")
    for d in ("module3_gas_safety", "launch"):
        for f in os.listdir(os.path.join(PKG, d)):
            if f.endswith(".py"):
                src = open(os.path.join(PKG, d, f), encoding="utf-8").read()
                assert not pat.search(src), f"{d}/{f} uses .warn( - use .warning("


def test_setup_tree_skips_bytecode_caches(tmp_path, monkeypatch):
    captured = {}
    stub = types.ModuleType("setuptools")
    stub.setup = lambda **kw: captured.update(kw)
    monkeypatch.setitem(sys.modules, "setuptools", stub)
    monkeypatch.chdir(PKG)
    ns = {"__name__": "setup_under_test"}
    exec(compile(open(os.path.join(PKG, "setup.py"), encoding="utf-8").read(), "setup.py", "exec"), ns)
    installed = [f for _, files in captured["data_files"] for f in files]
    assert installed and not any("__pycache__" in f or f.endswith(".pyc") for f in installed)
    # a cache next to the launch file is not installed
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("launch", "__pycache__"))
    open(os.path.join("launch", "x.launch.py"), "w").close()
    open(os.path.join("launch", "__pycache__", "x.launch.cpython-312.pyc"), "w").close()
    out = ns["tree"]("launch")
    assert out == [(os.path.join("share", "module3_gas_safety", "launch"), [os.path.join("launch", "x.launch.py")])]
