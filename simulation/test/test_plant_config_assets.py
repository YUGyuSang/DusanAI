"""Static consistency checks of the integration package: parameter YAMLs vs
declared defaults, RViz config, web assets, shell scripts, ROS-free module
rule and the DESIGN_SPEC topic contract."""
import ast
import os
import py_compile
import re
import shutil
import subprocess

import pytest
import yaml

from plant_simulation.clock_relay import CLOCK_RELAY_DEFAULTS
from plant_simulation.dashboard_state import DASHBOARD_DEFAULTS
from plant_simulation.layout_map import MAP_SERVER_DEFAULTS
from plant_simulation.metrics import METRICS_DEFAULTS
from plant_simulation.mission_logic import MISSION_DEFAULTS
from plant_simulation.mux_logic import MUX_DEFAULTS
from plant_simulation.odom_logic import ODOM_DEFAULTS

HERE = os.path.dirname(os.path.realpath(__file__))
SIM = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(SIM, ".."))
PKG_DIR = os.path.join(SIM, "plant_simulation")
WEB = os.path.join(SIM, "web")
SCRIPTS = os.path.join(REPO, "scripts")

PARAM_FILES = {
    ("mission.yaml", "mission_manager"): MISSION_DEFAULTS,
    ("dashboard.yaml", "monitor_dashboard"): DASHBOARD_DEFAULTS,
    ("dashboard.yaml", "metrics_logger"): METRICS_DEFAULTS,
    ("mux.yaml", "cmd_vel_mux"): MUX_DEFAULTS,
    ("mux.yaml", "odom_tf"): ODOM_DEFAULTS,
    ("mux.yaml", "map_server_lite"): MAP_SERVER_DEFAULTS,
    ("mux.yaml", "clock_relay"): CLOCK_RELAY_DEFAULTS,
}
NODE_DEFAULTS = {
    "mission_manager_node.py": "MISSION_DEFAULTS", "monitor_dashboard_node.py": "DASHBOARD_DEFAULTS",
    "metrics_logger_node.py": "METRICS_DEFAULTS", "cmd_vel_mux_node.py": "MUX_DEFAULTS",
    "odom_tf_node.py": "ODOM_DEFAULTS", "map_server_node.py": "MAP_SERVER_DEFAULTS",
    "clock_relay_node.py": "CLOCK_RELAY_DEFAULTS",
}


def _ptype(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    return type(v).__name__


# ----------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(PARAM_FILES), ids=lambda k: f"{k[0]}:{k[1]}")
def test_param_yaml_matches_declared_defaults(key):
    fname, node = key
    defaults = PARAM_FILES[key]
    with open(os.path.join(SIM, "config", fname), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    params = data[node]["ros__parameters"]
    assert set(params) == set(defaults), (f"missing {set(defaults) - set(params)}, "
                                          f"unknown {set(params) - set(defaults)}")
    for k, v in params.items():
        # rclpy refuses e.g. an INTEGER override of a DOUBLE parameter
        assert _ptype(v) == _ptype(defaults[k]), f"{node}.{k}: yaml {v!r} vs default {defaults[k]!r}"
        assert v == defaults[k], f"{node}.{k}: yaml {v!r} differs from the code default {defaults[k]!r}"
    assert "use_sim_time" not in defaults          # declared by rclpy itself


@pytest.mark.parametrize("fname,dict_name", sorted(NODE_DEFAULTS.items()))
def test_nodes_declare_their_default_dicts(fname, dict_name):
    src = open(os.path.join(PKG_DIR, fname), encoding="utf-8").read()
    assert f"declare_params(self, {dict_name})" in src
    assert "def main(" in src and "run_node(" in src


def test_mission_defaults_are_sane():
    p = MISSION_DEFAULTS
    assert p["dwell_time"] == 4.0 and p["min_readings"] == 5          # DESIGN_SPEC section 5
    assert p["gas_alarm_ppm"] == 25.0 and p["gas_alarm_hold"] == 1.0
    assert p["min_dwell_time"] < p["dwell_time"]
    assert MUX_DEFAULTS["timeout"] == 0.5 and MUX_DEFAULTS["rate"] == 20.0
    assert DASHBOARD_DEFAULTS["port"] == 8080 and DASHBOARD_DEFAULTS["sse_rate"] == 5.0
    assert METRICS_DEFAULTS["pose_rate"] == 2.0 and METRICS_DEFAULTS["summary_period"] == 10.0


# ----------------------------------------------------------------------------
# ROS-free rule / python hygiene
# ----------------------------------------------------------------------------
ROS_MODULES = ("rclpy", "std_msgs", "geometry_msgs", "nav_msgs", "sensor_msgs", "visualization_msgs", "tf2_ros",
               "rosgraph_msgs", "builtin_interfaces", "launch", "launch_ros", "cv_bridge")
PY_FILES = sorted(f for f in os.listdir(PKG_DIR) if f.endswith(".py"))


@pytest.mark.parametrize("fname", PY_FILES)
def test_python_module_compiles_lf_and_ros_free_rule(fname):
    path = os.path.join(PKG_DIR, fname)
    py_compile.compile(path, doraise=True)
    raw = open(path, "rb").read()
    assert b"\r\n" not in raw, "CRLF line endings"
    tree = ast.parse(raw.decode("utf-8"))
    top = []
    for n in tree.body:
        if isinstance(n, ast.Import):
            top += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            top.append(n.module)
    ros = [m for m in top if m.split(".")[0] in ROS_MODULES]
    if fname.endswith("_node.py"):
        assert "rclpy" in [m.split(".")[0] for m in top]
    else:
        assert not ros, f"{fname} imports ROS at module level: {ros}"
    assert not any(m.split(".")[0] in ("scipy", "cv_bridge", "tf_transformations") for m in top)


def test_package_xml_declares_the_python_ros_modules_imported_by_the_nodes():
    """Python ROS modules must come from a declared exec_depend - e.g. the
    `tf2_ros` Python module ships in tf2_ros_py, not in the C++ tf2_ros
    package (it was only present through rviz2 before)."""
    import xml.etree.ElementTree as ET
    deps = {e.text.strip() for e in ET.parse(os.path.join(SIM, "package.xml")).getroot().iter("exec_depend")}
    provider = {"tf2_ros": "tf2_ros_py"}               # python module -> ROS package that installs it
    for fname in PY_FILES:
        tree = ast.parse(open(os.path.join(PKG_DIR, fname), encoding="utf-8").read())
        for n in ast.walk(tree):
            names = ([a.name for a in n.names] if isinstance(n, ast.Import)
                     else [n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
            for m in names:
                top = m.split(".")[0]
                if top in ROS_MODULES and top != "cv_bridge":
                    assert provider.get(top, top) in deps, f"{fname} imports {m}: add <exec_depend>"                                                            f"{provider.get(top, top)}</exec_depend>"


def test_topic_names_follow_the_design_spec():
    spec = open(os.path.join(REPO, "docs", "DESIGN_SPEC.md"), encoding="utf-8").read()
    spec_topics = set(re.findall(r"`?(/[a-z_][a-z0-9_/]*)`?", spec))
    extra_ok = {"/clicked_point"}                      # RViz tool topic
    for fname in PY_FILES:
        if not fname.endswith("_node.py"):
            continue
        tree = ast.parse(open(os.path.join(PKG_DIR, fname), encoding="utf-8").read())
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and re.fullmatch(r"/[a-z_][a-z0-9_/]*",
                                                                                           n.value):
                assert n.value in spec_topics or n.value in extra_ok, f"{fname}: topic {n.value} not in DESIGN_SPEC"


# ----------------------------------------------------------------------------
# RViz
# ----------------------------------------------------------------------------
def _walk_displays(displays):
    for d in displays:
        yield d
        if d.get("Class") == "rviz_common/Group":
            yield from _walk_displays(d.get("Displays") or [])


def test_rviz_config():
    with open(os.path.join(SIM, "rviz", "plant.rviz"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vm = cfg["Visualization Manager"]
    assert vm["Global Options"]["Fixed Frame"] == "map"
    disps = list(_walk_displays(vm["Displays"]))
    classes = {d["Class"] for d in disps}
    assert all(c.startswith(("rviz_default_plugins/", "rviz_common/")) for c in classes)
    topics = {}
    for d in disps:
        t = d.get("Topic") or d.get("Description Topic")
        if isinstance(t, dict):
            topics[t["Value"]] = d
    required = {"/robot_description", "/map", "/gas/concentration_grid", "/comm/rssi_map",
                "/elevation_map/traversability", "/elevation_map/points", "/mpc/predicted_trajectory",
                "/mpc/footholds", "/mpc/grf", "/patrol/route", "/patrol/markers", "/nav/plan", "/gas/seek_path",
                "/twin/markers", "/inspection/gauge_overlay", "/inspection/thermal_fusion"}
    assert required <= set(topics), required - set(topics)
    assert topics["/map"]["Topic"]["Durability Policy"] == "Transient Local"
    assert "rviz_default_plugins/TF" in classes and "rviz_default_plugins/RobotModel" in classes


# ----------------------------------------------------------------------------
# web assets
# ----------------------------------------------------------------------------
def test_web_assets_self_contained():
    html = open(os.path.join(WEB, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(WEB, "app.js"), encoding="utf-8").read()
    css = open(os.path.join(WEB, "style.css"), encoding="utf-8").read()
    assert 'src="app.js"' in html and 'href="style.css"' in html
    for name, text in (("index.html", html), ("app.js", js), ("style.css", css)):
        assert not re.search(r"https?://", text), f"{name} references an external URL (no CDN allowed)"
    # every element id used by app.js exists in the page
    ids_html = set(re.findall(r'id="([^"]+)"', html))
    ids_js = set(re.findall(r'\$\("([A-Za-z0-9_]+)"\)', js))
    assert ids_js <= ids_html, ids_js - ids_html
    # every command button maps to a valid backend command
    from plant_simulation.web_server import validate_command
    cmds = re.findall(r'data-target="([a-z]+)" data-cmd="([^"]+)"', html)
    for target, cmd in cmds:
        assert validate_command(target, cmd)[0], (target, cmd)
    # every operator command still has a button (the labels are Korean, so the commands are checked)
    needed = {("mission", "start"), ("mission", "patrol"), ("mission", "seek"), ("mission", "rth"),
              ("mission", "pause"), ("mission", "abort"), ("locomotion", "stand"), ("locomotion", "sit"),
              ("locomotion", "recover"), ("locomotion", "estop"), ("locomotion", "release")}
    assert needed <= set(cmds), needed - set(cmds)
    assert "textContent" in js and "innerHTML" not in js        # untrusted strings never parsed as HTML


@pytest.mark.skipif(shutil.which("node") is None, reason="node.js not installed")
def test_app_js_syntax():
    r = subprocess.run(["node", "--check", os.path.join(WEB, "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ----------------------------------------------------------------------------
# shell scripts
# ----------------------------------------------------------------------------
SCRIPT_FILES = sorted(f for f in os.listdir(SCRIPTS) if f.endswith(".sh"))


def test_expected_scripts_exist():
    assert {"install_deps.sh", "build.sh", "run_sim.sh", "run_bringup.sh", "run_teleop.sh",
            "check_install.sh", "common.sh"} <= set(SCRIPT_FILES)


@pytest.mark.parametrize("fname", SCRIPT_FILES)
def test_script_format(fname):
    raw = open(os.path.join(SCRIPTS, fname), "rb").read()
    assert b"\r" not in raw, "CRLF line endings break bash"
    text = raw.decode("utf-8")
    if fname == "common.sh":
        assert text.startswith("# shellcheck shell=bash")
    else:
        assert text.startswith("#!/usr/bin/env bash\n")
        assert "\nset -e\n" in text
        assert '. "$(dirname "${BASH_SOURCE[0]}")/common.sh"' in text   # works with `bash scripts/x.sh`


def _bash():
    exe = shutil.which("bash")
    if exe is None:
        return None
    if os.name == "nt" and "system32" in exe.lower():   # WSL launcher, not a real bash
        return None
    return exe


@pytest.mark.skipif(_bash() is None, reason="bash not available")
@pytest.mark.parametrize("fname", SCRIPT_FILES)
def test_script_syntax(fname):
    r = subprocess.run([_bash(), "-n", os.path.join(SCRIPTS, fname)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_launch_and_run_scripts_agree():
    run = open(os.path.join(SCRIPTS, "run_bringup.sh"), encoding="utf-8").read()
    assert "ros2 launch plant_simulation bringup.launch.py" in run
    sim = open(os.path.join(SCRIPTS, "run_sim.sh"), encoding="utf-8").read()
    assert "ros2 launch plant_simulation sim.launch.py" in sim
    deps = open(os.path.join(SCRIPTS, "install_deps.sh"), encoding="utf-8").read()
    for pkg in ("ros-gz", "xacro", "robot-state-publisher", "rviz2", "tf2-ros", "python3-numpy", "python3-opencv",
                "python3-yaml", "xterm", "rosdep"):
        assert pkg in deps, pkg
