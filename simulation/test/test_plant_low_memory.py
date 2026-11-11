"""Low-memory profile (8 GB laptop), camera arguments, shadow-free world and respawn of the launch
files - dry runs against the launch stand-ins of test_plant_launch_files plus the real xacro."""
import importlib.util
import inspect
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_plant_launch_files import (MODULE_LAUNCH_PATHS, SIM, Context, DeclareLaunchArgument,  # noqa: E402
                                     IncludeLaunchDescription, Node, _stubs, load_launch_module, nodes,
                                     param_dicts, run_setup)

PROFILE_ARGS = ("inspection_rate", "inspection_width", "inspection_height", "depth_rate", "thermal_rate", "shadows",
                "gui_config")


def _sim_include(actions):
    return [a for a in actions if isinstance(a, IncludeLaunchDescription)
            and a.path.endswith(os.path.join("launch", "sim.launch.py"))][0]


def _env(actions, cls):
    return {a.args[0]: a.args[1] for a in actions if isinstance(a, cls["SetEnvironmentVariable"])}


def _xacro_mappings(sim_actions, cls):
    """{arg: value} of the xacro command line sim.launch.py builds for robot_state_publisher."""
    cmd = param_dicts(nodes(sim_actions, cls)["robot_state_publisher"])[0]["robot_description"].args[0].args[0]
    return {cmd[i].strip()[:-2]: cmd[i + 1] for i in range(2, len(cmd) - 1, 2)}


def _gz_args(sim_actions):
    return [a for a in sim_actions if isinstance(a, IncludeLaunchDescription)][0].launch_args["gz_args"].split()


def test_bringup_default_profile_is_unchanged(tmp_path):
    stubs = _stubs(tmp_path)
    cls = stubs["__classes__"]
    actions, _ = run_setup("bringup.launch.py", stubs)
    sim = _sim_include(actions).launch_args
    for k in PROFILE_ARGS:
        assert sim[k] == "", k                        # '' = xacro / world / gui.config defaults
    assert not _env(actions, cls)                      # no MALLOC_ARENA_MAX / BLAS thread limits
    n = nodes(actions, cls)
    assert param_dicts(n["demo_hud"]) == [{"use_sim_time": True, "layout_file": ""}]   # no HUD rate override
    sim_actions, _ = run_setup("sim.launch.py", stubs, sim)
    assert set(_xacro_mappings(sim_actions, cls)) == {"enable_thermal", "enable_cameras"}
    gz = _gz_args(sim_actions)
    assert gz[3].endswith(os.path.join("worlds", "plant.sdf"))
    assert gz[gz.index("--gui-config") + 1].endswith(os.path.join("config", "gui.config"))


def test_bringup_low_memory_profile(tmp_path):
    stubs = _stubs(tmp_path)
    cls = stubs["__classes__"]
    actions, _ = run_setup("bringup.launch.py", stubs, {"low_memory": "true", "rviz": "true", "demo": "true"})
    sim = _sim_include(actions).launch_args
    assert (sim["inspection_width"], sim["inspection_height"], sim["inspection_rate"]) == ("640", "480", "5")
    # the depth camera keeps the 10 Hz the walking (pipe crossings, obstacle pace) was checked with
    assert (sim["depth_rate"], sim["thermal_rate"], sim["shadows"]) == ("10", "2", "false")
    assert sim["gui_config"] == "gui_lean.config"
    env = _env(actions, cls)
    assert env["MALLOC_ARENA_MAX"] == os.environ.get("MALLOC_ARENA_MAX", "2")
    assert {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"} <= set(env)
    n = nodes(actions, cls)
    assert "rviz2" not in n                           # never with low_memory, even when asked for
    assert {"monitor_dashboard", "metrics_logger", "mission_manager", "demo_hud", "follow_camera"} <= set(n)
    assert param_dicts(n["demo_hud"])[0]["rate"] == pytest.approx(3.0)
    logs = " ".join(a.kwargs.get("msg", "") for a in actions if isinstance(a, cls["LogInfo"]))
    assert "LOW MEMORY" in logs and "640x480 @ 5 Hz" in logs and "RViz skipped" in logs
    # sim.launch.py with those arguments: xacro camera arguments, a world copy without shadows, lean GUI
    sim_actions, _ = run_setup("sim.launch.py", stubs, sim)
    m = _xacro_mappings(sim_actions, cls)
    assert m == {"enable_thermal": "true", "enable_cameras": "true", "depth_rate": "10", "inspection_rate": "5",
                 "thermal_rate": "2", "inspection_width": "640", "inspection_height": "480"}
    gz = _gz_args(sim_actions)
    world = gz[3]
    assert world.endswith("plant_noshadows.sdf") and os.path.isfile(world)
    text = open(world, encoding="utf-8").read()
    original = open(os.path.join(SIM, "worlds", "plant.sdf"), encoding="utf-8").read()
    assert "<shadows>true</shadows>" in original and "<cast_shadows>true</cast_shadows>" in original
    assert "<shadows>true</shadows>" not in text and "<cast_shadows>true</cast_shadows>" not in text
    # only true -> false, nothing else of the generated world changed
    assert text == original.replace("<shadows>true</shadows>", "<shadows>false</shadows>").replace(
        "<cast_shadows>true</cast_shadows>", "<cast_shadows>false</cast_shadows>")
    assert gz[gz.index("--gui-config") + 1].endswith(os.path.join("config", "gui_lean.config"))
    assert any("shadows off" in a.kwargs.get("msg", "") for a in sim_actions if isinstance(a, cls["LogInfo"]))


def test_bringup_low_memory_explicit_arguments_win(tmp_path):
    stubs = _stubs(tmp_path)
    actions, _ = run_setup("bringup.launch.py", stubs, {"low_memory": "true", "inspection_rate": "8",
                                                        "shadows": "true", "hud_rate": "5.0"})
    sim = _sim_include(actions).launch_args
    assert sim["inspection_rate"] == "8" and sim["shadows"] == "true" and sim["inspection_width"] == "640"
    n = nodes(actions, stubs["__classes__"])
    assert param_dicts(n["demo_hud"])[0]["rate"] == pytest.approx(5.0)
    sim_actions, _ = run_setup("sim.launch.py", stubs, sim)
    assert _gz_args(sim_actions)[3].endswith(os.path.join("worlds", "plant.sdf"))   # shadows:=true -> original


def test_low_memory_gui_config_keeps_what_the_demo_needs():
    lean = open(os.path.join(SIM, "config", "gui_lean.config"), encoding="utf-8").read()
    full = open(os.path.join(SIM, "config", "gui.config"), encoding="utf-8").read()
    plugins = lambda s: re.findall(r'<plugin filename="([A-Za-z]+)" name="([^"]+)"', s)  # noqa: E731
    lean_p, full_p = plugins(lean), plugins(full)
    assert set(lean_p) < set(full_p)
    names = {n for _, n in lean_p}
    # 3D view, keys for 't' / teleop, follow camera, HUD panel and world control stay
    assert {"3D View", "Key publisher", "Camera Tracking", "Robot HUD", "World control"} <= names
    assert not {"Inspection camera", "Thermal camera", "Component inspector", "Entity tree"} & names


def test_sim_launch_rejects_a_non_numeric_camera_argument(tmp_path):
    with pytest.raises(RuntimeError, match="inspection_rate"):
        run_setup("sim.launch.py", _stubs(tmp_path), {"inspection_rate": "fast"})


def test_world_without_shadows_helper(tmp_path):
    mod = load_launch_module("sim.launch.py", _stubs(tmp_path))
    w = tmp_path / "w.sdf"
    w.write_text("<sdf><world><scene><shadows>true</shadows></scene>"
                 "<light><cast_shadows>1</cast_shadows></light></world></sdf>", encoding="utf-8")
    out = mod.world_without_shadows(str(w), out_dir=str(tmp_path / "o"))
    assert out != str(w) and open(out, encoding="utf-8").read() == (
        "<sdf><world><scene><shadows>false</shadows></scene>"
        "<light><cast_shadows>false</cast_shadows></light></world></sdf>")
    assert mod.world_without_shadows(out, out_dir=str(tmp_path / "o2")) == out     # nothing to change
    missing = str(tmp_path / "missing.sdf")
    assert mod.world_without_shadows(missing) == missing


def test_xacro_camera_arguments_are_declared_by_the_urdf():
    mod = load_launch_module("sim.launch.py", _stubs(tmp=None))
    declared = set(re.findall(r'<xacro:arg\s+name="([A-Za-z0-9_]+)"',
                              open(os.path.join(SIM, "urdf", "spot.urdf.xacro"), encoding="utf-8").read()))
    assert set(mod.XACRO_CAMERA_ARGS) <= declared
    assert set(mod.XACRO_CAMERA_ARGS) == set(PROFILE_ARGS) - {"shadows", "gui_config"}


def test_xacro_expands_the_low_memory_cameras():
    """The real xacro with the arguments of the low-memory profile (and without them)."""
    xacro = pytest.importorskip("xacro")
    import xml.etree.ElementTree as ET
    urdf = os.path.join(SIM, "urdf", "spot.urdf.xacro")

    def sensors(mappings):
        root = ET.fromstring(xacro.process_file(urdf, mappings=dict(mappings)).toxml())
        return {s.get("name"): s for s in root.iter("sensor")}

    low = sensors({"depth_rate": "5", "inspection_rate": "5", "thermal_rate": "2", "inspection_width": "640",
                   "inspection_height": "480"})
    insp = low["inspection_camera"]
    assert (insp.findtext("camera/image/width"), insp.findtext("camera/image/height")) == ("640", "480")
    assert float(insp.findtext("update_rate")) == 5.0
    assert float(low["depth_camera"].findtext("update_rate")) == 5.0
    assert float(low["thermal_camera"].findtext("update_rate")) == 2.0
    assert low["depth_camera"].findtext("camera/image/width") == "212"          # depth size unchanged
    default = sensors({})
    insp = default["inspection_camera"]
    assert (insp.findtext("camera/image/width"), insp.findtext("camera/image/height")) == ("800", "600")
    assert [float(default[n].findtext("update_rate")) for n in ("inspection_camera", "depth_camera",
                                                                "thermal_camera")] == [10.0, 10.0, 5.0]


def test_respawn_of_safe_nodes(tmp_path):
    stubs = _stubs(tmp_path)
    cls = stubs["__classes__"]
    actions, _ = run_setup("bringup.launch.py", stubs, {"demo": "true"})
    n = nodes(actions, cls)
    for exe in ("cmd_vel_mux", "mission_manager", "monitor_dashboard", "demo_hud"):
        assert n[exe].respawn is True and n[exe].respawn_delay == pytest.approx(3.0), exe
    for exe in ("metrics_logger", "follow_camera"):
        assert not n[exe].respawn, exe                # split results / exits by design
    sim = _sim_include(actions).launch_args
    assert sim["respawn"] == "true"
    timer = [a for a in actions if isinstance(a, cls["TimerAction"])][0]
    m3 = [a for a in timer.kwargs["actions"] if isinstance(a, IncludeLaunchDescription)
          and a.path.endswith("gas_safety.launch.py")]
    if m3:
        assert m3[0].launch_args["respawn"] == "true"
    sim_actions, _ = run_setup("sim.launch.py", stubs, sim)
    sn = nodes(sim_actions, cls)
    for exe in ("robot_state_publisher", "parameter_bridge", "clock_relay", "map_server_lite"):
        assert sn[exe].respawn is True, exe
    for exe in ("odom_tf", "create"):
        assert not sn[exe].respawn, exe                # first-sample spawn detection / one-shot spawner
    off, _ = run_setup("bringup.launch.py", stubs, {"respawn": "false"})
    assert not any(x.respawn for x in nodes(off, cls).values())
    assert _sim_include(off).launch_args["respawn"] == "false"


def test_respawn_max_retries_only_where_launch_supports_it(tmp_path):
    """launch (Jazzy and later, the stubs): respawn_max_retries is a parameter of ExecuteProcess' BASE
    class ExecuteLocal - every respawned node gets at most 10 restarts, none restarts forever.
    Humble: no such parameter anywhere - not passed (it would be a TypeError)."""
    stubs = _stubs(tmp_path)
    cls = stubs["__classes__"]
    mod = load_launch_module("sim.launch.py", stubs)
    assert mod.respawn_kwargs(False) == {}
    assert "respawn_max_retries" not in inspect.signature(cls["ExecuteProcess"].__init__).parameters
    assert mod.respawn_kwargs(True) == {"respawn": True, "respawn_delay": 3.0, "respawn_max_retries": 10}
    actions, _ = run_setup("bringup.launch.py", stubs, {"demo": "true"})
    respawned = [x for x in nodes(actions, cls).values() if x.respawn]
    sim_actions, _ = run_setup("sim.launch.py", stubs, _sim_include(actions).launch_args)
    respawned += [x for x in nodes(sim_actions, cls).values() if x.respawn]
    ns = {x.executable for x in respawned}
    assert {"cmd_vel_mux", "mission_manager", "monitor_dashboard", "demo_hud", "robot_state_publisher",
            "parameter_bridge", "clock_relay", "map_server_lite"} <= ns
    assert all(x.respawn_max_retries == 10 for x in respawned)
    m3 = _load_gas_safety(stubs).generate_launch_description()
    cfg = {a.name: a.default for a in m3.entities if isinstance(a, DeclareLaunchArgument)}
    out = [x for e in m3.entities if isinstance(e, cls["OpaqueFunction"]) for x in e.kwargs["function"](Context(cfg))]
    m3_respawned = [x for x in out if isinstance(x, Node) and x.respawn]
    assert len(m3_respawned) == 4 and all(x.respawn_max_retries == 10 for x in m3_respawned)

    class HumbleExecuteProcess:                                                  # Humble: no such parameter
        def __init__(self, *, cmd, name=None, cwd=None, env=None, additional_env=None, shell=False,
                     sigterm_timeout=None, sigkill_timeout=None, emulate_tty=False, output="log",
                     respawn=False, respawn_delay=None, **kwargs):
            pass

    mod.ExecuteProcess = HumbleExecuteProcess
    assert mod.respawn_kwargs(True) == {"respawn": True, "respawn_delay": 3.0}


def _load_gas_safety(stubs):
    path = MODULE_LAUNCH_PATHS[2][1]
    saved = {k: sys.modules.get(k) for k in stubs if not k.startswith("__")}
    dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        for k, v in stubs.items():
            if not k.startswith("__"):
                sys.modules[k] = v
        spec = importlib.util.spec_from_file_location("gas_safety_launch_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = dwb
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_gas_safety_launch_respawns_all_but_the_battery(tmp_path):
    stubs = _stubs(tmp_path)
    cls = stubs["__classes__"]
    ld = _load_gas_safety(stubs).generate_launch_description()
    cfg = {a.name: a.default for a in ld.entities if isinstance(a, DeclareLaunchArgument)}
    assert cfg["respawn"] == "true"
    for value, expect in (("true", True), ("false", False)):
        ctx = Context(dict(cfg, respawn=value))
        out = [x for e in ld.entities if isinstance(e, cls["OpaqueFunction"]) for x in e.kwargs["function"](ctx)]
        ns = {x.executable: x for x in out if isinstance(x, Node)}
        assert set(ns) == {"gas_dispersion_sim", "source_seeker", "battery_simulator", "comm_simulator",
                           "return_to_home"}
        assert not ns["battery_simulator"].respawn     # would come back with the start charge
        for exe in ("gas_dispersion_sim", "source_seeker", "comm_simulator", "return_to_home"):
            assert bool(ns[exe].respawn) is expect, (exe, value)
        for x in ns.values():
            assert x.parameters[-1]["use_sim_time"] is not None
