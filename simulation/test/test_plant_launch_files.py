"""Launch files: compile, entry-point consistency, and a dry run of the
OpaqueFunction setup code against minimal stand-ins of the launch APIs
(ROS is not available on the development PC)."""
import ast
import importlib.util
import os
import py_compile
import re
import sys
import types

import pytest

HERE = os.path.dirname(os.path.realpath(__file__))
SIM = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(SIM, ".."))
LAUNCH_DIR = os.path.join(SIM, "launch")
LAUNCH_FILES = sorted(f for f in os.listdir(LAUNCH_DIR) if f.endswith(".launch.py"))
REPO_PACKAGES = {"dusan_common": "dusan_common", "module1_locomotion": "module1_locomotion",
                 "module2_inspection": "module2_inspection", "module3_gas_safety": "module3_gas_safety",
                 "plant_simulation": "simulation"}
EXTERNAL_EXECUTABLES = {
    "ros_gz_sim": {"create"},
    "ros_gz_bridge": {"parameter_bridge"},
    "robot_state_publisher": {"robot_state_publisher"},
    "rviz2": {"rviz2"},
}


def entry_points(pkg_dir):
    """console_scripts names declared in <pkg_dir>/setup.py."""
    with open(os.path.join(REPO, pkg_dir, "setup.py"), encoding="utf-8") as f:
        src = f.read()
    return set(re.findall(r'"\s*([A-Za-z0-9_]+)\s*=\s*[A-Za-z0-9_.]+:main"', src))


def test_expected_launch_files_exist():
    assert {"sim.launch.py", "bringup.launch.py", "dashboard.launch.py"} <= set(LAUNCH_FILES)


def _compile(path, tmp_path):
    """py_compile without leaving launch/__pycache__ behind: setup.py installs
    every file below launch/ into the share directory."""
    py_compile.compile(path, cfile=os.path.join(str(tmp_path), os.path.basename(path) + "c"), doraise=True)


@pytest.mark.parametrize("name", LAUNCH_FILES)
def test_launch_file_compiles(name, tmp_path):
    _compile(os.path.join(LAUNCH_DIR, name), tmp_path)


@pytest.mark.parametrize("name", LAUNCH_FILES)
def test_every_node_executable_exists(name):
    """Static check: Node(package=..., executable=...) pairs against setup.py entry points."""
    path = os.path.join(LAUNCH_DIR, name)
    tree = ast.parse(open(path, encoding="utf-8").read())
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}

    def lit(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return consts.get(node.id)
        return None

    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Node":
            kw = {k.arg: lit(k.value) for k in node.keywords}
            pairs.append((kw.get("package"), kw.get("executable")))
    assert pairs, "no Node() found"
    for pkg, exe in pairs:
        assert pkg and exe, f"non-literal package/executable in {name}"
        if pkg in REPO_PACKAGES:
            assert exe in entry_points(REPO_PACKAGES[pkg]), f"{pkg}/{exe} not in setup.py"
        else:
            assert exe in EXTERNAL_EXECUTABLES.get(pkg, set()), f"unexpected external {pkg}/{exe}"


def test_module_launch_files_referenced_by_bringup():
    mod = load_launch_module("bringup.launch.py", _stubs(tmp=None))
    for pkg, launch_file in mod.MODULE_LAUNCHES:
        assert pkg in REPO_PACKAGES and os.path.isfile(os.path.join(REPO, pkg, "package.xml"))
        launch_dir = os.path.join(REPO, pkg, "launch")
        if os.path.isdir(launch_dir) and os.listdir(launch_dir):
            assert launch_file in os.listdir(launch_dir), f"{pkg} provides {os.listdir(launch_dir)}"


# ----------------------------------------------------------------------------
# minimal stand-ins for launch / launch_ros / ament_index_python
# ----------------------------------------------------------------------------
class _Rec:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"{type(self).__name__}({self.args}, {self.kwargs})"


class LaunchConfiguration(_Rec):
    def perform(self, context):
        return context.launch_configurations.get(self.args[0], "")


class DeclareLaunchArgument(_Rec):
    @property
    def name(self):
        return self.args[0]

    @property
    def default(self):
        return self.kwargs.get("default_value", "")


class LaunchDescription(_Rec):
    @property
    def entities(self):
        return self.args[0]


class PythonLaunchDescriptionSource(_Rec):
    pass


class IncludeLaunchDescription(_Rec):
    @property
    def path(self):
        return self.args[0].args[0]

    @property
    def launch_args(self):
        return dict(self.kwargs.get("launch_arguments") or [])


class Node(_Rec):
    def __getattr__(self, item):
        return self.kwargs.get(item)


class ExecuteLocal(_Rec):
    """Shaped like launch (Jazzy and later): respawn / respawn_max_retries are parameters of
    ExecuteLocal.__init__ (respawn_max_retries -1 = restart forever) ..."""

    def __init__(self, *args, respawn=False, respawn_delay=None, respawn_max_retries=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.respawn, self.respawn_delay, self.respawn_max_retries = respawn, respawn_delay, respawn_max_retries


class ExecuteProcess(ExecuteLocal):
    """... and ExecuteProcess.__init__ only has cmd / prefix / name / cwd / env / additional_env and
    **kwargs for its base class."""

    def __init__(self, *args, cmd=None, prefix=None, name=None, cwd=None, env=None, additional_env=None, **kwargs):
        if cmd is not None:
            kwargs["cmd"] = cmd
        super().__init__(*args, **kwargs)


class Context:
    def __init__(self, cfg):
        self.launch_configurations = dict(cfg)


def _stubs(tmp):
    """sys.modules entries for the ROS launch APIs."""
    share_map = {pkg: os.path.join(REPO, d) for pkg, d in REPO_PACKAGES.items()}
    if tmp is not None:
        gz = os.path.join(str(tmp), "ros_gz_sim")
        os.makedirs(os.path.join(gz, "launch"), exist_ok=True)
        open(os.path.join(gz, "launch", "gz_sim.launch.py"), "w").close()
        share_map["ros_gz_sim"] = gz

    class PackageNotFoundError(KeyError):
        pass

    def get_package_share_directory(pkg):
        if pkg not in share_map:
            raise PackageNotFoundError(pkg)
        return share_map[pkg]

    names = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        names[name] = m
        return m

    simple = {n: type(n, (_Rec,), {}) for n in ("OpaqueFunction", "TimerAction", "LogInfo", "SetEnvironmentVariable",
                                               "AppendEnvironmentVariable", "ExecuteProcess", "IfCondition",
                                               "RegisterEventHandler", "OnProcessExit",
                                               "UnlessCondition", "Command", "PathJoinSubstitution",
                                               "PythonExpression", "FindPackageShare", "ParameterValue")}
    simple["ExecuteProcess"] = ExecuteProcess
    mod("launch", LaunchDescription=LaunchDescription)
    mod("launch.actions", DeclareLaunchArgument=DeclareLaunchArgument, IncludeLaunchDescription=IncludeLaunchDescription,
        **{k: simple[k] for k in ("OpaqueFunction", "TimerAction", "LogInfo", "SetEnvironmentVariable",
                                  "AppendEnvironmentVariable", "ExecuteProcess", "RegisterEventHandler")})
    mod("launch.event_handlers", OnProcessExit=simple["OnProcessExit"])
    mod("launch.conditions", IfCondition=simple["IfCondition"], UnlessCondition=simple["UnlessCondition"])
    mod("launch.launch_description_sources", PythonLaunchDescriptionSource=PythonLaunchDescriptionSource)
    mod("launch.substitutions", LaunchConfiguration=LaunchConfiguration, Command=simple["Command"],
        PathJoinSubstitution=simple["PathJoinSubstitution"], PythonExpression=simple["PythonExpression"])
    mod("launch_ros")
    mod("launch_ros.actions", Node=Node)
    mod("launch_ros.parameter_descriptions", ParameterValue=simple["ParameterValue"])
    mod("launch_ros.substitutions", FindPackageShare=simple["FindPackageShare"])
    mod("ament_index_python")
    mod("ament_index_python.packages", get_package_share_directory=get_package_share_directory,
        PackageNotFoundError=PackageNotFoundError)
    names["__classes__"] = simple
    return names


def load_launch_module(fname, stubs):
    saved = {k: sys.modules.get(k) for k in stubs if not k.startswith("__")}
    saved_dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True          # no launch/__pycache__ (it would be installed by setup.py)
    try:
        for k, v in stubs.items():
            if not k.startswith("__"):
                sys.modules[k] = v
        spec = importlib.util.spec_from_file_location(fname.replace(".", "_"), os.path.join(LAUNCH_DIR, fname))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        sys.dont_write_bytecode = saved_dwb
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def run_setup(fname, stubs, overrides=None):
    """generate_launch_description() + OpaqueFunction(s) -> flat list of actions."""
    m = load_launch_module(fname, stubs)
    ld = m.generate_launch_description()
    cls = stubs["__classes__"]
    cfg = {a.name: a.default for a in ld.entities if isinstance(a, DeclareLaunchArgument)}
    cfg.update(overrides or {})
    ctx = Context(cfg)
    out = []
    for e in ld.entities:
        if isinstance(e, cls["OpaqueFunction"]):
            out += e.kwargs["function"](ctx)
        elif not isinstance(e, DeclareLaunchArgument):
            out.append(e)
    return out, cfg


def flatten(actions, cls):
    for a in actions:
        yield a
        if isinstance(a, cls["TimerAction"]):
            yield from flatten(a.kwargs["actions"], cls)


def nodes(actions, cls):
    return {n.executable: n for n in flatten(actions, cls) if isinstance(n, Node)}


def param_dicts(node):
    return [p for p in node.parameters or [] if isinstance(p, dict)]


# ----------------------------------------------------------------------------
# dry runs
# ----------------------------------------------------------------------------
def test_sim_launch_dry_run(tmp_path):
    stubs = _stubs(tmp_path)
    cls = stubs["__classes__"]
    actions, cfg = run_setup("sim.launch.py", stubs)
    share = os.path.join(REPO, "simulation")
    env = [a.args for a in actions if isinstance(a, cls["AppendEnvironmentVariable"])]
    assert ("GZ_SIM_RESOURCE_PATH", os.path.join(share, "models")) in env
    assert ("GZ_SIM_RESOURCE_PATH", os.path.dirname(share)) in env
    inc = [a for a in actions if isinstance(a, IncludeLaunchDescription)]
    assert len(inc) == 1 and inc[0].path.endswith(os.path.join("launch", "gz_sim.launch.py"))
    gz_args = inc[0].launch_args["gz_args"]
    assert gz_args.startswith("-r -v 3 ") and gz_args.split()[3].endswith(os.path.join("worlds", "plant.sdf"))
    if os.path.exists(os.path.join(share, "config", "gui.config")):
        assert "--gui-config" in gz_args
    n = nodes(actions, cls)
    assert set(n) == {"robot_state_publisher", "create", "parameter_bridge", "clock_relay", "odom_tf",
                      "map_server_lite"}
    spawn = n["create"].arguments
    assert spawn[spawn.index("-name") + 1] == "spot" and spawn[spawn.index("-topic") + 1] == "robot_description"
    assert float(spawn[spawn.index("-x") + 1]) == pytest.approx(61.9)
    assert float(spawn[spawn.index("-z") + 1]) == pytest.approx(-0.32)
    assert float(spawn[spawn.index("-Y") + 1]) == pytest.approx(3.14159265, abs=1e-5)
    assert spawn[spawn.index("-world") + 1] == "plant"
    timers = [a for a in actions if isinstance(a, cls["TimerAction"])]
    assert timers and float(timers[0].kwargs["period"]) == pytest.approx(3.0)
    # spawn retry: a failed `create` schedules another attempt, a successful one only logs
    handler = [a for a in timers[0].kwargs["actions"] if isinstance(a, cls["RegisterEventHandler"])][0]
    ope = handler.args[0]
    assert ope.kwargs["target_action"] is n["create"]
    on_exit = ope.kwargs["on_exit"]
    assert isinstance(on_exit(types.SimpleNamespace(returncode=0), None)[0], cls["LogInfo"])
    retry = on_exit(types.SimpleNamespace(returncode=255), None)
    assert isinstance(retry[1], cls["TimerAction"])
    again = [x for x in retry[1].kwargs["actions"] if isinstance(x, Node)][0]
    assert again.executable == "create" and again.arguments == n["create"].arguments
    rd = param_dicts(n["robot_state_publisher"])[0]["robot_description"]
    cmd = rd.args[0].args[0]
    assert cmd[0] == "xacro " and cmd[1].endswith(os.path.join("urdf", "spot.urdf.xacro"))
    assert " enable_thermal:=" in cmd and cmd[cmd.index(" enable_thermal:=") + 1] == "true"
    bridge = param_dicts(n["parameter_bridge"])[0]
    assert bridge["config_file"].endswith(os.path.join("config", "ros_gz_bridge.yaml")) and bridge["use_sim_time"]
    # 1 kHz gz clock -> /clock_gz -> clock_relay -> /clock (200 Hz)
    assert list(n["parameter_bridge"].remappings) == [("/clock", "/clock_gz")]
    relay = n["clock_relay"]
    assert relay.package == "plant_simulation" and relay.name == "clock_relay"
    assert param_dicts(relay)[-1]["use_sim_time"] is False
    assert [p for p in relay.parameters if isinstance(p, str)][0].endswith(os.path.join("config", "mux.yaml"))
    for exe in ("odom_tf", "map_server_lite"):
        files = [p for p in n[exe].parameters if isinstance(p, str)]
        assert files and files[0].endswith(os.path.join("config", "mux.yaml"))
        assert param_dicts(n[exe])[-1]["use_sim_time"] is True


def test_sim_launch_headless_and_overrides(tmp_path):
    stubs = _stubs(tmp_path)
    cls = stubs["__classes__"]
    actions, _ = run_setup("sim.launch.py", stubs, {"gui": "false", "headless_rendering": "true", "x": "1.5",
                                                    "yaw": "0.5", "enable_thermal": "false",
                                                    "use_sim_time": "false", "bridge": "false"})
    inc = [a for a in actions if isinstance(a, IncludeLaunchDescription)][0]
    gz_args = inc.launch_args["gz_args"]
    assert gz_args.startswith("-s -r") and "--headless-rendering" in gz_args and "--gui-config" not in gz_args
    n = nodes(actions, cls)
    assert "parameter_bridge" not in n and "clock_relay" not in n
    spawn = n["create"].arguments
    assert float(spawn[spawn.index("-x") + 1]) == 1.5 and float(spawn[spawn.index("-Y") + 1]) == 0.5
    cmd = param_dicts(n["robot_state_publisher"])[0]["robot_description"].args[0].args[0]
    assert cmd[cmd.index(" enable_thermal:=") + 1] == "false"
    assert param_dicts(n["odom_tf"])[-1]["use_sim_time"] is False


def test_sim_launch_clock_relay_off_bridges_clock_directly(tmp_path):
    stubs = _stubs(tmp_path)
    cls = stubs["__classes__"]
    actions, _ = run_setup("sim.launch.py", stubs, {"clock_relay": "false"})
    n = nodes(actions, cls)
    assert "clock_relay" not in n
    assert not n["parameter_bridge"].remappings


def test_sim_launch_without_ros_gz_sim_runs_gz_directly():
    stubs = _stubs(tmp=None)                       # ros_gz_sim not "installed"
    cls = stubs["__classes__"]
    actions, _ = run_setup("sim.launch.py", stubs)
    procs = [a for a in actions if isinstance(a, cls["ExecuteProcess"])]
    assert procs and procs[0].kwargs["cmd"][:2] == ["gz", "sim"]


def test_bringup_dry_run(tmp_path):
    stubs = _stubs(tmp_path)
    cls = stubs["__classes__"]
    actions, cfg = run_setup("bringup.launch.py", stubs, {"rviz": "true", "thermal_source": "synthetic",
                                                          "mission_autostart": "true", "dashboard_port": "8090"})
    incs = [a for a in actions if isinstance(a, IncludeLaunchDescription)]
    assert len(incs) == 1 and incs[0].path.endswith(os.path.join("launch", "sim.launch.py"))
    assert incs[0].launch_args["enable_thermal"] == "false"          # synthetic thermal camera replaces gz
    timer = [a for a in actions if isinstance(a, cls["TimerAction"])][0]
    assert float(timer.kwargs["period"]) == pytest.approx(6.0)
    mods = timer.kwargs["actions"]
    assert len(mods) == 3
    for (pkg, lf), act in zip(load_launch_module("bringup.launch.py", stubs).MODULE_LAUNCHES, mods):
        if os.path.isfile(os.path.join(REPO, pkg, "launch", lf)):
            assert isinstance(act, IncludeLaunchDescription) and act.path.endswith(lf)
            assert act.launch_args["use_sim_time"] == "true"
            if pkg == "module2_inspection":
                assert act.launch_args["thermal_source"] == "synthetic"
        else:
            assert isinstance(act, cls["LogInfo"])                      # missing module -> logged, not fatal
    n = nodes(actions, cls)
    assert {"cmd_vel_mux", "mission_manager", "monitor_dashboard", "metrics_logger", "rviz2"} <= set(n)
    mm = param_dicts(n["mission_manager"])[-1]
    assert mm["autostart"] is True and mm["autostart_delay"] == pytest.approx(25.0)
    assert param_dicts(n["monitor_dashboard"])[-1]["port"] == 8090
    assert n["rviz2"].arguments[0] == "-d" and n["rviz2"].arguments[1].endswith(os.path.join("rviz", "plant.rviz"))
    files = [p for p in n["mission_manager"].parameters if isinstance(p, str)]
    assert files and files[0].endswith(os.path.join("config", "mission.yaml"))


def test_bringup_rejects_bad_thermal_source(tmp_path):
    with pytest.raises(RuntimeError):
        run_setup("bringup.launch.py", _stubs(tmp_path), {"thermal_source": "infrared"})


def test_bringup_defaults_no_rviz_metrics_on(tmp_path):
    stubs = _stubs(tmp_path)
    actions, cfg = run_setup("bringup.launch.py", stubs)
    n = nodes(actions, stubs["__classes__"])
    assert "rviz2" not in n and "metrics_logger" in n
    assert param_dicts(n["mission_manager"])[-1]["autostart"] is False
    assert cfg["dashboard_port"] == "8080" and cfg["gui"] == "true"


def test_dashboard_launch_dry_run(tmp_path):
    stubs = _stubs(tmp_path)
    m = load_launch_module("dashboard.launch.py", stubs)
    ld = m.generate_launch_description()
    ns = {e.executable: e for e in ld.entities if isinstance(e, Node)}
    assert set(ns) == {"monitor_dashboard", "metrics_logger"}
    assert ns["metrics_logger"].condition is not None


# ----------------------------------------------------------------------------
# module launch files included by bringup (owned by the module packages):
# read-only consistency checks of the integration contract
# ----------------------------------------------------------------------------
MODULE_LAUNCH_PATHS = [(pkg, os.path.join(REPO, pkg, "launch", lf))
                       for pkg, lf in (("module1_locomotion", "locomotion.launch.py"),
                                       ("module2_inspection", "inspection.launch.py"),
                                       ("module3_gas_safety", "gas_safety.launch.py"))]
# arguments every module launch is supposed to accept (DESIGN_SPEC / task contract);
# an undeclared one is harmless for IncludeLaunchDescription, so only use_sim_time is mandatory
COMMON_MODULE_ARGS = {"use_sim_time", "map_yaml", "layout_file"}


def _declared_args(path):
    return set(re.findall(r'DeclareLaunchArgument\(\s*"([A-Za-z0-9_]+)"', open(path, encoding="utf-8").read()))


def _node_specs(path):
    """(package, executable) of every Node(...) / dict(package=..., executable=...) in a launch file."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("Node", "dict"):
            kw = {}
            for k in node.keywords:
                if k.arg in ("package", "executable"):
                    v = k.value
                    kw[k.arg] = v.value if isinstance(v, ast.Constant) else consts.get(getattr(v, "id", None))
            if "package" in kw and "executable" in kw:
                out.append((kw["package"], kw["executable"]))
    return out


@pytest.mark.parametrize("pkg,path", MODULE_LAUNCH_PATHS, ids=[p for p, _ in MODULE_LAUNCH_PATHS])
def test_module_launch_executables_exist(pkg, path, tmp_path):
    if not os.path.isfile(path):
        pytest.skip(f"{path} not written yet")
    _compile(path, tmp_path)
    specs = _node_specs(path)
    assert specs, f"no nodes found in {path}"
    eps = entry_points(REPO_PACKAGES[pkg])
    for p, exe in specs:
        assert p == pkg, f"{path}: node of foreign package {p}"
        assert exe in eps, f"{pkg}/{exe} is started by {os.path.basename(path)} but not in setup.py"


def test_bringup_passes_only_declared_module_args(tmp_path):
    stubs = _stubs(tmp_path)
    cls = stubs["__classes__"]
    actions, _ = run_setup("bringup.launch.py", stubs, {"thermal_source": "synthetic", "teleop_terminal": "xterm"})
    timer = [a for a in actions if isinstance(a, cls["TimerAction"])][0]
    for (pkg, path), act in zip(MODULE_LAUNCH_PATHS, timer.kwargs["actions"]):
        if not os.path.isfile(path):
            continue
        declared = _declared_args(path)
        assert "use_sim_time" in declared, f"{pkg} launch does not declare use_sim_time"
        extra = set(act.launch_args) - COMMON_MODULE_ARGS
        assert extra <= declared, f"bringup passes {extra - declared} which {pkg} does not declare"
    m2 = dict(timer.kwargs["actions"][1].launch_args) if os.path.isfile(MODULE_LAUNCH_PATHS[1][1]) else {}
    if m2:
        assert m2["thermal_source"] == "synthetic"
    m1 = dict(timer.kwargs["actions"][0].launch_args) if os.path.isfile(MODULE_LAUNCH_PATHS[0][1]) else {}
    if m1:
        assert m1["teleop_terminal"] == "xterm"


def test_sim_launch_xacro_args_are_declared_by_the_urdf():
    urdf = os.path.join(SIM, "urdf", "spot.urdf.xacro")
    if not os.path.isfile(urdf):
        pytest.skip("spot.urdf.xacro not written yet")
    declared = set(re.findall(r'<xacro:arg\s+name="([A-Za-z0-9_]+)"', open(urdf, encoding="utf-8").read()))
    src = open(os.path.join(LAUNCH_DIR, "sim.launch.py"), encoding="utf-8").read()
    passed = set(re.findall(r'" ([A-Za-z0-9_]+):="', src))
    assert passed, "sim.launch.py passes no xacro arguments?"
    assert passed <= declared, f"xacro args {passed - declared} are not declared in spot.urdf.xacro"
