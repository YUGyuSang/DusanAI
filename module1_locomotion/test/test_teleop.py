"""Unit tests: keyboard teleop logic (ROS free) + launch file argument logic."""
import importlib.util
import math
import os
import sys
import types

import pytest
import yaml

from module1_locomotion.teleop_keys import (
    CTRL_C, DOWN, ESC, GZ_ESC, KEYMAP, LEFT, QT_KEY_DOWN, QT_KEY_ESCAPE, QT_KEY_LEFT, QT_KEY_RIGHT,
    QT_KEY_UP, RIGHT, UP, CommandStreamer, KeyDecoder, TeleopConfig, TeleopState, format_status,
    qt_key_to_token)

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kinds(events):
    return [(e.kind, e.value) for e in events]


# ------------------------------------------------------------ key decoding
def test_qt_key_codes():
    assert qt_key_to_token(0x57) == "w" and qt_key_to_token(ord("A")) == "a"
    assert qt_key_to_token(0x31) == "1" and qt_key_to_token(0x30) == "0"
    assert qt_key_to_token(0x20) == " "
    assert qt_key_to_token(0x2B) == "+" and qt_key_to_token(0x2D) == "-" and qt_key_to_token(0x3D) == "="
    assert qt_key_to_token(QT_KEY_UP) == UP and qt_key_to_token(QT_KEY_DOWN) == DOWN
    assert qt_key_to_token(QT_KEY_LEFT) == LEFT and qt_key_to_token(QT_KEY_RIGHT) == RIGHT
    assert qt_key_to_token(QT_KEY_ESCAPE) == GZ_ESC
    assert qt_key_to_token(0x01000004) is None and qt_key_to_token("bad") is None
    # every spec key reachable from Gazebo: W A S D Q E X, digits, letters, space, +/-
    for ch in "wasdqex012345rfmpghtobnv +-":
        tok = qt_key_to_token(ord(ch.upper()))
        assert tok in KEYMAP, ch


def test_key_decoder_sequences():
    d = KeyDecoder()
    assert d.feed(b"wasd") == ["w", "a", "s", "d"]
    assert d.feed(b"\x1b[A\x1b[B\x1b[C\x1b[D") == [UP, DOWN, RIGHT, LEFT]
    assert d.feed(b"\x1bOA") == [UP]                       # application cursor mode
    assert d.feed(b"\x1b") == [] and d.pending              # could be the start of a sequence
    assert d.feed(b"[") == [] and d.feed(b"C") == [RIGHT]   # split across reads
    assert d.feed(b"\x1b") == [] and d.flush() == [ESC]     # lone ESC after the timeout
    assert not d.pending and d.flush() == []
    assert d.feed(b"\x03") == [CTRL_C]
    assert d.feed(b"\x1b[3~x") == ["x"]                     # Delete key ignored
    assert d.feed(b"\x1bw") == [ESC, "w"]
    assert d.feed(b"\x1b\x1b") == [ESC] and d.pending
    d.flush()
    assert d.feed(b"\r\n\t +") == [" ", "+"]


# ------------------------------------------------------------ hold / latch
def test_hold_mode_timeout_and_axes():
    st = TeleopState(TeleopConfig(mode="hold", max_vx=0.5, max_vy=0.3, max_wz=0.8, speed_scale=0.5))
    assert st.velocity(0.0) == (0.0, 0.0, 0.0) and not st.streaming(0.0)
    st.handle_key("w", 0.0)
    assert st.velocity(0.0) == pytest.approx((0.25, 0.0, 0.0))
    assert st.velocity(0.55) == pytest.approx((0.25, 0.0, 0.0))     # bridges the repeat delay
    assert st.velocity(0.61) == (0.0, 0.0, 0.0)
    t = 1.0                                                          # OS auto-repeat at 30 Hz
    for k in range(60):
        st.handle_key("w", t + k / 30.0)
        assert st.velocity(t + k / 30.0 + 0.01)[0] == pytest.approx(0.25)
    st.handle_key(DOWN, 3.0)
    assert st.velocity(3.0)[0] == pytest.approx(-0.25)               # opposite key flips
    st.handle_key("a", 3.1)
    st.handle_key("q", 3.2)
    vx, vy, wz = st.velocity(3.25)
    assert vx == pytest.approx(-0.25) and vy == pytest.approx(0.15) and wz == pytest.approx(0.4)
    st.handle_key(RIGHT, 3.3)
    assert st.velocity(3.3)[2] == pytest.approx(-0.4)
    st.handle_key("E", 3.3)                                          # upper case = same key
    assert st.velocity(3.3)[1] == pytest.approx(-0.15)
    st.handle_key("x", 3.4)
    assert st.velocity(3.4) == (0.0, 0.0, 0.0)


def test_latch_mode_increments_and_clamps():
    st = TeleopState(TeleopConfig(mode="latch", max_vx=0.5, speed_scale=1.0, latch_steps=5))
    for _ in range(3):
        st.handle_key("w", 0.0)
    assert st.velocity(100.0)[0] == pytest.approx(0.3)               # persists, no timeout
    st.handle_key("s", 0.0)
    assert st.velocity(0.0)[0] == pytest.approx(0.2)
    for _ in range(20):
        st.handle_key(UP, 0.0)
    assert st.velocity(0.0)[0] == pytest.approx(0.5)                 # clamped at the maximum
    st.handle_key("-", 0.0)                                          # scale applies live
    assert st.velocity(0.0)[0] == pytest.approx(0.45)
    st.handle_key("a", 0.0)
    assert st.velocity(0.0)[2] > 0
    st.handle_key("x", 0.0)
    assert st.velocity(0.0) == (0.0, 0.0, 0.0)


def test_speed_scale_limits():
    st = TeleopState(TeleopConfig(speed_scale=0.5))
    for _ in range(10):
        st.handle_key("+", 0.0)
    assert st.speed_scale == pytest.approx(1.0)
    st.handle_key("_", 0.0)
    assert st.speed_scale == pytest.approx(0.9)
    for _ in range(20):
        st.handle_key("-", 0.0)
    assert st.speed_scale == pytest.approx(0.1)
    st.handle_key("=", 0.0)
    assert st.speed_scale == pytest.approx(0.2)


# --------------------------------------------------------------- commands
def test_estop_toggle():
    st = TeleopState()
    st.handle_key("w", 0.0)
    ev = st.handle_key(" ", 0.1)
    assert kinds(ev) == [("estop", True), ("locomotion", "estop")]
    assert st.velocity(0.1) == (0.0, 0.0, 0.0) and st.streaming(0.1)   # zeros keep mux priority
    ev = st.handle_key("w", 0.2)
    assert ev[0].kind == "info" and st.velocity(0.2) == (0.0, 0.0, 0.0)
    assert kinds(st.handle_key(" ", 0.3)) == [("estop", False), ("locomotion", "release")]
    assert not st.streaming(0.3)


def test_estop_released_elsewhere_clears_the_teleop_estop():
    """Review of fix/resume (stub stack r10): SPACE engaged the E-STOP, the dashboard /
    `ros2 topic pub ... release` released the controller - the teleop kept its flag and
    streamed zero velocity for good (mux 'teleop'), the robot stood while the mission said
    PATROL.  The flag now follows the controller once it confirmed the E-STOP."""
    st = TeleopState()
    st.on_loco_mode("WALK", 0.0)
    assert kinds(st.handle_key(" ", 1.0)) == [("estop", True), ("locomotion", "estop")]
    assert st.on_loco_mode("WALK", 1.05) == [] and st.estop        # not processed yet: never undone
    assert st.on_loco_mode("ESTOP", 1.1) == [] and st.estop
    ev = st.on_loco_mode("STAND", 9.0)                             # released on the dashboard
    assert kinds(ev)[0] == ("estop", False) and ev[1].kind == "info" and "released outside" in ev[1].value
    assert not st.estop and not st.streaming(9.0)
    assert st.on_loco_mode("WALK", 9.1) == []
    # an E-STOP engaged on the dashboard: SPACE releases it instead of engaging it a second time
    st.on_loco_mode("ESTOP", 20.0)
    assert not st.estop
    assert kinds(st.handle_key(" ", 21.0)) == [("estop", False), ("locomotion", "release")]
    # the controller still reports ESTOP right after our own release: a quick second SPACE engages
    assert kinds(st.handle_key(" ", 21.2)) == [("estop", True), ("locomotion", "estop")]
    # FALLEN / RECOVERING are not ESTOP: a SPACE during a recovery engages, the flag stays until confirmed
    st2 = TeleopState()
    st2.on_loco_mode("RECOVERING", 0.0)
    assert kinds(st2.handle_key(" ", 1.0))[0] == ("estop", True)
    assert st2.on_loco_mode("RECOVERING", 1.05) == [] and st2.estop
    assert st2.on_loco_mode(None, 1.1) == [] and st2.estop


def test_locomotion_mission_battery_keys():
    st = TeleopState(TeleopConfig(battery_test_level=19.0, battery_full_level=100.0))
    expected = {"1": "stand", "2": "sit", "3": "gait:trot", "4": "gait:crawl", "5": "gait:auto",
                "r": "recover", "f": "push", "0": "reset_pose"}
    for k, cmd in expected.items():
        assert kinds(st.handle_key(k, 0.0)) == [("locomotion", cmd)]
    for k, cmd in {"p": "patrol", "g": "seek", "h": "rth", "t": "start"}.items():
        assert kinds(st.handle_key(k, 0.0)) == [("mission", cmd)]
    assert kinds(st.handle_key("b", 0.0)) == [("battery", 19.0)]
    assert kinds(st.handle_key("n", 0.0)) == [("battery", 100.0)]
    assert kinds(st.handle_key("?", 0.0)) == [("help", None)]
    assert kinds(st.handle_key(ESC, 0.0)) == [("quit", None)]
    assert kinds(st.handle_key(CTRL_C, 0.0)) == [("quit", None)]
    assert st.handle_key("z", 0.0) == [] and st.handle_key("UNKNOWN", 0.0) == []
    assert kinds(st.handle_key(GZ_ESC, 0.0)) == [("info", "stop")]   # ESC in Gazebo = stop, not quit


def test_pause_resume_toggle():
    st = TeleopState()
    assert kinds(st.handle_key("o", 0.0)) == [("mission", "pause")]   # no mission state: local toggle
    assert kinds(st.handle_key("o", 0.0)) == [("mission", "resume")]
    st.mission_state = "PAUSED"
    assert kinds(st.handle_key("o", 0.0)) == [("mission", "resume")]
    st.mission_state = "PATROL"
    assert kinds(st.handle_key("o", 0.0)) == [("mission", "pause")]


def test_manual_override_streaming():
    st = TeleopState()
    assert kinds(st.handle_key("m", 0.0)) == [("override", True)]
    assert st.streaming(100.0) and st.velocity(100.0) == (0.0, 0.0, 0.0)
    assert kinds(st.handle_key("m", 0.0)) == [("override", False)]
    assert not st.streaming(100.0)


def test_command_streamer_rate_and_single_zero():
    st = TeleopState(TeleopConfig(mode="hold", hold_timeout=0.6))
    cs = CommandStreamer(20.0)
    out = []
    st.handle_key("w", 0.0)
    for k in range(200):                     # 2 s at 100 Hz loop rate
        t = k * 0.01
        cmd = cs.step(t, st.streaming(t), st.velocity(t))
        if cmd is not None:
            out.append((t, cmd))
    moving = [c for t, c in out if c[0] > 0]
    zeros = [c for t, c in out if c == (0.0, 0.0, 0.0)]
    assert 11 <= len(moving) <= 14           # 0.6 s at 20 Hz
    assert len(zeros) == 1                   # exactly one zero on stop, then silence
    assert out[-1][1] == (0.0, 0.0, 0.0) and out[-1][0] < 0.75
    gaps = [b[0] - a[0] for a, b in zip(out[:-1], out[1:])]
    assert all(g >= 0.035 for g in gaps)                 # never faster than the rate (loop granularity)
    assert sum(gaps) / len(gaps) == pytest.approx(0.05, abs=0.003)


def test_status_line_and_config():
    st = TeleopState(TeleopConfig(speed_scale=0.7))
    st.handle_key("m", 0.0)
    st.handle_key(" ", 0.0)
    line = format_status(st, 0.0, "WALK", "trot", 87.4, "PATROL", gz_keys=3)
    for s in ("x0.70", "MANUAL", "E-STOP", "loco WALK/trot", "batt 87%", "mission PATROL", "gz 3"):
        assert s in line
    assert len(line) <= 110
    assert len(format_status(st, 0.0, "WALK", width=40)) == 40
    assert "batt ?" in format_status(TeleopState(), 0.0, battery_pct=float("nan"))
    assert "loco WALK/trot [stable]" in format_status(TeleopState(), 0.0, "WALK", "trot", loco_algo="stable")


def test_locomotion_mode_key():
    st = TeleopState()
    assert kinds(st.handle_key("v", 0.0)) == [("locomotion", "mode:toggle")]
    assert kinds(st.handle_key("V", 0.0)) == [("locomotion", "mode:toggle")]    # Shift / CapsLock
    cfg = TeleopConfig.from_dict({"mode": "sideways", "max_vx": 1, "latch_steps": 4.0, "x": 3})
    assert cfg.mode == "hold" and cfg.max_vx == 1.0 and isinstance(cfg.max_vx, float)
    assert cfg.latch_steps == 4 and isinstance(cfg.latch_steps, int)
    assert TeleopConfig.from_dict({"mode": "latch"}).mode == "latch"


def test_teleop_yaml_matches_config():
    with open(os.path.join(PKG_DIR, "config", "teleop.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["keyboard_teleop"]["ros__parameters"]
    d = TeleopConfig()
    for k in ("mode", "max_vx", "max_vy", "max_wz", "speed_scale", "scale_step", "min_scale",
              "max_scale", "hold_timeout", "latch_steps", "battery_test_level", "battery_full_level"):
        assert type(cfg[k]) is type(getattr(d, k)), k
        assert cfg[k] == getattr(d, k), k
    assert cfg["publish_rate"] == 20.0 and cfg["hold_timeout"] == 0.6


# ------------------------------------------------------- launch file logic
class _Sub:
    """Minimal stand-in for launch substitutions / actions (records kwargs)."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def perform(self, context):
        return context.launch_configurations[self.args[0]]


def _stub_modules():
    mods = {}
    for name in ("launch", "launch.actions", "launch.conditions", "launch.substitutions",
                 "launch_ros", "launch_ros.actions", "launch_ros.substitutions"):
        mods[name] = types.ModuleType(name)

    class LaunchDescription(list):
        def __init__(self, entities):
            super().__init__(entities)

    mods["launch"].LaunchDescription = LaunchDescription
    for n in ("DeclareLaunchArgument", "LogInfo", "OpaqueFunction"):
        setattr(mods["launch.actions"], n, type(n, (_Sub,), {}))
    mods["launch.conditions"].IfCondition = type("IfCondition", (_Sub,), {})
    for n in ("LaunchConfiguration", "PathJoinSubstitution"):
        setattr(mods["launch.substitutions"], n, type(n, (_Sub,), {}))
    mods["launch_ros.actions"].Node = type("Node", (_Sub,), {})
    mods["launch_ros.substitutions"].FindPackageShare = type("FindPackageShare", (_Sub,), {})
    return mods


def test_launch_file_structure(monkeypatch):
    for name, mod in _stub_modules().items():
        monkeypatch.setitem(sys.modules, name, mod)
    # never leave launch/__pycache__ behind (setup.py installs the whole launch/ tree)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    path = os.path.join(PKG_DIR, "launch", "locomotion.launch.py")
    spec = importlib.util.spec_from_file_location("locomotion_launch_under_test", path)
    lmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lmod)
    ld = lmod.generate_launch_description()
    declared = {e.args[0]: e.kwargs.get("default_value") for e in ld if type(e).__name__ == "DeclareLaunchArgument"}
    assert declared["use_sim_time"] == "true" and declared["teleop_terminal"] == "none"
    nodes = {e.kwargs["executable"]: e.kwargs for e in ld if type(e).__name__ == "Node"}
    assert nodes["terrain_mapper"]["name"] == "terrain_mapper"
    ctl = nodes["locomotion_controller"]
    assert ctl["name"] == "locomotion_controller" and ctl["package"] == "module1_locomotion"
    cfg_path = ctl["parameters"][0].args[0]          # LaunchConfiguration('controller_config')
    assert cfg_path == "controller_config"
    pj = declared["controller_config"]
    assert pj.args[0][1:] == ["config", "locomotion.yaml"]
    opaque = [e for e in ld if type(e).__name__ == "OpaqueFunction"][0]
    fn = opaque.kwargs["function"]

    def ctx(term, sim="true"):
        c = types.SimpleNamespace()
        c.launch_configurations = {"teleop_terminal": term, "use_sim_time": sim, "teleop_mode": "latch",
                                   "teleop_config": "/tmp/teleop.yaml"}
        return c

    assert fn(ctx("none")) == []
    acts = fn(ctx("xterm"))
    node = [a for a in acts if type(a).__name__ == "Node"][0]
    assert node.kwargs["prefix"] == "xterm -e" and node.kwargs["executable"] == "keyboard_teleop"
    assert node.kwargs["parameters"][1] == {"use_sim_time": True, "mode": "latch"}
    node = [a for a in fn(ctx("gz", "false")) if type(a).__name__ == "Node"][0]
    assert "prefix" not in node.kwargs and node.kwargs["parameters"][1]["use_sim_time"] is False
    node = [a for a in fn(ctx("tilix -e")) if type(a).__name__ == "Node"][0]
    assert node.kwargs["prefix"] == "tilix -e"
    assert all(type(a).__name__ != "Node" for a in fn(ctx("bogus")))


def test_node_modules_compile_and_define_main(tmp_path):
    import py_compile
    for mod in ("terrain_mapping_node", "keyboard_teleop"):
        path = os.path.join(PKG_DIR, "module1_locomotion", mod + ".py")
        py_compile.compile(path, cfile=str(tmp_path / (mod + ".pyc")), doraise=True)
        with open(path, encoding="utf-8") as f:
            assert "\ndef main(" in f.read()


# ------------------------------------------------- node wrapper (fake rclpy)
def _fake_ros_helpers():
    """The fake rclpy / message layer lives in test_elevation_map.py (loaded
    by path so it works whatever the pytest import mode is)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_elevation_map.py")
    spec = importlib.util.spec_from_file_location("_m1_fake_ros_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_m1_fake_ros_helpers", mod)
    spec.loader.exec_module(mod)
    return mod


def test_keyboard_teleop_node_with_fake_ros(monkeypatch):
    import json
    helpers = _fake_ros_helpers()
    env = helpers.FakeRosEnv(monkeypatch, overrides={"max_vx": 1,          # int for a float parameter
                                                     "speed_scale": 0.5, "use_terminal": False})
    node_mod = importlib.import_module("module1_locomotion.keyboard_teleop")
    clock = types.SimpleNamespace(t=1000.0)               # controllable wall clock of the node
    monkeypatch.setattr(node_mod, "time", types.SimpleNamespace(monotonic=lambda: clock.t,
                                                                sleep=lambda s: None))
    node = node_mod.KeyboardTeleop()
    assert node.use_tty is False                         # no TTY -> Gazebo-keys-only mode
    assert node.state.cfg.max_vx == 1.0
    assert {t: p.type.__name__ for t, p in node.pubs.items()} == {
        "/cmd_vel/teleop": "Twist", "/teleop/active": "Bool", "/locomotion/command": "String",
        "/mission/command": "String", "/battery/set_level": "Float32"}
    assert {t: s[0].__name__ for t, s in node.subs.items()} == {
        "/locomotion/state": "String", "/battery/state": "BatteryState", "/mission/state": "String",
        "/keyboard/keypress": "Int32"}
    Int32 = env.msg["std_msgs.msg"].Int32
    String = env.msg["std_msgs.msg"].String
    key = lambda code: node.subs["/keyboard/keypress"][1](Int32(data=code))
    cmd, active = node.pubs["/cmd_vel/teleop"], node.pubs["/teleop/active"]

    node.tick(clock.t)                                   # idle: no twist, heartbeat of /teleop/active
    assert cmd.msgs == [] and active.msgs[-1].data is False
    key(ord("W"))                                        # Qt key code of 'w' (upper-case ASCII)
    for k in range(25):                                  # 0.5 s of 50 Hz ticks, key held (no repeat yet)
        clock.t += 0.02
        node.tick(clock.t)
    assert 9 <= len(cmd.msgs) <= 11                      # ~20 Hz while driving
    assert cmd.msgs[-1].linear.x == pytest.approx(0.5) and cmd.msgs[-1].angular.z == 0.0
    # released: after the 0.6 s hold timeout exactly one zero twist, then silence
    for k in range(25):
        clock.t += 0.02
        node.tick(clock.t)
    zeros = [m for m in cmd.msgs if m.linear.x == 0.0]
    assert cmd.msgs[-1].linear.x == 0.0 and len(zeros) == 1
    # command keys -> topics
    key(ord("3"))
    key(ord("T"))
    key(ord("B"))
    key(0x20)                                            # space = E-STOP
    assert [m.data for m in node.pubs["/locomotion/command"].msgs] == ["gait:trot", "estop"]
    assert [m.data for m in node.pubs["/mission/command"].msgs] == ["start"]
    assert node.pubs["/battery/set_level"].msgs[-1].data == pytest.approx(19.0)
    key(0x20)
    assert node.pubs["/locomotion/command"].msgs[-1].data == "release"
    key(ord("M"))                                        # manual override -> /teleop/active true
    assert active.msgs[-1].data is True
    # status inputs
    node.subs["/locomotion/state"][1](String(data=json.dumps({"mode": "WALK", "gait": "crawl",
                                                               "gait_auto": True,
                                                               "locomotion_mode": "stable"})))
    node.subs["/battery/state"][1](env.msg["sensor_msgs.msg"].BatteryState(percentage=0.42))
    node.subs["/mission/state"][1](String(data=json.dumps({"state": "PAUSED"})))
    assert node.loco_mode == "WALK" and node.loco_gait == "crawl(auto)" and node.loco_algo == "stable"
    assert node.battery_pct == pytest.approx(42.0) and node.state.mission_state == "PAUSED"
    key(ord("O"))                                        # pause/resume toggles on the mission state
    assert node.pubs["/mission/command"].msgs[-1].data == "resume"
    key(0x01000000)                                      # Escape in Gazebo = stop, never quit
    assert node.quit is False
    # shutdown path: zero twist + override released
    node.stop_robot()
    assert cmd.msgs[-1].linear.x == 0.0 and active.msgs[-1].data is False


def test_keyboard_teleop_node_follows_an_estop_release_from_the_dashboard(monkeypatch):
    """Node level: SPACE in the Gazebo window (KeyPublisher) -> E-STOP, zero stream on
    /cmd_vel/teleop; the controller is released by the dashboard -> /locomotion/state
    leaves ESTOP -> the teleop stops streaming (one zero, then silence: the mux falls back
    to the mission)."""
    import json
    helpers = _fake_ros_helpers()
    helpers.FakeRosEnv(monkeypatch, overrides={"use_terminal": False})
    node_mod = importlib.import_module("module1_locomotion.keyboard_teleop")
    clock = types.SimpleNamespace(t=500.0)
    monkeypatch.setattr(node_mod, "time", types.SimpleNamespace(monotonic=lambda: clock.t,
                                                                sleep=lambda s: None))
    node = node_mod.KeyboardTeleop()
    Int32 = node_mod.Int32
    String = node_mod.String
    loco = lambda mode: node.subs["/locomotion/state"][1](String(data=json.dumps({"mode": mode})))  # noqa: E731
    cmd = node.pubs["/cmd_vel/teleop"]

    def ticks(seconds):
        for _ in range(int(round(seconds / 0.02))):
            clock.t += 0.02
            node.tick(clock.t)
    loco("WALK")
    node.subs["/keyboard/keypress"][1](Int32(data=0x20))
    assert node.pubs["/locomotion/command"].msgs[-1].data == "estop" and node.state.estop
    ticks(0.04)
    loco("WALK")                                       # controller has not processed it yet
    ticks(0.5)
    loco("ESTOP")
    ticks(1.0)
    n = len(cmd.msgs)
    assert n >= 25 and all(m.linear.x == 0.0 for m in cmd.msgs)   # 20 Hz zero stream = mux priority
    loco("STAND")                                      # dashboard '비상정지 해제'
    assert not node.state.estop
    ticks(1.0)
    assert len(cmd.msgs) == n + 1 and cmd.msgs[-1].linear.x == 0.0   # one zero, then silence
    assert node.pubs["/locomotion/command"].msgs[-1].data == "estop"  # nothing sent by the sync itself


def test_rclpy_init_without_signal_handlers(monkeypatch):
    """main() initialises rclpy without its SIGINT handler when supported (so
    Ctrl-C still publishes the final zero twist / override release) and
    falls back to the plain call on an rclpy without SignalHandlerOptions."""
    helpers = _fake_ros_helpers()
    helpers.FakeRosEnv(monkeypatch)
    node_mod = importlib.import_module("module1_locomotion.keyboard_teleop")
    calls = []
    sig_mod = types.ModuleType("rclpy.signals")
    sig_mod.SignalHandlerOptions = types.SimpleNamespace(NO="NO", ALL="ALL")
    monkeypatch.setitem(sys.modules, "rclpy.signals", sig_mod)
    monkeypatch.setattr(node_mod.rclpy, "init", lambda args=None, **kw: calls.append(kw))
    assert node_mod._init_rclpy(None) is True and calls[-1] == {"signal_handler_options": "NO"}

    def old_init(args=None):                         # rclpy without the keyword
        calls.append("plain")
    monkeypatch.setattr(node_mod.rclpy, "init", old_init)
    assert node_mod._init_rclpy(None) is False and calls[-1] == "plain"
    monkeypatch.delitem(sys.modules, "rclpy.signals")
    assert node_mod._init_rclpy(None) is False and calls[-1] == "plain"


def test_mission_label_marks_a_silent_mission_manager():
    """'t' answered by nobody (mission_manager not running / crashed) must be
    visible in the status line instead of the last known state."""
    from module1_locomotion.teleop_keys import mission_label, mission_stale
    assert mission_label(None, None, 10.0) is None
    assert mission_label("IDLE", 9.0, 10.0) == "IDLE"
    assert mission_label("PATROL", 2.0, 10.0) == "PATROL(stale 8s)"
    assert mission_stale(None, 10.0) and mission_stale(2.0, 10.0) and not mission_stale(9.0, 10.0)
    st = TeleopState(TeleopConfig())
    line = format_status(st, 10.0, mission=mission_label("IDLE", 1.0, 10.0))
    assert "mission IDLE(stale 9s)" in line
