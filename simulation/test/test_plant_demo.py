"""Demo mode (reference video): HUD compositor (ROS-free), demo_hud node against
the strict rclpy stand-ins, follow_camera helper (fake gz CLI), bringup demo
arguments, and the gui.config / bridge / script / particle colour contracts."""
import os
import re
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

import test_plant_launch_files as tl
from dusan_common.layout import Layout
from plant_simulation import demo_hud as dh
from plant_simulation import follow_camera as fc
from plant_simulation.mission_logic import MISSION_DEFAULTS

HERE = os.path.dirname(os.path.realpath(__file__))
SIM = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(SIM, ".."))

FULL = {"gas_ppm": 611.8, "gas_alarm": True, "battery_pct": 76.0, "charging": False, "loco_mode": "WALK",
        "locomotion_mode": "stable", "gait": "trot", "mission_state": "SEEK", "mission_target": "",
        "mission_progress": [5, 8], "sim_time": 312.4}


@pytest.fixture(scope="module")
def ascii_renderer():
    return dh.TextRenderer(language="en")


def camera(h=600, w=800, value=120):
    img = np.full((h, w, 3), value, np.uint8)
    img[h // 3:2 * h // 3, w // 3:2 * w // 3] = (30, 160, 220)
    return img


# ============================================================================
# compositor (ROS-free)
# ============================================================================
def test_hud_defaults_match_the_mission_alarm_threshold():
    assert dh.HUD_DEFAULTS["gas_alarm_ppm"] == MISSION_DEFAULTS["gas_alarm_ppm"]
    assert dh.HUD_DEFAULTS["gas_warn_ppm"] < dh.HUD_DEFAULTS["gas_alarm_ppm"]
    assert dh.HUD_DEFAULTS["hud_topic"] == "/demo/hud"
    assert set(dh.EMPTY_STATUS) >= {"gas_ppm", "gas_alarm", "battery_pct", "mission_state", "gait"}


@pytest.mark.parametrize("ppm,flag,level", [(None, False, "unknown"), (float("nan"), False, "unknown"),
                                            (0.0, False, "normal"), (9.9, False, "normal"),
                                            (10.0, False, "warning"), (24.9, False, "warning"),
                                            (25.0, False, "alarm"), (3.0, True, "alarm"), (None, True, "alarm")])
def test_gas_level(ppm, flag, level):
    assert dh.gas_level(ppm, 10.0, 25.0, flag) == level


def test_gas_bar_is_log_scaled_and_clamped():
    f = [dh.gas_bar_fraction(c, 1000.0) for c in (0, 1, 10, 100, 1000, 5000)]
    assert f[0] == 0.0 and f[-1] == 1.0 and f[-2] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(f[:-2], f[1:-1]))
    assert dh.gas_bar_fraction(25.0, 1000.0) == pytest.approx(np.log10(26) / np.log10(1001))
    assert dh.gas_bar_fraction(None, 1000.0) == 0.0 and dh.gas_bar_fraction(-3, 1000.0) == 0.0


def test_texts_korean_and_english():
    assert dh.gas_text(611.84, True) == "현재 가스 농도: 611.8 ppm"
    assert dh.gas_text(None, False) == "GAS: -- ppm"
    assert dh.mission_text("PATROL", "G03", [2, 8], True) == "임무: 순찰 이동 → G03 (2/8)"
    assert dh.mission_text("SEEK", "G03", [2, 8], False) == "MISSION: GAS SEARCH (2/8)"   # target only while patrolling
    assert dh.mission_text("", korean=False) == "MISSION: --"
    assert dh.mission_text("WEIRD", korean=False) == "MISSION: WEIRD"
    assert dh.locomotion_text("WALK", "stable", "trot", True) == "보행: WALK · stable · trot"
    assert dh.locomotion_text(korean=False) == "LOCOMOTION: --"
    assert dh.battery_text(76.4) == "76%" and dh.battery_text(None) == "--%"
    for st in ("IDLE", "PATROL", "INSPECTING", "SEEK", "RTH", "DOCKED", "MANUAL", "PAUSED", "DONE"):
        assert st in dh.MISSION_LABELS                        # every mission_logic state has a label
    from plant_simulation.mission_logic import STATES
    assert set(STATES) <= set(dh.MISSION_LABELS)


@pytest.mark.parametrize("size", [(640, 480), (320, 240), (800, 600), (1280, 720)])
def test_compose_shapes(size, ascii_renderer):
    w, h = size
    out = dh.compose_hud(camera(), FULL, {"width": w, "height": h}, ascii_renderer, t=0.1)
    assert out.shape == (h, w, 3) and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]


def test_compose_keeps_the_camera_visible_and_letterboxes(ascii_renderer):
    cam = camera(480, 640)
    out = dh.compose_hud(cam, {}, renderer=ascii_renderer)
    # centre of the picture (between the bars) is the camera image itself
    assert np.array_equal(out[240, 320], cam[240, 320])
    wide = dh.compose_hud(camera(300, 1200), {}, renderer=ascii_renderer)
    assert wide[150, 320].tolist() == [0, 0, 0] or wide[100, 320].tolist() == [0, 0, 0]   # black bars
    mono = dh.compose_hud(np.full((60, 80), 200, np.uint8), {}, renderer=ascii_renderer)
    assert mono.shape == (480, 640, 3) and mono[240, 320].tolist() == [200, 200, 200]


def test_compose_without_camera_shows_placeholder(ascii_renderer):
    out = dh.compose_hud(None, {}, renderer=ascii_renderer)
    assert out.shape == (480, 640, 3)
    assert out[150, 100].tolist() == list(dh.DARK)
    empty = dh.compose_hud(np.zeros((0, 0, 3), np.uint8), {}, renderer=ascii_renderer)
    assert empty[150, 100].tolist() == list(dh.DARK)


def test_gas_bar_colour_follows_the_level(ascii_renderer):
    for ppm, level in ((3.0, "normal"), (15.0, "warning"), (600.0, "alarm")):
        out = dh.compose_hud(camera(), {"gas_ppm": ppm}, renderer=ascii_renderer)
        assert out[54, 14].tolist() == list(dh.COLORS[level]), level


def test_alarm_frame_blinks(ascii_renderer):
    red = list(dh.COLORS["alarm"])
    on = dh.compose_hud(camera(), FULL, renderer=ascii_renderer, t=0.1)          # int(0.2) % 2 == 0 -> on
    off = dh.compose_hud(camera(), FULL, renderer=ascii_renderer, t=0.6)         # int(1.2) % 2 == 1 -> off
    calm = dh.compose_hud(camera(), dict(FULL, gas_alarm=False, gas_ppm=2.0), renderer=ascii_renderer, t=0.1)
    assert on[240, 1].tolist() == red and on[240, 638].tolist() == red
    assert off[240, 1].tolist() != red and calm[240, 1].tolist() != red
    # the badge is drawn in both phases (darker while the frame is off)
    assert (on[90:120, 300:340] == red).all(axis=2).any()
    assert not (calm[90:120, 300:340] == red).all(axis=2).any()


def test_ascii_fallback_without_pillow(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)          # `import PIL` raises ImportError
    r = dh.TextRenderer(language="ko")
    assert not r.korean and not r.pillow and "Pillow" in r.reason
    out = dh.compose_hud(camera(), FULL, renderer=r, t=0.1)
    assert out.shape == (480, 640, 3)
    w, h = r.measure("현재", 20)                             # non-ASCII is replaced, not a crash
    assert w > 0 and h == 20


def test_missing_font_path_falls_back(tmp_path):
    r = dh.TextRenderer(font_path=str(tmp_path / "nope.ttf"), language="ko")
    if r.pillow:                                            # another Korean font was found on this machine
        assert r.font_file and os.path.isfile(r.font_file)
    else:
        assert not r.korean and r.reason


def test_korean_rendering_when_a_font_is_available():
    r = dh.TextRenderer(language="ko")
    if not r.korean:
        pytest.skip(f"no Pillow / Korean font here: {r.reason}")
    out = dh.compose_hud(camera(), FULL, renderer=r, t=0.1)
    # gas text is drawn in the alarm colour in the top bar
    top = out[8:40, 12:330]
    assert (np.abs(top.astype(int) - np.array(dh.COLORS["alarm"])).sum(axis=2) < 40).sum() > 200


def test_hud_compositor_wrapper(ascii_renderer):
    h = dh.HudCompositor({"width": 320, "height": 240}, renderer=ascii_renderer)
    assert h.compose(None, {}).shape == (240, 320, 3)


# ============================================================================
# demo_hud node (strict rclpy stand-ins of test_plant_nodes_stubbed)
# ============================================================================
import test_plant_nodes_stubbed as stub  # noqa: E402

ros = stub.ros                                             # fixture


def _image(w=800, h=600, enc="rgb8"):
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = 200                                      # red in rgb8
    return stub.Image(height=h, width=w, encoding=enc, step=3 * w, data=img.tobytes())


@pytest.fixture()
def hud_node(ros, monkeypatch):
    sys.modules.pop("plant_simulation.demo_hud_node", None)
    import importlib
    mod = importlib.import_module("plant_simulation.demo_hud_node")
    node = mod.DemoHudNode()
    yield ros, node
    sys.modules.pop("plant_simulation.demo_hud_node", None)


def test_demo_hud_node_publishes_bgr_and_rgb(hud_node):
    W, node = hud_node
    assert {s.topic for s in W.subs} >= {"/inspection_camera/image", "/gas/sensor", "/battery/state",
                                         "/locomotion/state", "/mission/state"}
    W.deliver("/inspection_camera/image", _image())
    W.deliver("/gas/sensor", stub.Float32(data=611.8))
    W.deliver("/battery/state", stub.BatteryState(percentage=0.764, power_supply_status=2))
    W.deliver("/locomotion/state", stub.String(data='{"mode": "WALK", "locomotion_mode": "stable", "gait": "trot"}'))
    W.deliver("/mission/state", stub.String(data='{"state": "SEEK", "target": "", "progress": [5, 8], '
                                                 '"gas_alarm": true}'))
    st = node.status()
    assert st["gas_ppm"] == pytest.approx(611.8, abs=1e-3) and st["battery_pct"] == pytest.approx(76.4)
    assert st["charging"] is False and st["gait"] == "trot" and st["mission_state"] == "SEEK" and st["gas_alarm"]
    W.tick(1, dt=0.2)
    bgr = W.published("/demo/hud")
    rgb = W.published("/demo/hud_rgb")
    assert len(bgr) == 1 and len(rgb) == 1
    m = bgr[0]
    assert (m.encoding, m.width, m.height, m.step) == ("bgr8", 640, 480, 640 * 3)
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(480, 640, 3)
    b = np.frombuffer(bytes(rgb[0].data), np.uint8).reshape(480, 640, 3)
    assert rgb[0].encoding == "rgb8" and np.array_equal(a[..., ::-1], b)
    assert a[240, 320].tolist() == [0, 0, 200]             # the camera's red, as BGR


def test_demo_hud_node_stale_inputs(hud_node, monkeypatch):
    W, node = hud_node
    W.deliver("/gas/sensor", stub.Float32(data=12.0))
    W.deliver("/battery/state", stub.BatteryState(percentage=55.0, power_supply_status=1))   # 0..100 tolerated
    st = node.status()
    assert st["battery_pct"] == pytest.approx(55.0) and st["charging"] is True
    import time as _time
    later = _time.monotonic() + node.p["status_timeout"] + 1.0
    st = node.status(now=later)
    assert st["gas_ppm"] is None and "battery_pct" not in st
    assert node._camera_bgr(later) is None
    W.tick(1, dt=0.2)                                       # still publishes (placeholder frame)
    assert len(W.published("/demo/hud")) == 1


def test_demo_hud_node_bad_camera_encoding_is_not_fatal(hud_node):
    W, node = hud_node
    W.deliver("/inspection_camera/image", _image(enc="yuv422"))
    W.tick(1, dt=0.2)
    assert len(W.published("/demo/hud")) == 1
    assert any("cannot decode" in line for _, line in node.get_logger().lines)


# ============================================================================
# follow_camera helper
# ============================================================================
ECHO_FOLLOW = 'track_mode: FOLLOW\nfollow_target {\n  name: "spot"\n}\nfollow_offset {\n  x: -2.5\n  z: 1.6\n}\n'


def test_follow_camera_messages():
    assert fc.parse_offset("-2.5,0,1.6") == (-2.5, 0.0, 1.6) == fc.parse_offset(" -2.5  0 1.6 ")
    with pytest.raises(ValueError):
        fc.parse_offset("1,2")
    msg = fc.track_message("spot", (-2.5, 0.0, 1.6), 0.02)
    assert msg == ('track_mode: FOLLOW, follow_target: {name: "spot"}, follow_offset: {x: -2.5, y: 0, z: 1.6}, '
                   'follow_pgain: 0.02')
    cmd = fc.track_command("gz", "spot", (-2.5, 0.0, 1.6), 0.02)
    assert cmd[:7] == ["gz", "topic", "-t", "/gui/track", "-m", "gz.msgs.CameraTrack", "-p"] and cmd[7] == msg
    follow, offset = fc.follow_service_commands("gz", "spot", (-2.5, 0.0, 1.6))
    assert follow[follow.index("-s") + 1] == "/gui/follow" and follow[-1] == 'data: "spot"'
    assert offset[offset.index("-s") + 1] == "/gui/follow/offset" and offset[-1] == "x: -2.5, y: 0, z: 1.6"
    assert "gz.msgs.Vector3d" in offset and "gz.msgs.StringMsg" in follow
    assert fc.robot_topics("spot", "plant") == ("/model/spot/odometry", "/world/plant/model/spot/joint_state")


def test_is_following():
    assert fc.is_following(ECHO_FOLLOW, "spot")
    assert not fc.is_following(ECHO_FOLLOW, "other")
    assert not fc.is_following('follow_target {\n  name: "spot"\n}\n', "spot")      # NONE is omitted (proto3)
    assert not fc.is_following("track_mode: TRACK\n", "spot")
    assert not fc.is_following("", "spot")


class FakeGz:
    """Scripted gz CLI: topics appear after `ready_after` polls."""

    def __init__(self, ready_after=2, gui=True, confirm=True, services=("/gui/follow",)):
        self.calls, self.polls, self.t = [], 0, 0.0
        self.ready_after, self.gui, self.confirm_ok, self.services = ready_after, gui, confirm, services

    def run(self, cmd, timeout):
        self.calls.append(cmd)
        args = cmd[1:]
        if args[:2] == ["topic", "-l"]:
            self.polls += 1
            if self.polls < self.ready_after:
                return 0, "/clock\n/stats\n"
            tp = ["/clock", "/model/spot/odometry"] + (["/gui/camera/pose", "/gui/track"] if self.gui else [])
            return 0, "\n".join(tp)
        if args[:2] == ["topic", "-e"]:
            return (0, ECHO_FOLLOW) if self.confirm_ok else (-1, "")
        if args[:2] == ["topic", "-t"]:
            return 0, ""
        if args[:2] == ["service", "-l"]:
            return 0, "\n".join(self.services)
        if args[:1] == ["service"]:
            return 0, "data: true"
        return 1, "?"

    def clock(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def _fc(g, **kw):
    lines = []
    f = fc.FollowCamera("gz", "spot", "plant", (-2.5, 0.0, 1.6), 0.02, runner=g.run, clock=g.clock, sleep=g.sleep,
                        log=lines.append, **kw)
    return f, lines


def test_follow_camera_engages_via_gui_track():
    g = FakeGz(ready_after=3)
    f, lines = _fc(g, timeout=60.0)
    assert f.main() == 0
    sent = [c for c in g.calls if c[1:3] == ["topic", "-t"]]
    assert len(sent) == 1 and sent[0][3] == "/gui/track"
    assert g.polls == 3 and "via /gui/track" in lines[-1]
    assert not [c for c in g.calls if c[1] == "service"]


def test_follow_camera_falls_back_to_gui_follow_service():
    g = FakeGz(ready_after=1, confirm=False)
    f, lines = _fc(g, timeout=60.0, attempts=3)
    assert f.main() == 0
    assert len([c for c in g.calls if c[1:3] == ["topic", "-t"]]) == 3
    svc = [c[c.index("-s") + 1] for c in g.calls if c[1] == "service" and "-s" in c]
    assert svc == ["/gui/follow", "/gui/follow/offset"] and "via /gui/follow" in lines[-1]


def test_follow_camera_without_gui_or_robot_gives_up_quietly():
    g = FakeGz(ready_after=1, gui=False)
    f, lines = _fc(g, timeout=10.0, poll=2.0)
    assert f.main() == 0 and "no Gazebo GUI" in lines[-1]
    assert not [c for c in g.calls if c[1:3] == ["topic", "-t"]]
    g2 = FakeGz(ready_after=10 ** 6)
    f2, lines2 = _fc(g2, timeout=10.0)
    assert f2.main() == 0 and "not found" in lines2[-1]

    class NoGz(FakeGz):
        def run(self, cmd, timeout):
            return -1, "not found"
    g3 = NoGz()
    f3, lines3 = _fc(g3, timeout=4.0)
    assert f3.main() == 0 and "no Gazebo topics" in lines3[-1]
    g4 = FakeGz(ready_after=1, confirm=False, services=())
    f4, lines4 = _fc(g4, timeout=10.0, attempts=2)
    assert f4.main() == 0 and "could not confirm" in lines4[-1]


def test_follow_camera_main_never_fails(monkeypatch, capsys):
    monkeypatch.setattr(fc.shutil, "which", lambda name: None)
    assert fc.main(["--target", "spot", "--offset=-2,0,1", "--ros-args", "-r", "__node:=follow_camera"]) == 0
    assert "not on PATH" in capsys.readouterr().out
    assert fc.main(["--offset=1,2"]) == 0                       # bad offset -> logged, exit 0
    assert "bad arguments" in capsys.readouterr().out
    assert fc.main(["--no-such-flag"]) == 0
    seen = {}

    class Rec(fc.FollowCamera):
        def main(self):
            seen.update(target=self.target, world=self.world, offset=self.offset, pgain=self.pgain)
            return 0
    monkeypatch.setattr(fc, "FollowCamera", Rec)
    assert fc.main(["--gz", "gz", "--target", "robo", "--world", "w", "--offset=-3,0.5,2", "--pgain", "0.05",
                    "--ros-args", "-r", "__node:=x"]) == 0
    assert seen == {"target": "robo", "world": "w", "offset": (-3.0, 0.5, 2.0), "pgain": 0.05}


# ============================================================================
# bringup demo arguments / entry points
# ============================================================================
def test_new_entry_points_exist():
    eps = tl.entry_points("simulation")
    assert {"demo_hud", "follow_camera"} <= eps
    src = open(os.path.join(SIM, "setup.py"), encoding="utf-8").read()
    assert '"demo_hud = plant_simulation.demo_hud_node:main"' in src
    assert '"follow_camera = plant_simulation.follow_camera:main"' in src
    for mod in ("demo_hud_node.py", "follow_camera.py", "demo_hud.py"):
        text = open(os.path.join(SIM, "plant_simulation", mod), encoding="utf-8").read()
        assert "\ndef main(" in text or mod == "demo_hud.py"
    assert callable(fc.main)


def test_bringup_defaults_start_hud_but_no_follow_camera(tmp_path):
    stubs = tl._stubs(tmp_path)
    actions, cfg = tl.run_setup("bringup.launch.py", stubs)
    n = tl.nodes(actions, stubs["__classes__"])
    assert cfg["demo"] == "false" and cfg["hud"] == "true" and cfg["follow_camera"] == ""
    assert "demo_hud" in n and "follow_camera" not in n
    assert tl.param_dicts(n["demo_hud"])[-1]["use_sim_time"] is True
    assert tl.param_dicts(n["mission_manager"])[-1]["autostart"] is False


def test_bringup_demo_mode(tmp_path):
    stubs = tl._stubs(tmp_path)
    actions, cfg = tl.run_setup("bringup.launch.py", stubs, {"demo": "true"})
    n = tl.nodes(actions, stubs["__classes__"])
    assert {"demo_hud", "follow_camera", "mission_manager", "cmd_vel_mux"} <= set(n)
    mm = tl.param_dicts(n["mission_manager"])[-1]
    assert mm["autostart"] is True and mm["autostart_delay"] == pytest.approx(25.0)
    args = n["follow_camera"].arguments
    lay = Layout.load()
    assert args[args.index("--target") + 1] == lay["robot_name"]
    assert args[args.index("--world") + 1] == lay["world_name"]
    assert "--offset=-2.5,0.0,1.6" in args
    logs = [a.kwargs.get("msg", "") for a in actions if isinstance(a, stubs["__classes__"]["LogInfo"])]
    assert any("DEMO" in m for m in logs)
    # demo without a GUI: nothing to follow with
    actions, _ = tl.run_setup("bringup.launch.py", stubs, {"demo": "true", "gui": "false"})
    n = tl.nodes(actions, stubs["__classes__"])
    assert "follow_camera" not in n and "demo_hud" in n


def test_bringup_follow_and_hud_switches(tmp_path):
    stubs = tl._stubs(tmp_path)
    actions, _ = tl.run_setup("bringup.launch.py", stubs, {"follow_camera": "true", "follow_offset": "-3 0.5 2",
                                                           "hud": "false"})
    n = tl.nodes(actions, stubs["__classes__"])
    assert "demo_hud" not in n and "--offset=-3,0.5,2" in n["follow_camera"].arguments
    assert tl.param_dicts(n["mission_manager"])[-1]["autostart"] is False       # follow alone is not a demo
    actions, _ = tl.run_setup("bringup.launch.py", stubs, {"demo": "true", "follow_camera": "false"})
    assert "follow_camera" not in tl.nodes(actions, stubs["__classes__"])


def test_bringup_layout_names_fallback(tmp_path):
    m = tl.load_launch_module("bringup.launch.py", tl._stubs(tmp_path))
    assert m._layout_names(str(tmp_path / "missing.yaml")) == ("plant", "spot")
    p = tmp_path / "lay.yaml"
    p.write_text("world_name: w2\nrobot_name: r2\n", encoding="utf-8")
    assert m._layout_names(str(p)) == ("w2", "r2")


# ============================================================================
# gui.config / bridge / script / particle colour
# ============================================================================
def _gui_plugins():
    s = open(os.path.join(SIM, "config", "gui.config"), encoding="utf-8").read()
    assert "\r" not in s
    root = ET.fromstring("<root>" + re.sub(r"<\?xml[^>]*\?>", "", s) + "</root>")
    return root.findall("plugin")


def test_gui_config_follow_target_and_hud_panel():
    plugins = _gui_plugins()
    ct = [p for p in plugins if p.get("filename") == "CameraTracking"]
    assert len(ct) == 1
    lay = Layout.load()
    assert ct[0].findtext("follow_target") == lay["robot_name"]
    off = tuple(float(v) for v in ct[0].findtext("follow_offset").split())
    assert off == pytest.approx(fc.FOLLOW_DEFAULTS["offset"])
    assert float(ct[0].findtext("follow_pgain")) == pytest.approx(fc.FOLLOW_DEFAULTS["pgain"])
    # bringup's follow_offset default is the same offset
    m = tl.load_launch_module("bringup.launch.py", tl._stubs(None))
    decl = {a.name: a.default for a in m.generate_launch_description().entities
            if isinstance(a, tl.DeclareLaunchArgument)}
    assert fc.parse_offset(decl["follow_offset"]) == pytest.approx(off)
    hud = [p for p in plugins if p.get("filename") == "ImageDisplay" and p.findtext("topic") == "/demo/hud"]
    assert len(hud) == 1 and hud[0].findtext("topic_picker") == "false"
    props = {e.get("key"): e.text for e in hud[0].find("gz-gui").iter("property")}
    assert props["state"] == "floating"
    assert hud[0].find("gz-gui/anchors").get("target") == "3D View"
    # the raw camera panels are still there
    topics = {p.findtext("topic") for p in plugins if p.get("filename") == "ImageDisplay"}
    assert {"/inspection_camera/image", "/thermal_camera/image", "/demo/hud"} == topics


def test_bridge_forwards_the_hud_to_gazebo():
    with open(os.path.join(SIM, "config", "ros_gz_bridge.yaml"), encoding="utf-8") as f:
        entries = yaml.safe_load(f)
    hud = [e for e in entries if e["ros_topic_name"] == dh.HUD_DEFAULTS["gz_hud_topic"]]
    assert len(hud) == 1
    e = hud[0]
    assert (e["gz_topic_name"], e["ros_type_name"], e["gz_type_name"], e["direction"]) == \
        ("/demo/hud", "sensor_msgs/msg/Image", "gz.msgs.Image", "ROS_TO_GZ")
    # the bgr8 ROS topic itself is not bridged (gz-gui ImageDisplay cannot show BGR)
    assert not [x for x in entries if x["ros_topic_name"] == dh.HUD_DEFAULTS["hud_topic"]]


def test_run_demo_script():
    path = os.path.join(REPO, "scripts", "run_demo.sh")
    raw = open(path, "rb").read()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert '. "$(dirname "${BASH_SOURCE[0]}")/common.sh"' in text and "source_ws" in text
    # every argument goes through parse_launch_flags (--low-memory -> low_memory:=true)
    assert 'parse_launch_flags "$@"' in text
    assert 'exec ros2 launch plant_simulation bringup.launch.py demo:=true "${LAUNCH_ARGS[@]}"' in text


def test_gas_plume_is_green():
    root = ET.parse(os.path.join(SIM, "models", "gas_leak_valve", "model.sdf")).getroot()
    em = root.find("model/link/particle_emitter")
    r, g, b, a = (float(v) for v in em.findtext("material/diffuse").split())
    assert g > 0.9 and g > r + 0.3 and g > b + 0.3            # ogre2 colours particles by the material
    r, g, b, a = (float(v) for v in em.findtext("color_start").split())
    assert g > r + 0.3 and g > b + 0.3 and 0.0 < a < 1.0
    assert float(em.findtext("color_end").split()[3]) == 0.0
    src = open(os.path.join(REPO, "tools", "generate_world.py"), encoding="utf-8").read()
    assert em.findtext("color_start") in src and em.findtext("material/diffuse") in src   # regenerated, not hand-edited
