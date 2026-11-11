"""
ROS 2 node ``keyboard_teleop`` - terminal raw-mode teleop (+ Gazebo keys).

Run it in a real terminal::

    ros2 run module1_locomotion keyboard_teleop
    ros2 run module1_locomotion keyboard_teleop --ros-args -p mode:=latch

Publishes
  /cmd_vel/teleop      geometry_msgs/Twist   20 Hz while driving / manual override /
                                             E-STOP, then ONE zero twist when it stops
  /teleop/active       std_msgs/Bool         manual override flag (on change + 2 Hz,
                                             transient-local)
  /locomotion/command  std_msgs/String       stand, sit, gait:*, mode:toggle, recover, push, reset_pose,
                                             estop, release
  /mission/command     std_msgs/String       patrol, seek, rth, start, pause, resume
  /battery/set_level   std_msgs/Float32      percent (19 = RTH test, 100 = full)
Subscribes
  /keyboard/keypress   std_msgs/Int32        Qt key codes (Gazebo KeyPublisher, optional)
  /locomotion/state    std_msgs/String JSON  mode / gait for the status line
  /battery/state       sensor_msgs/BatteryState
  /mission/state       std_msgs/String JSON

Terminal handling: termios non-canonical, no echo, no signal keys (Ctrl-C
arrives as a byte and quits cleanly after sending a zero twist).  The
previous terminal mode is restored on every exit path (normal quit,
exception, SIGINT/SIGTERM/SIGHUP, atexit).  When stdin is not a TTY (e.g.
started by ros2 launch without a terminal prefix) the node runs in
"Gazebo keys only" mode instead of failing.

The key map and the velocity logic live in teleop_keys.py (ROS free, tested).
"""
import atexit
import json
import math
import os
import select
import shutil
import signal
import sys
import threading
import time

try:  # POSIX only (the robot PC); Windows has no termios
    import termios
except ImportError:  # pragma: no cover
    termios = None

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32, Int32, String

from dusan_common.ros_helpers import latched_qos, sensor_qos

from .teleop_keys import (HELP_TEXT, CommandStreamer, KeyDecoder, TeleopConfig, TeleopState,
                          format_status, mission_label, mission_stale, qt_key_to_token)


def _reliable_qos(depth=10):
    return QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)


class RawTerminal:
    """Context for a terminal in non-canonical, no-echo, no-signal mode."""

    def __init__(self, fd: int):
        self.fd = fd
        self._old = None

    def open(self) -> None:
        self._old = termios.tcgetattr(self.fd)
        new = termios.tcgetattr(self.fd)
        new[0] &= ~(termios.IXON | termios.ICRNL)                   # iflag: no XON/XOFF, keep CR
        new[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG | termios.IEXTEN)   # lflag
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSADRAIN, new)
        atexit.register(self.restore)

    def restore(self) -> None:
        if self._old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None

    def read(self, timeout: float) -> bytes:
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return b""
        data = os.read(self.fd, 64)
        if not data:          # EOF: terminal closed
            raise EOFError("terminal closed")
        return data


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("keyboard_teleop")
        P = self._param
        dflt = TeleopConfig()
        cfg = TeleopConfig.from_dict({
            "mode": P("mode", dflt.mode),
            "max_vx": P("max_vx", dflt.max_vx),
            "max_vy": P("max_vy", dflt.max_vy),
            "max_wz": P("max_wz", dflt.max_wz),
            "speed_scale": P("speed_scale", dflt.speed_scale),
            "scale_step": P("scale_step", dflt.scale_step),
            "min_scale": P("min_scale", dflt.min_scale),
            "max_scale": P("max_scale", dflt.max_scale),
            "hold_timeout": P("hold_timeout", dflt.hold_timeout),
            "latch_steps": P("latch_steps", dflt.latch_steps),
            "battery_test_level": P("battery_test_level", dflt.battery_test_level),
            "battery_full_level": P("battery_full_level", dflt.battery_full_level),
        })
        self.publish_rate = max(1.0, float(P("publish_rate", 20.0)))
        self.status_rate = max(0.5, float(P("status_rate", 5.0)))
        self.active_period = max(0.1, float(P("active_republish_period", 0.5)))
        use_terminal = bool(P("use_terminal", True))
        gz_keys = bool(P("gz_keys", True))
        t_cmd = P("cmd_vel_topic", "/cmd_vel/teleop")
        t_active = P("active_topic", "/teleop/active")
        t_loco = P("locomotion_command_topic", "/locomotion/command")
        t_mission = P("mission_command_topic", "/mission/command")
        t_batt_set = P("battery_set_topic", "/battery/set_level")
        t_loco_state = P("locomotion_state_topic", "/locomotion/state")
        t_batt_state = P("battery_state_topic", "/battery/state")
        t_mission_state = P("mission_state_topic", "/mission/state")
        t_keys = P("keypress_topic", "/keyboard/keypress")

        self.state = TeleopState(cfg)
        self.lock = threading.RLock()
        self.quit = False
        self.use_tty = bool(use_terminal and termios is not None and sys.stdin is not None
                            and sys.stdin.isatty())
        self.loco_mode = None
        self.loco_gait = None
        self.loco_algo = None          # locomotion_mode of the controller (stable | mpc)
        self.battery_pct = None
        self.mission = None
        self._t_mission = None         # wall time of the last /mission/state (stale -> mission_manager down?)
        self.gz_key_count = 0
        self.streamer = CommandStreamer(self.publish_rate)
        self._last_active = None
        self._next_active = 0.0
        self._next_status = 0.0
        self._next_log = time.monotonic() + 10.0

        self.pub_cmd = self.create_publisher(Twist, t_cmd, _reliable_qos(10))
        self.pub_active = self.create_publisher(Bool, t_active, latched_qos(1))
        self.pub_loco = self.create_publisher(String, t_loco, _reliable_qos(10))
        self.pub_mission = self.create_publisher(String, t_mission, _reliable_qos(10))
        self.pub_batt = self.create_publisher(Float32, t_batt_set, _reliable_qos(10))
        qos_in = sensor_qos()
        self.create_subscription(String, t_loco_state, self._on_loco_state, qos_in)
        self.create_subscription(BatteryState, t_batt_state, self._on_battery, qos_in)
        self.create_subscription(String, t_mission_state, self._on_mission_state, qos_in)
        if gz_keys:
            self.create_subscription(Int32, t_keys, self._on_gz_key, _reliable_qos(20))
        mode = "terminal + Gazebo keys" if self.use_tty else "Gazebo keys only (stdin is not a TTY)"
        self.get_logger().info(f"keyboard_teleop: {mode}, velocity mode '{cfg.mode}', "
                               f"publishing {t_cmd} at {self.publish_rate:.0f} Hz")

    # ----------------------------------------------------------------- utils
    def _param(self, name, default):
        """Declare a parameter; a wrong type in the YAML (e.g. ``1`` for a
        float) is retried with dynamic typing and coerced to the default's
        type (TeleopConfig.from_dict coerces numbers), else the default is
        kept - the teleop never dies on a config typo."""
        try:
            self.declare_parameter(name, default)
            v = self.get_parameter(name).value
            return default if v is None else v
        except Exception as e:
            first_error = e
        try:
            from rcl_interfaces.msg import ParameterDescriptor
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
            v = self.get_parameter(name).value
            if v is None:
                return default
            if isinstance(default, bool) and not isinstance(v, bool):
                v = str(v).strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, float):
                v = float(v)
            elif isinstance(default, str):
                v = str(v)
            self.get_logger().warning(f"parameter '{name}': wrong type in the parameter file, using {v!r}")
            return v
        except Exception:
            self.get_logger().error(f"parameter '{name}': {first_error} -> using default {default!r}")
            return default

    def say(self, text: str) -> None:
        """Log line above the live status line (terminal) or to the ROS log."""
        if self.use_tty:
            with self.lock:
                sys.stdout.write("\r\x1b[2K" + text + "\r\n")
                sys.stdout.flush()
                self._next_status = 0.0          # redraw the status line immediately
        else:
            self.get_logger().info(text)

    def print_help(self) -> None:
        if self.use_tty:
            with self.lock:
                sys.stdout.write("\r\x1b[2K" + HELP_TEXT.replace("\n", "\r\n") + "\r\n")
                sys.stdout.flush()
        else:
            self.get_logger().info("Gazebo key mode - key map:\n" + HELP_TEXT)

    # ------------------------------------------------------------- callbacks
    def _on_loco_state(self, msg: String):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        if isinstance(d, dict):
            self.loco_mode = d.get("mode", self.loco_mode)
            self.loco_algo = d.get("locomotion_mode", self.loco_algo)
            gait = d.get("gait", self.loco_gait)
            if d.get("gait_auto"):
                gait = f"{gait}(auto)"
            self.loco_gait = gait

    def _on_battery(self, msg: BatteryState):
        p = float(msg.percentage)
        if math.isfinite(p):
            self.battery_pct = p * 100.0 if p <= 1.0 else p

    def _on_mission_state(self, msg: String):
        try:
            d = json.loads(msg.data)
            st = d.get("state") if isinstance(d, dict) else str(d)
        except Exception:
            st = msg.data.strip() or None
        self.mission = st
        self._t_mission = time.monotonic()
        with self.lock:
            self.state.mission_state = st

    def _on_gz_key(self, msg: Int32):
        tok = qt_key_to_token(msg.data)
        if tok is None:
            return
        self.gz_key_count += 1
        self.handle_token(tok)

    # ------------------------------------------------------------------ logic
    def handle_token(self, tok: str) -> None:
        with self.lock:
            events = self.state.handle_key(tok, time.monotonic())
        for ev in events:
            self._apply(ev)

    def _apply(self, ev) -> None:
        if ev.kind == "locomotion":
            self.pub_loco.publish(String(data=str(ev.value)))
            self.say(f"-> /locomotion/command  {ev.value}")
        elif ev.kind == "mission":
            self.pub_mission.publish(String(data=str(ev.value)))
            self.say(f"-> /mission/command  {ev.value}")
            if mission_stale(self._t_mission, time.monotonic()):
                # nobody answers: the command goes nowhere - say so instead of silence
                self.say("   ! no /mission/state from mission_manager - it is not running (or crashed): "
                         "look for a Traceback / 'process has died' in the bringup terminal, restart bringup")
        elif ev.kind == "battery":
            self.pub_batt.publish(Float32(data=float(ev.value)))
            self.say(f"-> /battery/set_level  {ev.value:.0f} %")
        elif ev.kind == "override":
            self._publish_active(bool(ev.value))
            self.say("manual override " + ("ON  (teleop keeps priority)" if ev.value else "OFF"))
        elif ev.kind == "estop":
            self.say("E-STOP ENGAGED" if ev.value else "E-STOP released")
        elif ev.kind == "help":
            self.print_help()
        elif ev.kind == "quit":
            self.quit = True
        elif ev.kind == "info":
            self.say(str(ev.value))

    def _publish_twist(self, vx: float, vy: float, wz: float) -> None:
        t = Twist()
        t.linear.x = float(vx)
        t.linear.y = float(vy)
        t.angular.z = float(wz)
        self.pub_cmd.publish(t)

    def _publish_active(self, value: bool, now: float = None) -> None:
        self.pub_active.publish(Bool(data=bool(value)))
        self._last_active = bool(value)
        self._next_active = (time.monotonic() if now is None else now) + self.active_period

    def tick(self, now: float) -> None:
        """Periodic work (called ~50 Hz from the main loop, wall clock)."""
        with self.lock:
            vel = self.state.velocity(now)
            streaming = self.state.streaming(now)
            override = self.state.manual_override
        cmd = self.streamer.step(now, streaming, vel)   # 20 Hz while active, one zero on stop
        if cmd is not None:
            self._publish_twist(*cmd)
        if override != self._last_active or now >= self._next_active:
            self._publish_active(override, now)
        if self.use_tty and now >= self._next_status:
            self._next_status = now + 1.0 / self.status_rate
            cols = shutil.get_terminal_size((100, 24)).columns      # never wrap the status line
            with self.lock:
                line = format_status(self.state, now, self.loco_mode, self.loco_gait, self.battery_pct,
                                     mission_label(self.mission, self._t_mission, now), self.gz_key_count,
                                     width=cols - 1, loco_algo=self.loco_algo)
                sys.stdout.write("\r\x1b[2K" + line)
                sys.stdout.flush()
        elif not self.use_tty and now >= self._next_log:
            self._next_log = now + 10.0
            with self.lock:
                line = format_status(self.state, now, self.loco_mode, self.loco_gait, self.battery_pct,
                                     mission_label(self.mission, self._t_mission, now), self.gz_key_count,
                                     width=200, loco_algo=self.loco_algo)
            self.get_logger().info(line)

    def stop_robot(self) -> None:
        """Final zero twist + manual override released."""
        with self.lock:
            self.state.zero()
            self.state.manual_override = False
        self._publish_twist(0.0, 0.0, 0.0)
        self._publish_active(False)

    def run(self, term) -> None:
        self.print_help()
        decoder = KeyDecoder()
        esc_since = None
        while rclpy.ok() and not self.quit:
            if term is not None:
                try:
                    data = term.read(0.02)
                except (OSError, EOFError):
                    self.quit = True
                    break
                toks = decoder.feed(data) if data else []
                now = time.monotonic()
                if decoder.pending:
                    if esc_since is None:
                        esc_since = now
                    elif now - esc_since > 0.05:        # lone ESC (no sequence followed)
                        toks += decoder.flush()
                        esc_since = None
                else:
                    esc_since = None
                for tok in toks:
                    self.handle_token(tok)
            else:
                time.sleep(0.02)
            self.tick(time.monotonic())


def _spin(executor) -> None:
    try:
        executor.spin()
    except Exception:  # ExternalShutdownException / context shut down
        pass


def _raise_exit(signum, _frame):
    raise SystemExit(128 + signum)


def _init_rclpy(args) -> bool:
    """``rclpy.init`` WITHOUT rclpy's own SIGINT/SIGTERM handlers when the
    running rclpy supports it (``SignalHandlerOptions``, Humble and later).

    rclpy's handler shuts the context down on Ctrl-C before ``main`` can
    react, so the final zero twist and ``/teleop/active = false`` could not
    be published any more - and a latched manual-override flag would keep the
    cmd_vel mux on teleop.  Without it, SIGINT raises KeyboardInterrupt in
    the main thread and the normal shutdown path runs.  Falls back to the
    plain call on any older / different rclpy.  Returns True when the
    handlers were disabled."""
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        return True
    except (ImportError, AttributeError, TypeError):
        rclpy.init(args=args)
        return False


def main(args=None):
    _init_rclpy(args)
    node = KeyboardTeleop()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=_spin, args=(executor,), daemon=True)
    term = None
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
        if sig is not None:
            try:
                signal.signal(sig, _raise_exit)
            except (ValueError, OSError):
                pass
    try:
        if node.use_tty:
            try:
                term = RawTerminal(sys.stdin.fileno())
                term.open()
            except (termios.error, OSError, ValueError) as e:   # e.g. a TTY we may not configure
                term = None
                node.use_tty = False
                node.get_logger().warning(f"cannot use the terminal ({e}); Gazebo keys only")
        spinner.start()
        node.run(term)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if term is not None:
            term.restore()
            sys.stdout.write("\r\n")
            sys.stdout.flush()
        try:
            if rclpy.ok():
                node.stop_robot()
                time.sleep(0.15)          # let the last messages leave
        except Exception:
            pass
        try:
            executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        if spinner.is_alive():
            spinner.join(timeout=1.0)
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
