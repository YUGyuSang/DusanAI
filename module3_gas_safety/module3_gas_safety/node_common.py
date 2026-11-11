"""
Small rclpy glue shared by the Module-3 nodes.

rclpy and message packages are imported lazily inside the functions, so this
module (and everything importing it at module level) can be imported and
py_compiled on a machine without ROS.
"""
from __future__ import annotations

import math


def run_node(node_factory, args=None) -> None:
    """Standard main(): init, spin, clean shutdown (Humble .. Lyrical).

    Ctrl-C arrives either as ``KeyboardInterrupt`` or, on Jazzy and later, as
    ``rclpy.executors.ExternalShutdownException`` raised from ``spin``.
    """
    import rclpy

    rclpy.init(args=args)
    node = None
    try:
        node = node_factory()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except BaseException as exc:  # noqa: B902 - re-raised unless it is a shutdown
        if type(exc).__name__ != "ExternalShutdownException":
            raise
    finally:
        if node is not None:
            hook = getattr(node, "on_shutdown", None)
            if callable(hook):
                try:
                    hook()
                except Exception:
                    pass
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def guarded(node, name: str, fn, every: int = 50):
    """Wrap a subscription / timer callback so that an exception is logged
    instead of leaving ``rclpy.spin``: ``run_node`` re-raises, so one malformed
    message (a JSON list instead of an object, a grid whose data does not fit
    its size, a NaN) or a logic error would otherwise kill the node for the rest
    of the run - a dead return_to_home / battery_simulator / source_seeker stops
    the safety supervision or the gas search without anything visible on the
    dashboard.  The traceback is logged on the first failure and then every
    ``every`` failures; the wrapper returns None after a failure."""
    import traceback

    count = [0]

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            count[0] += 1
            if count[0] == 1 or count[0] % max(1, int(every)) == 0:
                node.get_logger().error(f"{name} failed ({count[0]}x): {type(exc).__name__}: {exc}\n"
                                        f"{traceback.format_exc()}")
            return None

    wrapper.failures = lambda: count[0]
    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


def finite_or_none(v):
    """float(v) when it is a finite number, else None (NaN / inf / garbage)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def json_dict(msg) -> dict:
    """std_msgs/String JSON -> dict ({} for malformed JSON, null, lists, strings ...)."""
    from dusan_common.ros_helpers import parse_json_msg
    d = parse_json_msg(msg, None)
    return d if isinstance(d, dict) else {}


def declare(node, defaults: dict) -> dict:
    """Declare every parameter of ``defaults`` (the default fixes the type)
    and return the current values, with ints promoted where a float is
    expected (YAML ``1`` for a double parameter would otherwise be rejected
    by rclpy before we ever see it, so the YAML files use explicit floats;
    this is just a second line of defence for values set at runtime)."""
    out = {}
    for name, default in defaults.items():
        node.declare_parameter(name, default)
        v = node.get_parameter(name).value
        if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        if isinstance(default, list) and v is not None:
            v = list(v)
        out[name] = v
    return out


def now_sec(node) -> float:
    """Node time in seconds (simulation time when use_sim_time is true)."""
    return node.get_clock().now().nanoseconds * 1e-9


def stamp(node):
    return node.get_clock().now().to_msg()


def reliable_qos(depth: int = 10):
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
    return QoSProfile(depth=depth, durability=DurabilityPolicy.VOLATILE,
                      reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)


def best_effort_qos(depth: int = 5):
    """Best-effort subscription: matches reliable *and* best-effort publishers."""
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
    return QoSProfile(depth=depth, durability=DurabilityPolicy.VOLATILE,
                      reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)


def latched_qos(depth: int = 1):
    from dusan_common.ros_helpers import latched_qos as _lq
    return _lq(depth)


def subscribe_latched(node, msg_type, topic: str, callback):
    """Subscribe to a topic that should be latched (transient local).

    A transient-local subscription receives the stored sample of a latched
    publisher when joining late, but is incompatible with a volatile
    publisher; a second volatile best-effort subscription covers that case.
    Callbacks must therefore be idempotent.
    """
    subs = [node.create_subscription(msg_type, topic, callback, latched_qos(1)),
            node.create_subscription(msg_type, topic, callback, best_effort_qos(1))]
    return subs


def odom_pose_twist(msg):
    """nav_msgs/Odometry -> (x, y, yaw, vx_world, vy_world, wz, speed).

    The project contract (DESIGN_SPEC section 4) has the twist in the base frame.
    """
    from dusan_common.geometry import yaw_from_quat
    p = msg.pose.pose.position
    yaw = yaw_from_quat(msg.pose.pose.orientation)
    vxb = msg.twist.twist.linear.x
    vyb = msg.twist.twist.linear.y
    c, s = math.cos(yaw), math.sin(yaw)
    return (p.x, p.y, yaw, c * vxb - s * vyb, s * vxb + c * vyb, msg.twist.twist.angular.z,
            math.hypot(vxb, vyb))


def odom_pose_twist_finite(msg):
    """:func:`odom_pose_twist`, or None when any value is NaN / inf.

    A diverged physics step (or a bad bridge sample) can put non-finite
    numbers into /odom; the grid lookups of the models (int(floor(nan)))
    would raise inside a timer and kill the node, so such samples are
    dropped and the last good pose is kept."""
    v = odom_pose_twist(msg)
    return v if all(math.isfinite(float(x)) for x in v) else None


def warn_throttled(node, key: str, text: str, period_s: float = 5.0) -> None:
    """``get_logger().warning`` at most once per ``period_s`` (wall clock) per key."""
    import time
    store = getattr(node, "_m3_warn_times", None)
    if store is None:
        store = {}
        setattr(node, "_m3_warn_times", store)
    now = time.monotonic()
    last = store.get(key)
    if last is None or now - last >= period_s:
        store[key] = now
        node.get_logger().warning(text)


def twist_msg(vx: float, vy: float, wz: float):
    from geometry_msgs.msg import Twist
    m = Twist()
    m.linear.x = float(vx)
    m.linear.y = float(vy)
    m.angular.z = float(wz)
    return m


def load_layout_and_map(node, layout_file: str, map_yaml: str):
    """Layout + nav map (map file, else rasterised layout) with logging."""
    from .layout_map import load_nav_map, resolve_layout
    layout = resolve_layout(layout_file)
    grid, desc = load_nav_map(map_yaml, layout)
    node.get_logger().info(f"layout: {layout.path} | nav map: {desc} ({grid.width}x{grid.height} @ {grid.resolution} m)")
    if desc == "layout-rasterised":
        node.get_logger().warning("plant_map.yaml not found - using a map rasterised from world_layout.yaml")
    return layout, grid
