"""
Small rclpy glue shared by the plant_simulation nodes.

rclpy / message packages are imported lazily inside the functions so that
this module can be imported (and py_compiled) on a machine without ROS.
"""
import math
import traceback
import zlib


def run_node(node_factory, args=None):
    """Standard main(): init, spin, clean shutdown.

    Handles Ctrl-C both as ``KeyboardInterrupt`` (Humble style) and as
    ``rclpy.executors.ExternalShutdownException`` (Jazzy and later, where the
    SIGINT handler shuts the context down while ``spin`` is running).  If the
    node defines ``on_shutdown()`` it is called before the node is destroyed
    (used by the metrics logger to write its final summary).
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
                except Exception as exc:  # pragma: no cover - best effort on exit
                    print(f"[{node.get_name()}] on_shutdown failed: {exc}")
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def guarded(node, name, fn, every=50):
    """Wrap a ROS callback so that an exception is logged instead of leaving
    ``rclpy.spin`` - run_node re-raises, so one bad message or logic error
    would otherwise kill the node for the rest of the run (field report: the
    mission manager died, /mission/state went stale on the dashboard and 't'
    did nothing).  The traceback is logged the first time and then every
    ``every`` failures.  The wrapper returns None after a failure."""
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
    return wrapper


def now_sec(node):
    """Current node time (sim time when use_sim_time is true) in seconds."""
    return node.get_clock().now().nanoseconds * 1e-9


def reliable_qos(depth=10):
    """Reliable / volatile QoS (ROS default) - commands and JSON status topics."""
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
    return QoSProfile(depth=depth, durability=DurabilityPolicy.VOLATILE,
                      reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)


def best_effort_qos(depth=5):
    """Best-effort / volatile QoS.  A best-effort subscription matches both
    reliable and best-effort publishers, so it is the safe choice for streams
    whose publisher QoS is owned by somebody else (images, grids, sensors)."""
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
    return QoSProfile(depth=depth, durability=DurabilityPolicy.VOLATILE,
                      reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)


def subscribe_latched(node, msg_type, topic, callback):
    """Subscribe to a topic that is *supposed* to be latched (transient local).

    Two subscriptions are created: a transient-local one (receives the stored
    sample of a latched publisher when we join late) and a volatile
    best-effort one (still works if the publisher turned out to be volatile,
    where a transient-local subscription would be QoS-incompatible and never
    receive anything).  Callbacks must therefore be idempotent.
    """
    from dusan_common.ros_helpers import latched_qos
    return [node.create_subscription(msg_type, topic, callback, latched_qos()),
            node.create_subscription(msg_type, topic, callback, best_effort_qos(depth=1))]


def subscribe_any(node, msg_type, topic, callback, depth=5):
    """Subscribe so that publishers with ANY reliability/durability match.

    A reliable subscription does not match a best-effort publisher; a
    best-effort subscription matches everything but may drop samples of a
    reliable publisher under load.  Using both gives lossless delivery from
    reliable publishers and still works with best-effort ones.  The callback
    may see the same sample twice, so it must be idempotent (state flags,
    JSON status snapshots).
    """
    return [node.create_subscription(msg_type, topic, callback, reliable_qos(depth)),
            node.create_subscription(msg_type, topic, callback, best_effort_qos(depth))]


def finite(v, default=None):
    """float(v) if it is a finite number, else ``default``."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def map_fingerprint(msg):
    """Cheap identity of an OccupancyGrid-like message (geometry + CRC32 of
    the cells), used to ignore re-publications of an unchanged map.  Works
    with rclpy's ``array.array`` cell data as well as plain lists."""
    info = msg.info
    data = msg.data
    try:
        raw = memoryview(data).tobytes()
    except TypeError:                      # list / tuple of ints
        import numpy as np
        raw = np.asarray(data, dtype=np.int8).tobytes()
    return (int(info.width), int(info.height), round(float(info.resolution), 6),
            round(float(info.origin.position.x), 4), round(float(info.origin.position.y), 4),
            str(getattr(msg.header, "frame_id", "")), zlib.crc32(raw))
