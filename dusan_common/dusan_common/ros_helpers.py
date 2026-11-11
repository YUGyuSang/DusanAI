"""
rclpy helpers shared by all nodes (import only inside ROS nodes).
"""
import json
import math


def latched_qos(depth=1):
    """Transient-local reliable QoS (like ROS1 latched topics) for maps/plans."""
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
    return QoSProfile(depth=depth, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)


def sensor_qos(depth=5):
    """Best-effort QoS for high-rate sensor streams: rclpy's
    qos_profile_sensor_data (keep-last, best-effort, volatile) with the given
    depth.  A new profile object is returned on every call, so a caller that
    modifies it does not change the profile of other subscriptions."""
    try:
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
    except ImportError:  # minimal rclpy stand-ins of the offline unit tests
        from rclpy.qos import qos_profile_sensor_data
        return qos_profile_sensor_data
    return QoSProfile(depth=int(depth), history=HistoryPolicy.KEEP_LAST,
                      reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if not math.isfinite(v) else v
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def to_json(obj):
    """JSON encode; numpy types supported, NaN/inf become null."""
    return json.dumps(_sanitize(obj), default=_json_default)


def _sanitize(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return _sanitize(obj.tolist())
        if isinstance(obj, np.floating):
            return _sanitize(float(obj))
    except Exception:
        pass
    return obj


def json_msg(obj):
    """Build a std_msgs/String carrying JSON (NaN/inf -> null)."""
    from std_msgs.msg import String
    m = String()
    m.data = to_json(obj)
    return m


def parse_json_msg(msg, default=None):
    try:
        return json.loads(msg.data)
    except Exception:
        return default


def declare_params(node, params):
    """Declare a dict of {name: default} parameters and return {name: value}."""
    out = {}
    for name, default in params.items():
        node.declare_parameter(name, default)
        out[name] = node.get_parameter(name).value
    return out


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def make_marker(ns, mid, mtype, frame_id="map", stamp=None, pose=(0.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1),
                color=(1.0, 1.0, 1.0, 1.0), text="", lifetime=0.0):
    """Small visualization_msgs/Marker factory."""
    from visualization_msgs.msg import Marker
    m = Marker()
    m.header.frame_id = frame_id
    if stamp is not None:
        m.header.stamp = stamp
    m.ns = ns
    m.id = int(mid)
    m.type = mtype
    m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pose[:3])
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
    m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in color)
    m.text = text
    m.lifetime.sec = int(lifetime)
    m.lifetime.nanosec = int((lifetime - int(lifetime)) * 1e9)
    return m
