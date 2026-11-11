"""
ROS 2 node ``synthetic_thermal_camera`` - publishes a synthetic radiometric
thermal image rendered from the robot pose (``synthetic_thermal.py``).

Used when the Gazebo thermal sensor is unavailable (launch argument
``thermal_source:=synthetic``).  Output matches the Gazebo sensor contract:

    /thermal_camera/image   sensor_msgs/Image  mono16, Kelvin = value x 0.01, 320 x 240, 5 Hz

Subscribes /odom (nav_msgs/Odometry, world frame).  Until the first odometry
message arrives nothing is published (or, with ``publish_without_odom``,
the layout spawn pose is used).

The node replaces the Gazebo thermal camera only when the simulation runs without it
(``plant_simulation bringup.launch.py thermal_source:=synthetic`` does that; standalone:
``sim.launch.py enable_thermal:=false``).  It also listens on its own output topic and warns
when frames arrive that it did not publish (``OwnFrameFilter``: a second thermal source).
"""
import array

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from dusan_common.geometry import odom_to_pose
from dusan_common.layout import Layout
from dusan_common.robot_model import NOMINAL_BODY_HEIGHT
from dusan_common.ros_helpers import sensor_qos
from module2_inspection.synthetic_thermal import OwnFrameFilter, SyntheticThermalCamera, full_pose

try:  # Jazzy and later: spin() raises this when the context is shut down (e.g. SIGINT from launch)
    from rclpy.executors import ExternalShutdownException
except ImportError:  # pragma: no cover - older rclpy
    class ExternalShutdownException(Exception):
        """Placeholder for rclpy versions without ExternalShutdownException."""

PARAMS = {
    "layout_file": "",
    "rate": 5.0,
    "topic": "/thermal_camera/image",
    "odom_topic": "/odom",
    "frame_id": "thermal_camera_link",
    "encoding": "mono16",          # mono16 | 16UC1 | L16  (all uint16, Kelvin x 100)
    "resolution": 0.01,
    "width": 320,
    "height": 240,
    "hfov": 0.80,
    "netd_k": 0.05,
    "column_fpn_k": 0.03,
    "row_noise_k": 0.02,
    "blur_sigma_px": 0.6,
    "machine_k": 303.15,
    "seed": 0,
    "use_odom_z": True,            # False: base height = floor + body_height
    "body_height": NOMINAL_BODY_HEIGHT,
    "publish_without_odom": False,
}


def mono16_image_msg(counts: np.ndarray, encoding: str, frame_id: str, stamp) -> Image:
    """uint16 counts -> sensor_msgs/Image (little endian, ``step = 2 * width``).

    ``data`` is assigned as ``array.array('B')``, which the generated setter of
    a ``uint8[]`` field accepts directly (assigning ``bytes`` would validate
    every byte in Python)."""
    img = np.ascontiguousarray(counts, dtype="<u2")
    msg = Image()
    msg.header.frame_id = str(frame_id)
    msg.header.stamp = stamp
    msg.height = int(img.shape[0])
    msg.width = int(img.shape[1])
    msg.encoding = str(encoding)
    msg.is_bigendian = 0
    msg.step = int(img.shape[1] * 2)
    msg.data = array.array("B", img.tobytes())
    return msg


class SyntheticThermalNode(Node):
    def __init__(self):
        super().__init__("synthetic_thermal_camera")
        self.cfg = {}
        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
            self.cfg[name] = self.get_parameter(name).value
        c = self.cfg
        self.layout = Layout.load(c["layout_file"] or None)
        self.camera = SyntheticThermalCamera(
            self.layout, width=int(c["width"]), height=int(c["height"]), hfov=float(c["hfov"]),
            netd_k=float(c["netd_k"]), column_fpn_k=float(c["column_fpn_k"]), row_noise_k=float(c["row_noise_k"]),
            blur_sigma_px=float(c["blur_sigma_px"]), resolution=float(c["resolution"]), seed=int(c["seed"]),
            machine_k=float(c["machine_k"]))
        self.pose = None
        self.create_subscription(Odometry, c["odom_topic"], self.on_odom, sensor_qos())
        self.pub = self.create_publisher(Image, c["topic"], 5)
        # frames on the output topic that this node did not publish come from a second source
        # (normally the Gazebo thermal camera of a simulation started without
        # enable_thermal:=false) and would reach thermal_fusion interleaved with ours
        self.own_frames = OwnFrameFilter(history=50)
        self.n_foreign = 0
        self.create_subscription(Image, c["topic"], self.on_output_topic, sensor_qos())
        self.create_timer(1.0 / max(0.1, float(c["rate"])), self.on_timer)
        self.n = 0
        self.get_logger().info(
            f"synthetic thermal camera {c['width']}x{c['height']} hfov {c['hfov']} -> {c['topic']} "
            f"({c['encoding']}, K = value x {c['resolution']}) at {c['rate']} Hz")

    def on_odom(self, msg: Odometry):
        self.pose = odom_to_pose(msg)

    def on_output_topic(self, msg: Image):
        if not self.own_frames.is_foreign(msg.data):
            return
        self.n_foreign += 1
        self.get_logger().warning(
            f"{self.n_foreign} frame(s) on {self.cfg['topic']} that this node did not publish: a second "
            "thermal source (normally the Gazebo thermal camera) publishes there - start the simulation "
            "with enable_thermal:=false, or use 'ros2 launch plant_simulation bringup.launch.py "
            "thermal_source:=synthetic'", throttle_duration_sec=30.0)

    def current_pose(self):
        if self.pose is None:
            if not self.cfg["publish_without_odom"]:
                return None
            x, y, _z, yaw = self.layout.spawn
            return full_pose(self.layout, x, y, yaw, float(self.cfg["body_height"]))
        x, y, z, roll, pitch, yaw = self.pose
        if not self.cfg["use_odom_z"]:
            z = self.layout.floor_z(x, y) + float(self.cfg["body_height"])
        return (x, y, z, roll, pitch, yaw)

    def on_timer(self):
        pose = self.current_pose()
        if pose is None:
            self.get_logger().warning(f"waiting for {self.cfg['odom_topic']} ...", throttle_duration_sec=10.0)
            return
        try:
            l16 = self.camera.render_l16(pose)
        except Exception as exc:
            self.get_logger().error(f"synthetic thermal render failed: {exc}", throttle_duration_sec=5.0)
            return
        msg = mono16_image_msg(l16, str(self.cfg["encoding"]), str(self.cfg["frame_id"]),
                               self.get_clock().now().to_msg())
        self.own_frames.add_own(msg.data)
        self.pub.publish(msg)
        self.n += 1
        if self.n == 1:
            tmax_c = l16.max() * float(self.cfg["resolution"]) - 273.15
            self.get_logger().info(f"first synthetic thermal frame published (max {tmax_c:.1f} C)")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SyntheticThermalNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
