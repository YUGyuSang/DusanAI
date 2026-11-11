"""
map_server_lite: publishes the 2-D navigation map as a latched OccupancyGrid.

  param map_yaml   ''  -> Layout.load(layout_file).nav_map_yaml()
  topic /map       nav_msgs/OccupancyGrid, transient-local (latched),
                   republished every `republish_period` s (fresh stamp) so
                   that late volatile subscribers also get it.

If the YAML/PGM is missing or unreadable the node rasterises a fallback map
from world_layout.yaml (plant_simulation.layout_map) and keeps retrying the
file, switching to it as soon as it appears.
"""
import os

import rclpy  # noqa: F401
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout
from dusan_common.ros_helpers import declare_params, latched_qos

from plant_simulation.layout_map import MAP_SERVER_DEFAULTS, rasterize_layout
from plant_simulation.node_utils import run_node


class MapServerLite(Node):
    def __init__(self):
        super().__init__("map_server_lite")
        self.p = declare_params(self, MAP_SERVER_DEFAULTS)
        self.layout = Layout.load(self.p["layout_file"] or None)
        self.map_yaml = self.p["map_yaml"] or self.layout.nav_map_yaml()
        self.pub = self.create_publisher(OccupancyGrid, self.p["topic"], latched_qos())
        self.grid = None
        self.from_file = False
        self._msg = None             # cached OccupancyGrid (re-stamped on every publish)
        self._load()
        self._publish()
        self.create_timer(max(0.5, float(self.p["republish_period"])), self._tick)

    def _load(self):
        """Load the map file; fall back to the layout raster when needed."""
        if os.path.exists(self.map_yaml):
            try:
                grid = OccupancyGridMap.load(self.map_yaml)
                grid.frame_id = self.p["frame_id"]
                self.grid, self.from_file, self._msg = grid, True, None
                n_occ = int((grid.occ == 100).sum())
                self.get_logger().info(
                    f"loaded {self.map_yaml}: {grid.width}x{grid.height} @ {grid.resolution:.3f} m, origin "
                    f"({grid.origin_x:.2f}, {grid.origin_y:.2f}), {n_occ} occupied cells")
                return
            except Exception as exc:
                self.get_logger().error(f"failed to load map {self.map_yaml}: {exc}", throttle_duration_sec=30.0)
        else:
            self.get_logger().warning(f"map file not found: {self.map_yaml}", throttle_duration_sec=30.0)
        if self.grid is None and self.p["fallback_to_layout"]:
            self.grid = rasterize_layout(self.layout)
            self.grid.frame_id = self.p["frame_id"]
            self.from_file, self._msg = False, None
            self.get_logger().warning(
                f"publishing a FALLBACK map rasterised from {self.layout.path} "
                f"({self.grid.width}x{self.grid.height}); run tools/generate_world.py to create the real map")

    def _publish(self):
        if self.grid is None:
            return
        if self._msg is None:
            self._msg = self.grid.to_msg()
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self._msg)

    def _tick(self):
        if not self.from_file:
            self._load()   # the generated map may appear later (or be fixed) -> switch to it
        self._publish()


def main(args=None):
    run_node(MapServerLite, args)


if __name__ == "__main__":
    main()
