"""Depth camera + terrain_mapper emulation for the headless Gazebo check (pure python).

The headless gz server has no rendering, so the depth camera of the robot
(spot.urdf.xacro ``depth_camera``: 212 x 120, hfov 1.5, 0.1 .. 4 m, 10 Hz,
mounted 0.34 m ahead of base_link and pitched 0.6 rad down) is replaced by a
ray cast against the world's COLLISION primitives (:func:`world.ray_hit`)
from the ground-truth camera pose.  Its depth image (metres along the optical
axis, +inf beyond the far clip, -inf inside the near clip: what Gazebo's
R_FLOAT32 depth image holds) then goes through the SAME, ROS-free code path as
``terrain_mapping_node`` (image fallback branch):

    depth_image_to_points -> camera_points_to_world (PointFilter, odometry pose)
    -> MapFallGuard.accept (no fusion while the robot is down, set_upright)
    -> ElevationMap.recenter / integrate            (10 Hz, every depth frame)
    -> ElevationMap.recenter / predict / to_cloud_xyz + TerrainClassifier.update
                                                     (publish_rate 5 Hz)

with the parameters of module1_locomotion/config/terrain.yaml and the friction
prior of world_layout.yaml.  The controller side rebuilds its map from the
published cloud (``ElevationMap.from_cloud_xyz``) and gets the classifier
output as /terrain/info (``set_terrain_info``), as the locomotion node does.

Differences to Ubuntu: Gazebo renders the VISUAL meshes (the collisions of
plant.sdf were measured from them, pallets / grating details differ), the
robot's own legs are not in the image, no rendering latency.
"""
from __future__ import annotations

import os
from dataclasses import fields as dc_fields
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from dusan_common import robot_model as rm
from dusan_common.camera import sensor_extrinsic
from dusan_common.geometry import pose_matrix

from module1_locomotion.elevation_map import (FALL_LOOKBACK, FALL_SETTLE, FALL_SNAPSHOT_PERIOD, ElevationMap,
                                              ElevationMapParams, MapFallGuard)
from module1_locomotion.pointcloud_utils import (DepthIntrinsics, PointFilter, camera_points_to_world,
                                                 depth_image_to_points)
from module1_locomotion.terrain_classifier import ClassifierParams, TerrainClassifier

from . import world as W

TERRAIN_YAML = os.path.join(W.ROOT, "module1_locomotion", "config", "terrain.yaml")


class DepthCamera:
    """Ray-cast depth camera over collision primitives."""

    def __init__(self, shapes: list, sensor: str = "depth_camera"):
        s = rm.SENSORS[sensor]
        self.sensor = sensor
        self.near, self.far = float(s["near"]), float(s["far"])
        self.rate = float(s["rate"])
        self.intr = DepthIntrinsics.from_sensor(sensor)
        a, b = self.intr.ray_grid()
        # link-frame ray directions (1, a, b): the ray parameter IS the optical-axis depth
        self.dirs = np.stack([np.ones_like(a), a, b], axis=-1).reshape(-1, 3)
        self.reach = self.far * float(np.max(np.linalg.norm(self.dirs, axis=1)))
        self.T_bc = sensor_extrinsic(sensor)
        self.shapes = [dict(sh, aabb=W.shape_aabb(sh)) for sh in shapes]

    def pose(self, base_pos: Sequence[float], base_rpy: Sequence[float]) -> np.ndarray:
        return pose_matrix(*[float(v) for v in base_pos], *[float(v) for v in base_rpy]) @ self.T_bc

    def render(self, base_pos: Sequence[float], base_rpy: Sequence[float]) -> np.ndarray:
        """(H, W) float32 depth image [m] of the camera at the base pose."""
        T = self.pose(base_pos, base_rpy)
        O = T[:3, 3]
        D = self.dirs @ T[:3, :3].T
        t = np.full(D.shape[0], np.inf)
        for sh in self.shapes:
            lo, hi = sh["aabb"]
            gap = np.maximum(np.maximum(lo - O, O - hi), 0.0)
            if float(np.linalg.norm(gap)) > self.reach:
                continue
            t = np.minimum(t, W.ray_hit(sh, O, D))
        depth = np.where(t > self.far, np.inf, np.where(t < self.near, -np.inf, t))
        return depth.reshape(self.intr.height, self.intr.width).astype(np.float32)


def load_terrain_params(path: str = TERRAIN_YAML) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get("terrain_mapper", {}).get("ros__parameters", {}))


class TerrainMapperSim:
    """terrain_mapping_node without ROS (same parameters, same calls)."""

    def __init__(self, params: Optional[Dict] = None, layout: Optional[str] = W.LAYOUT,
                 sensor: str = "depth_camera"):
        p = load_terrain_params() if params is None else dict(params)
        self.p = p
        self.sensor = sensor
        self.publish_rate = max(0.5, float(p.get("publish_rate", 5.0)))
        self.image_stride = max(1, int(p.get("image_stride", 1)))
        fd = PointFilter()
        f = dict(p.get("filter", {}) or {})
        self.filter = PointFilter(
            min_range=float(f.get("min_range", fd.min_range)), max_range=float(f.get("max_range", fd.max_range)),
            body_box_min=tuple(float(v) for v in f.get("body_box_min", fd.body_box_min)),
            body_box_max=tuple(float(v) for v in f.get("body_box_max", fd.body_box_max)),
            max_height_above_base=float(f.get("max_height_above_base", fd.max_height_above_base)),
            stride=max(1, int(f.get("point_stride", fd.stride))))
        mp = {k: v for k, v in (p.get("map", {}) or {}).items() if k in {x.name for x in dc_fields(ElevationMapParams)}}
        cp = {k: v for k, v in (p.get("classifier", {}) or {}).items() if k in {x.name for x in dc_fields(ClassifierParams)}}
        self.emap = ElevationMap(params=ElevationMapParams.from_dict(mp))
        self.fall_guard = MapFallGuard(self.emap, lookback=float(p.get("fall_lookback", FALL_LOOKBACK)),
                                       settle=float(p.get("fall_settle", FALL_SETTLE)),
                                       period=float(p.get("fall_snapshot_period", FALL_SNAPSHOT_PERIOD)),
                                       enabled=bool(p.get("fall_guard", True)))
        self.classifier = TerrainClassifier(ClassifierParams.from_dict(cp))
        if layout and bool(p.get("friction_prior", True)):
            from module1_locomotion.friction_map import FrictionMap
            try:
                self.classifier.set_friction_map(FrictionMap.load(layout, max_mu=float(p.get("friction_zone_max_mu",
                                                                                               0.6))))
            except (OSError, ValueError, KeyError):
                pass
        self.intr = DepthIntrinsics.from_sensor(sensor)
        self.n_frames = 0
        self.info: Dict = {}

    def set_upright(self, upright: bool, t: float):
        """_on_loco_state: fall guard (robot_down of /locomotion/state)."""
        return self.fall_guard.set_upright(upright, t)

    def integrate_depth(self, depth: np.ndarray, pose: Sequence[float], t: float, extra_points=None) -> int:
        """_on_image + _integrate: depth image [m] at the odometry pose (x, y, z, roll, pitch, yaw).
        ``extra_points``: (N, 3) camera-link points added to the image's (test: phantom returns)."""
        if not self.fall_guard.accept(t):
            return 0
        d = np.asarray(depth, dtype=np.float64).copy()
        d[~np.isfinite(d)] = np.nan
        pts = depth_image_to_points(d, self.intr, stride=self.image_stride,
                                    min_depth=self.filter.min_range, max_depth=self.filter.max_range)
        if extra_points is not None and len(extra_points):
            pts = np.vstack([pts.reshape(-1, 3), np.asarray(extra_points, dtype=np.float64).reshape(-1, 3)])
        if pts.shape[0] == 0:
            return 0
        pw, rng, origin = camera_points_to_world(pts, pose, "link", self.sensor, self.filter)
        self.emap.recenter(float(pose[0]), float(pose[1]))
        self.emap.integrate(pw, origin, t=t, ranges=rng)
        self.n_frames += 1
        return int(pw.shape[0])

    def publish(self, pose: Sequence[float], t: float) -> Tuple[np.ndarray, Dict]:
        """_on_timer: (organised /elevation_map/points cloud, /terrain/info dict)."""
        self.emap.recenter(float(pose[0]), float(pose[1]))
        self.emap.predict(t)
        cloud = self.emap.to_cloud_xyz()
        info = self.classifier.update(self.emap, pose, t)
        info["stamp"] = t
        info["frame_id"] = "odom"
        info["source"] = "image"
        self.info = info
        return cloud, info
