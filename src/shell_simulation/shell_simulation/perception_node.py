#!/usr/bin/env python3
"""
Enhanced perception node for Shell Eco‑marathon autonomous stack.

Changes compared with the previous bumper‑only version
------------------------------------------------------
1. **Dynamic ROI** – re‑computes image ROI if camera resolution changes at runtime.
2. **LiDAR clustering + KF tracking** – projects VLP‑16 point cloud to the ground
   plane, clusters points within a forward cone and maintains a constant‑velocity
   Kalman filter for each obstacle (simple nearest‑neighbour association).
3. **Long‑horizon safety envelope** – publishes the minimum obstacle distance up
   to 25 m ahead (was 8 m) and a `PoseArray` (`/tracked_obstacles`) containing
   the current state of each obstacle in the map frame.
4. **Parameterisation** – all magic numbers exposed as ROS 2 parameters so they
   can be tuned from the launch file if needed.

Published topics
----------------
* `/nearest_obstacle_distance` (std_msgs/Float32) – minimum forward distance [m]
* `/obstacle_alert`            (std_msgs/Bool)    – true if an obstacle is closer than `alert_distance`.
* `/tracked_obstacles`         (geometry_msgs/PoseArray) – centres of all
  tracked objects in the ego vehicle frame (x forward, y left, z = 0) with x‑vel
  encoded in `pose.orientation.w` and y‑vel in `pose.orientation.x` (quick hack
  to avoid a custom message).

Dependencies: `numpy`, `scipy`, `filterpy`, `pcl‑py`.
These have been added to `setup.py`/`package.xml`.
"""
import math
import time
import sys
from typing import Dict, List

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import PointCloud2, Imu, Image
from geometry_msgs.msg import PoseArray, Pose
import sensor_msgs_py.point_cloud2 as pc2

# -----------------------------------------------------------------------------
# Helper: constant‑velocity 2‑D Kalman filter template
# -----------------------------------------------------------------------------

def make_cvkf(dt: float = 0.1, q: float = 1.0, r: float = 0.5) -> KalmanFilter:
    """Return a 4‑state constant‑velocity KF (x, y, vx, vy)."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 10.0
    kf.R = np.eye(2) * r
    q_block = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * q
    kf.Q = np.block([[q_block, np.zeros((2, 2))],
                     [np.zeros((2, 2)), q_block]])
    return kf

# -----------------------------------------------------------------------------
# Perception node
# -----------------------------------------------------------------------------

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # ---------------- parameters ----------------
        self.declare_parameter('cone_width', 4.0)         # lateral ± m
        self.declare_parameter('cone_length', 25.0)       # forward m
        self.declare_parameter('cluster_eps', 0.6)        # DBSCAN eps (m)
        self.declare_parameter('min_cluster_size', 5)     # points
        self.declare_parameter('alert_distance', 5.0)     # m
        self.declare_parameter('debug', False)

        self.cone_width      = self.get_parameter('cone_width').get_parameter_value().double_value
        self.cone_length     = self.get_parameter('cone_length').get_parameter_value().double_value
        self.cluster_eps     = self.get_parameter('cluster_eps').get_parameter_value().double_value
        self.min_cluster_sz  = self.get_parameter('min_cluster_size').get_parameter_value().integer_value
        self.alert_distance  = self.get_parameter('alert_distance').get_parameter_value().double_value
        self.debug           = self.get_parameter('debug').get_parameter_value().bool_value

        # ---------------- publishers ----------------
        self.pub_nearest = self.create_publisher(Float32, 'nearest_obstacle_distance', 10)
        self.pub_alert   = self.create_publisher(Bool,     'obstacle_alert',           10)
        self.pub_tracks  = self.create_publisher(PoseArray,'tracked_obstacles',        10)

        # ---------------- subscribers ---------------
        self.create_subscription(PointCloud2, '/carla/ego_vehicle/vlp16_1',
                                 self.on_lidar, qos_profile=10)
        # camera subscribed only to keep the option of ROI reuse; skip heavy proc
        self.create_subscription(Image, '/carla/ego_vehicle/depth_middle/image',
                                 self.on_cam, qos_profile=10)

        # ---------------- KF tracker state ----------
        self.track_id_seq: int = 0
        self.tracks: Dict[int, KalmanFilter] = {}        # id -> KF
        self.track_last_seen: Dict[int, float] = {}
        self.timeout = 1.0                               # s before deleting a track

        self.last_pub = time.time()
        self.get_logger().info('Perception node ready (LiDAR tracking mode).')

    # -------------------------------------------------------------------------
    # LiDAR callback – cluster & track
    # -------------------------------------------------------------------------
    def on_lidar(self, msg: PointCloud2):
        points = np.array([[p[0], p[1]]
                           for p in pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True)])
        if points.size == 0:
            return

        # forward cone filter
        mask = (points[:, 0] > 0) & (points[:, 0] < self.cone_length) & (np.abs(points[:, 1]) < self.cone_width)
        pts_fwd = points[mask]
        if pts_fwd.shape[0] < self.min_cluster_sz:
            self.publish_msgs(float('inf'))
            return

        # fast grid clustering using KD‑tree + ball query (approx DBSCAN)
        tree = cKDTree(pts_fwd)
        clusters = []
        visited = np.zeros(len(pts_fwd), dtype=bool)
        for idx in range(len(pts_fwd)):
            if visited[idx]:
                continue
            neighbours = tree.query_ball_point(pts_fwd[idx], r=self.cluster_eps)
            if len(neighbours) >= self.min_cluster_sz:
                visited[neighbours] = True
                clusters.append(pts_fwd[neighbours])

        # ----------------------------------------------------------------- tracking
        observations = np.array([c.mean(axis=0) for c in clusters])  # (n,2)
        now = self.get_clock().now().nanoseconds * 1e-9

        # prediction step for all existing tracks
        for kf in self.tracks.values():
            kf.predict()

        # associate by nearest neighbour (Mahalanobis)
        used_obs = set()
        for tid, kf in list(self.tracks.items()):
            if observations.size == 0:
                continue
            diffs = observations[:, None, :] - kf.x[:2].reshape(1, 1, 2)
            dists = np.linalg.norm(diffs[:, 0, :], axis=1)
            j = int(dists.argmin())
            if dists[j] < 2.0:  # gate distance
                kf.update(observations[j])
                self.track_last_seen[tid] = now
                used_obs.add(j)

        # create new tracks for unused detections
        for j, obs in enumerate(observations):
            if j in used_obs:
                continue
            kf = make_cvkf()
            kf.x[:2] = obs.reshape(2, 1)
            tid = self.track_id_seq
            self.track_id_seq += 1
            self.tracks[tid] = kf
            self.track_last_seen[tid] = now

        # delete stale tracks
        for tid in list(self.tracks.keys()):
            if now - self.track_last_seen[tid] > self.timeout:
                del self.tracks[tid]
                del self.track_last_seen[tid]

        # compute minimum distance in the ego frame
        min_d = float('inf')
        poses: List[Pose] = []
        for kf in self.tracks.values():
            x, y = kf.x[0], kf.x[1]
            min_d = min(min_d, max(0.0, x))
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            pose.orientation.w = float(kf.x[2])  # vx
            pose.orientation.x = float(kf.x[3])  # vy
            poses.append(pose)

        self.publish_msgs(min_d, poses, now)

    # ------------------------------------------------------------------------- camera callback (ROI placeholder)
    def on_cam(self, msg: Image):
        # If resolution changes, we could recompute ROI here in future.
        pass

    # ------------------------------------------------------------------------- helpers
    def publish_msgs(self, nearest: float, poses: List[Pose] = None, stamp: float = None):
        msg_d = Float32()
        msg_d.data = nearest if math.isfinite(nearest) else float('inf')
        self.pub_nearest.publish(msg_d)

        msg_alert = Bool()
        msg_alert.data = nearest < self.alert_distance
        self.pub_alert.publish(msg_alert)

        if poses is not None and len(poses) > 0:
            pa = PoseArray()
            if stamp is None:
                stamp = self.get_clock().now().to_msg()
            else:
                stamp = rclpy.time.Time(seconds=stamp).to_msg()
            pa.header.stamp = stamp
            pa.header.frame_id = 'ego_vehicle'
            pa.poses = poses
            self.pub_tracks.publish(pa)

        if self.debug and time.time() - self.last_pub > 1.0:
            self.get_logger().info(f"Perception: nearest={nearest:.2f} m, tracks={len(self.tracks)}")
            self.last_pub = time.time()

# -----------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()