#!/usr/bin/env python3
"""
Enhanced perception node for Shell Eco‑marathon autonomous stack.
LiDAR clustering + KF tracking. Publishes nearest obstacle and tracked obstacles.
"""
import math
import time
import sys # Only for sys.exit in case of critical init error, if desired
from typing import Dict, List

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial import cKDTree # For efficient nearest neighbor search

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile # Can be used for explicit QoS if desired

from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import PointCloud2, Image # Image still present for placeholder
from geometry_msgs.msg import PoseArray, Pose
import sensor_msgs_py.point_cloud2 as pc2

def make_cvkf(dt: float = 0.1, q_noise_std: float = 1.0, r_meas_std: float = 0.5) -> KalmanFilter:
    """Return a 4‑state constant‑velocity KF (x, y, vx, vy)."""
    kf = KalmanFilter(dim_x=4, dim_z=2) # State: x,y,vx,vy; Measurement: x,y
    kf.F = np.array([[1, 0, dt, 0],     # State transition matrix
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],     # Measurement function
                     [0, 1, 0, 0]])
    kf.P *= 10.0                       # Initial state covariance (uncertainty)
    kf.R = np.eye(2) * (r_meas_std**2) # Measurement noise covariance
    
    # Process noise covariance ( discretized white noise model for acceleration)
    # q_block accounts for uncertainty in the constant velocity model
    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4
    q_block = np.array([[dt4/4, dt3/2], [dt3/2, dt2]]) * (q_noise_std**2)
    kf.Q = np.block([[q_block, np.zeros((2,2))],
                     [np.zeros((2,2)), q_block]])
    return kf

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('lidar_topic', '/carla/ego_vehicle/vlp16_1')
        self.declare_parameter('camera_topic', '/carla/ego_vehicle/depth_middle/image')
        self.declare_parameter('cone_width_m', 4.0)
        self.declare_parameter('cone_length_m', 25.0)
        self.declare_parameter('cluster_eps_m', 0.6)
        self.declare_parameter('min_cluster_points', 5)
        self.declare_parameter('alert_distance_m', 5.0)
        self.declare_parameter('kf_association_gate_m', 2.0)
        self.declare_parameter('track_timeout_s', 1.0)
        self.declare_parameter('kf_dt_s', 0.1) # Assumed dt for KF if not dynamically calculated
        self.declare_parameter('kf_process_noise_std', 1.0)
        self.declare_parameter('kf_measurement_noise_std', 0.5)
        self.declare_parameter('debug_log', False)

        lidar_topic = self.get_parameter('lidar_topic').value
        camera_topic = self.get_parameter('camera_topic').value
        self.cone_width = self.get_parameter('cone_width_m').value
        self.cone_length = self.get_parameter('cone_length_m').value
        self.cluster_eps = self.get_parameter('cluster_eps_m').value
        self.min_cluster_size = self.get_parameter('min_cluster_points').value
        self.alert_distance = self.get_parameter('alert_distance_m').value
        self.kf_gate_m = self.get_parameter('kf_association_gate_m').value
        self.track_timeout_s = self.get_parameter('track_timeout_s').value
        self.kf_dt = self.get_parameter('kf_dt_s').value
        self.kf_q_std = self.get_parameter('kf_process_noise_std').value
        self.kf_r_std = self.get_parameter('kf_measurement_noise_std').value
        self.debug_log_enabled = self.get_parameter('debug_log').value

        qos_profile_default = 10 # Your original QoS
        self.pub_nearest_dist = self.create_publisher(Float32, 'nearest_obstacle_distance', qos_profile_default)
        self.pub_obstacle_alert = self.create_publisher(Bool, 'obstacle_alert', qos_profile_default)
        self.pub_tracked_obstacles = self.create_publisher(PoseArray,'tracked_obstacles', qos_profile_default)

        self.create_subscription(PointCloud2, lidar_topic, self.on_lidar_received, qos_profile_default)
        self.create_subscription(Image, camera_topic, self.on_camera_received, qos_profile_default) # Placeholder

        self.next_track_id: int = 0
        self.active_tracks: Dict[int, KalmanFilter] = {}
        self.track_last_update_time: Dict[int, float] = {}
        
        self.last_debug_log_time_s = time.time()
        self.get_logger().info('Perception node ready.')

    def on_lidar_received(self, msg: PointCloud2):
        # Extract 2D points (x, y) from the point cloud
        points_xy = np.array([[p[0], p[1]]
                              for p in pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True)])
        if points_xy.size == 0:
            # If no points, still need to run prediction and timeout for existing tracks
            # and publish default values.
            current_time_s = self.get_clock().now().nanoseconds * 1e-9
            self.update_kf_tracks(np.array([]), current_time_s) # Pass empty observations
            self.publish_obstacle_data(float('inf'), [], current_time_s)
            return

        # Filter points to a forward cone region
        mask = (points_xy[:, 0] > 0.1) & \
               (points_xy[:, 0] < self.cone_length) & \
               (np.abs(points_xy[:, 1]) < self.cone_width / 2.0) # Corrected to half-width
        
        points_in_cone = points_xy[mask]

        if points_in_cone.shape[0] < self.min_cluster_size:
            current_time_s = self.get_clock().now().nanoseconds * 1e-9
            self.update_kf_tracks(np.array([]), current_time_s) # Pass empty observations
            self.publish_obstacle_data(float('inf'), [], current_time_s)
            return

        # Cluster points using KDTree ball query (approximates DBSCAN)
        tree = cKDTree(points_in_cone)
        point_clusters = []
        visited_indices = np.zeros(len(points_in_cone), dtype=bool)
        for i in range(len(points_in_cone)):
            if visited_indices[i]:
                continue
            # Find neighbors within cluster_eps radius
            neighbor_indices = tree.query_ball_point(points_in_cone[i], r=self.cluster_eps)
            if len(neighbor_indices) >= self.min_cluster_size:
                # Mark all points in this new cluster (including core point's neighbors) as visited
                for ni in neighbor_indices: visited_indices[ni] = True 
                point_clusters.append(points_in_cone[neighbor_indices])
        
        # Get centroids of clusters as observations for KF
        cluster_observations = np.array([cl.mean(axis=0) for cl in point_clusters]) if point_clusters else np.array([])
        
        current_time_s = self.get_clock().now().nanoseconds * 1e-9
        self.update_kf_tracks(cluster_observations, current_time_s)
        
        # Prepare data for publishing
        min_forward_dist_m = float('inf')
        tracked_poses: List[Pose] = []
        for track_id, kf_filter in self.active_tracks.items():
            pos_x, pos_y = kf_filter.x[0,0], kf_filter.x[1,0] # kf.x is a column vector
            vel_x, vel_y = kf_filter.x[2,0], kf_filter.x[3,0]

            if pos_x > 0: # Consider only forward obstacles for min_dist
                min_forward_dist_m = min(min_forward_dist_m, pos_x)
            
            pose = Pose()
            pose.position.x = float(pos_x)
            pose.position.y = float(pos_y)
            pose.position.z = 0.0 # Assuming ground plane
            # Hack: Encode velocities in orientation
            pose.orientation.w = float(vel_x) 
            pose.orientation.x = float(vel_y)
            pose.orientation.y = 0.0 # Keep y,z as 0 for this hack to be clear
            pose.orientation.z = 0.0
            tracked_poses.append(pose)
            
        self.publish_obstacle_data(min_forward_dist_m, tracked_poses, current_time_s)

    def update_kf_tracks(self, observations_xy: np.ndarray, current_time_s: float):
        # Prediction step for all existing tracks
        for kf_filter in self.active_tracks.values():
            # dt for KF prediction can be fixed (self.kf_dt) or dynamic.
            # Dynamic dt would require storing last update time per track for more accuracy.
            # For simplicity here, using fixed self.kf_dt in make_cvkf,
            # but prediction step itself doesn't need dt if F matrix was pre-calculated with it.
            kf_filter.predict()

        # Data Association (simple nearest neighbor with gating)
        used_observation_indices = set()
        associated_track_ids = []

        if observations_xy.size > 0: # Only associate if there are observations
            for track_id, kf_filter in list(self.active_tracks.items()): # list() for safe deletion
                predicted_pos = kf_filter.x[:2].reshape(1,2) # kf.x is (4,1), so take first 2 elements
                
                min_dist_sq_to_obs = float('inf')
                best_obs_idx = -1

                for j, obs_pos in enumerate(observations_xy):
                    if j in used_observation_indices:
                        continue
                    
                    # Using squared Euclidean distance for efficiency before sqrt
                    dist_sq = np.sum((obs_pos - predicted_pos)**2)
                    if dist_sq < min_dist_sq_to_obs:
                        min_dist_sq_to_obs = dist_sq
                        best_obs_idx = j
                
                if best_obs_idx != -1 and math.sqrt(min_dist_sq_to_obs) < self.kf_gate_m:
                    kf_filter.update(observations_xy[best_obs_idx].reshape(2,1)) # obs needs to be (2,1)
                    self.track_last_update_time[track_id] = current_time_s
                    used_observation_indices.add(best_obs_idx)
                    associated_track_ids.append(track_id)

        # Create new tracks for unused (unassociated) observations
        if observations_xy.size > 0:
            for j, obs_pos in enumerate(observations_xy):
                if j not in used_observation_indices:
                    kf = make_cvkf(dt=self.kf_dt, q_noise_std=self.kf_q_std, r_meas_std=self.kf_r_std)
                    kf.x[:2] = obs_pos.reshape(2,1) # Initialize position
                    # Initial velocity is often zero or estimated from first few observations
                    # Here, make_cvkf sets up F, H, P, Q, R. x will have 0 initial velocity.
                    
                    new_id = self.next_track_id
                    self.next_track_id += 1
                    self.active_tracks[new_id] = kf
                    self.track_last_update_time[new_id] = current_time_s

        # Delete stale tracks (not updated recently)
        stale_track_ids = []
        for track_id, last_seen in self.track_last_update_time.items():
            if (current_time_s - last_seen) > self.track_timeout_s:
                stale_track_ids.append(track_id)
        
        for track_id in stale_track_ids:
            if track_id in self.active_tracks: del self.active_tracks[track_id]
            if track_id in self.track_last_update_time: del self.track_last_update_time[track_id]

    def on_camera_received(self, msg: Image):
        # Placeholder: "If resolution changes, we could recompute ROI here in future."
        pass

    def publish_obstacle_data(self, nearest_dist_m: float, poses: List[Pose], stamp_s: float):
        msg_dist = Float32()
        msg_dist.data = nearest_dist_m if math.isfinite(nearest_dist_m) else float('inf')
        self.pub_nearest_dist.publish(msg_dist)

        msg_alert_val = Bool()
        msg_alert_val.data = nearest_dist_m < self.alert_distance
        self.pub_obstacle_alert.publish(msg_alert_val)

        if poses: # Only publish PoseArray if there are poses
            pose_array_msg = PoseArray()
            # Convert float seconds to rclpy.time.Time then to_msg()
            pose_array_msg.header.stamp = rclpy.time.Time(seconds=stamp_s).to_msg()
            pose_array_msg.header.frame_id = 'ego_vehicle' # As per original
            pose_array_msg.poses = poses
            self.pub_tracked_obstacles.publish(pose_array_msg)

        if self.debug_log_enabled and (time.time() - self.last_debug_log_time_s > 1.0):
            self.get_logger().info(f"Nearest: {nearest_dist_m:.2f}m, Tracks: {len(self.active_tracks)}")
            self.last_debug_log_time_s = time.time()

def main(args=None):
    rclpy.init(args=args)
    perception_node = PerceptionNode()
    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.get_logger().info("Shutting down perception node.")
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()