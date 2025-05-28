#!/usr/bin/env python3
"""
Control node (rev-3b) — Shell Eco‑marathon autonomous stack.
Goal Proximity Check & Acknowledgement.
Pure‑Pursuit, PID throttle, Eco‑coast, Optional speed override, Safety watchdog.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_msgs.msg import Float32, Float64, String, Bool
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

LOOKAHEAD_MIN_DEFAULT = 4.0      # m
LOOKAHEAD_TIME_DEFAULT = 0.5     # s  (L = Lmin + t * v)
GOAL_REACHED_THRESHOLD_M_DEFAULT = 3.0 # Competition spec
DEFAULT_WHEELBASE = 2.8          # m, example for CARLA standard vehicles
MAX_STEER_ANGLE_RAD_DEFAULT = math.radians(35.0) # Example

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        self.declare_parameter('kp', 0.4)
        self.declare_parameter('ki', 0.05)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('i_clamp', 0.3)
        self.declare_parameter('cruise_speed_fallback', 8.0)
        self.declare_parameter('coast_trigger_ttc_s', 6.0)
        self.declare_parameter('hard_brake_trigger_ttc_s', 2.0)
        self.declare_parameter('lookahead_min_m', LOOKAHEAD_MIN_DEFAULT)
        self.declare_parameter('lookahead_time_s', LOOKAHEAD_TIME_DEFAULT)
        self.declare_parameter('goal_reached_threshold_m', GOAL_REACHED_THRESHOLD_M_DEFAULT)
        self.declare_parameter('max_speed_mph', 30.0)
        self.declare_parameter('vehicle_wheelbase_m', DEFAULT_WHEELBASE)
        self.declare_parameter('max_steer_angle_rad', MAX_STEER_ANGLE_RAD_DEFAULT)


        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.i_clamp = self.get_parameter('i_clamp').value
        self.v_cruise_fallback = self.get_parameter('cruise_speed_fallback').value
        self.coast_trigger_ttc_s = self.get_parameter('coast_trigger_ttc_s').value
        self.hard_brake_trigger_ttc_s = self.get_parameter('hard_brake_trigger_ttc_s').value
        self.lookahead_min_m = self.get_parameter('lookahead_min_m').value
        self.lookahead_time_s = self.get_parameter('lookahead_time_s').value
        self.goal_reached_threshold_m = self.get_parameter('goal_reached_threshold_m').value
        self.max_speed_mps = self.get_parameter('max_speed_mph').value * 0.44704
        self.wheelbase = self.get_parameter('vehicle_wheelbase_m').value
        self.max_steer_angle_rad = self.get_parameter('max_steer_angle_rad').value

        qos_cmd = 10 # Your original for commands
        qos_latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.pub_steer   = self.create_publisher(Float64, 'steering_command', qos_cmd)
        self.pub_throttle= self.create_publisher(Float64, 'throttle_command', qos_cmd)
        self.pub_brake   = self.create_publisher(Float64, 'brake_command', qos_cmd)
        self.pub_gear    = self.create_publisher(String, 'gear_command', qos_latched)
        self.pub_goal_ack = self.create_publisher(Bool, '/goal_reached_ack', qos_cmd)

        self.pub_gear.publish(String(data='forward'))

        self.path: Optional[Path] = None
        self.waypoints_np: Optional[np.ndarray] = None
        self.speeds_np: Optional[np.ndarray] = None
        self.current_high_level_goal_np: Optional[np.ndarray] = None
        self.nearest_obstacle_m = float('inf')
        self.desired_speed_override_mps: Optional[float] = None
        self.mission_complete_flag = False

        self.create_subscription(Path, 'nav_path', self.on_path_received, qos_cmd)
        self.create_subscription(PoseStamped, '/current_target_waypoint', self.on_current_hl_goal, qos_cmd)
        self.create_subscription(Float32, 'nearest_obstacle_distance', self.on_obstacle_dist, qos_cmd)
        self.create_subscription(Float32, 'desired_speed', self.on_desired_speed_override, qos_cmd)
        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.on_odometry_update, qos_cmd)
        self.create_subscription(Bool, '/mission_complete', self.on_mission_status_update, qos_latched)

        self.pid_prev_error = 0.0
        self.pid_integral_error = 0.0
        self.last_odom_time = self.get_clock().now()
        self.last_cmd_publish_time_s = time.time()
        self.watchdog_timer = self.create_timer(0.2, self.run_watchdog_check)

        self.get_logger().info('Control node ready.')

    def on_path_received(self, msg: Path):
        if not msg.poses:
            self.path, self.waypoints_np, self.speeds_np = None, None, None
            return
        self.path = msg
        self.waypoints_np = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self.speeds_np    = np.array([p.pose.orientation.z for p in msg.poses])

    def on_obstacle_dist(self, msg: Float32):
        self.nearest_obstacle_m = msg.data

    def on_desired_speed_override(self, msg: Float32):
        self.desired_speed_override_mps = msg.data
        
    def on_mission_status_update(self, msg: Bool):
        self.mission_complete_flag = msg.data
        if self.mission_complete_flag:
            self.get_logger().info("Mission complete signal received.")

    def on_current_hl_goal(self, msg: PoseStamped):
        self.current_high_level_goal_np = np.array([msg.pose.position.x, msg.pose.position.y])

    def on_odometry_update(self, msg: Odometry):
        current_time = self.get_clock().now()
        dt_s = (current_time - self.last_odom_time).nanoseconds * 1e-9
        if dt_s <= 1e-3: return
        self.last_odom_time = current_time

        ego_x_m = msg.pose.pose.position.x
        ego_y_m = msg.pose.pose.position.y
        ego_v_mps = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        ego_pos_np = np.array([ego_x_m, ego_y_m])

        if self.current_high_level_goal_np is not None and not self.mission_complete_flag:
            dist_to_hl_goal_m = np.linalg.norm(self.current_high_level_goal_np - ego_pos_np)
            if dist_to_hl_goal_m < self.goal_reached_threshold_m:
                self.pub_goal_ack.publish(Bool(data=True))
                self.current_high_level_goal_np = None 

        if self.path is None or self.waypoints_np is None or self.waypoints_np.shape[0] < 2 or self.speeds_np is None:
            self.pub_throttle.publish(Float64(data=0.0))
            self.pub_brake.publish(Float64(data=1.0))
            self.pub_steer.publish(Float64(data=0.0))
            self.last_cmd_publish_time_s = time.time()
            return

        lookahead_dist_m = max(self.lookahead_min_m, self.lookahead_time_s * ego_v_mps)
        dists_to_path_wps_m = np.linalg.norm(self.waypoints_np - ego_pos_np, axis=1)
        closest_idx = np.argmin(dists_to_path_wps_m)

        lookahead_target_idx = -1
        for i in range(closest_idx, len(self.waypoints_np)):
            if dists_to_path_wps_m[i] > lookahead_dist_m:
                 lookahead_target_idx = i
                 break
        if lookahead_target_idx == -1:
            lookahead_target_idx = len(self.waypoints_np) - 1

        target_path_pt_np = self.waypoints_np[lookahead_target_idx]
        dx_world_m = target_path_pt_np[0] - ego_x_m
        dy_world_m = target_path_pt_np[1] - ego_y_m

        q = msg.pose.pose.orientation
        # Yaw approximation (ensure w is positive for atan2's range)
        yaw_rad = 2 * math.atan2(q.z * math.copysign(1.0, q.w), abs(q.w))

        dy_vehicle_m = -dx_world_m * math.sin(yaw_rad) + dy_world_m * math.cos(yaw_rad)
        actual_dist_to_target_m = math.hypot(dx_world_m, dy_world_m)
        
        curvature = 0.0
        if actual_dist_to_target_m > 1e-3:
            curvature = (2 * dy_vehicle_m) / (actual_dist_to_target_m**2)
        
        steer_angle_rad = math.atan(self.wheelbase * curvature)
        steer_cmd_norm = np.clip(steer_angle_rad / self.max_steer_angle_rad, -1.0, 1.0)

        v_target_path_mps = self.speeds_np[lookahead_target_idx] if lookahead_target_idx >= 0 else self.v_cruise_fallback
        v_target_mps = v_target_path_mps
        
        if self.desired_speed_override_mps is not None:
            v_target_mps = min(v_target_mps, self.desired_speed_override_mps)
        v_target_mps = min(v_target_mps, self.max_speed_mps)

        if self.nearest_obstacle_m < float('inf') and ego_v_mps > 0.1:
            ttc_s = self.nearest_obstacle_m / ego_v_mps
            if ttc_s < self.hard_brake_trigger_ttc_s:
                v_target_mps = 0.0
            elif ttc_s < self.coast_trigger_ttc_s:
                v_target_mps = min(v_target_mps, 0.3 + 0.2 * ttc_s) 
                v_target_mps = max(0.0, v_target_mps)

        error_mps = v_target_mps - ego_v_mps
        self.pid_integral_error += error_mps * dt_s
        self.pid_integral_error = np.clip(self.pid_integral_error, -self.i_clamp, self.i_clamp)
        
        derivative_mps = (error_mps - self.pid_prev_error) / dt_s if dt_s > 1e-3 else 0.0
        self.pid_prev_error = error_mps
        
        pid_output_u = self.kp * error_mps + self.ki * self.pid_integral_error + self.kd * derivative_mps

        throttle_cmd_norm = np.clip(pid_output_u, 0.0, 1.0)
        brake_cmd_norm = np.clip(-pid_output_u, 0.0, 1.0) if pid_output_u < 0 else 0.0

        if v_target_mps > 0.1 and error_mps < 0 and brake_cmd_norm > 0 and brake_cmd_norm < 0.3:
            brake_cmd_norm = 0.0

        if self.mission_complete_flag and dists_to_path_wps_m[-1] < 1.0 and ego_v_mps < 0.5:
             throttle_cmd_norm = 0.0
             brake_cmd_norm = 1.0

        self.pub_steer.publish(Float64(data=float(steer_cmd_norm)))
        self.pub_throttle.publish(Float64(data=float(throttle_cmd_norm)))
        self.pub_brake.publish(Float64(data=float(brake_cmd_norm)))
        self.last_cmd_publish_time_s = time.time()

    def run_watchdog_check(self):
        if time.time() - self.last_cmd_publish_time_s > 0.2:
            self.get_logger().warn("Watchdog: No control commands recently. Stopping vehicle.")
            self.pub_throttle.publish(Float64(data=0.0))
            self.pub_brake.publish(Float64(data=1.0))
            self.pub_steer.publish(Float64(data=0.0))

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down control node.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()