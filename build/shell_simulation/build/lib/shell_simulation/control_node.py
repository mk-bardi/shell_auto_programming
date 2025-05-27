#!/usr/bin/env python3
from __future__ import annotations
import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from std_msgs.msg import Float32, Float64, Bool, String
from nav_msgs.msg import Odometry, Path

def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy

class ControlNode(Node):
    def __init__(self) -> None:
        super().__init__('control_node')

        self.declare_parameter('wheel_base', 2.8)
        self.declare_parameter('pp_lookahead_time', 0.5)
        self.declare_parameter('pp_L_min', 2.0)
        self.declare_parameter('pp_L_max', 15.0)
        self.declare_parameter('speed_pid.k_p', 0.8)
        self.declare_parameter('speed_pid.k_i', 0.1)
        self.declare_parameter('speed_pid.k_d', 0.0)
        self.declare_parameter('throttle_jerk_max', 0.4)
        self.declare_parameter('brake_jerk_max', 0.6)
        self.declare_parameter('throttle_alpha', 0.2)
        self.declare_parameter('steer_rate_max', 0.04)
        self.declare_parameter('speed_limit_mps', 13.0)
        self.declare_parameter('target_speed_mps', 8.0)
        self.declare_parameter('control_period', 0.05)
        self.declare_parameter('obs_brake_full', 3.0)
        self.declare_parameter('obs_brake_start', 6.0)

        p = self.get_parameter
        self.Lf = p('wheel_base').value
        self.tau = p('control_period').value
        self.L_time = p('pp_lookahead_time').value
        self.L_min = p('pp_L_min').value
        self.L_max = p('pp_L_max').value
        self.k_p = p('speed_pid.k_p').value
        self.k_i = p('speed_pid.k_i').value
        self.k_d = p('speed_pid.k_d').value
        self.t_jerk = p('throttle_jerk_max').value
        self.b_jerk = p('brake_jerk_max').value
        self.alpha = p('throttle_alpha').value
        self.steer_rate_max = p('steer_rate_max').value
        self.speed_cap = p('speed_limit_mps').value
        self.speed_cruise = p('target_speed_mps').value
        self.obs_full = p('obs_brake_full').value
        self.obs_start = p('obs_brake_start').value

        self.pub_throttle = self.create_publisher(Float64, '/throttle_command', 10)
        self.pub_brake = self.create_publisher(Float64, '/brake_command', 10)
        self.pub_steer = self.create_publisher(Float64, '/steering_command', 10)
        self.pub_gear = self.create_publisher(String, '/gear_command', 10)
        self.pub_hand = self.create_publisher(Bool, '/handbrake_command', 10)

        sensor_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(Path, '/nav_path', self.cb_path, qos_profile=sensor_qos)
        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.cb_odom, qos_profile=sensor_qos)
        self.create_subscription(Float32, '/carla/ego_vehicle/speedometer', self.cb_speed, qos_profile=sensor_qos)
        self.create_subscription(Float32, '/nearest_obstacle_distance', self.cb_obs, qos_profile=sensor_qos)
        self.create_subscription(Bool, '/mission_complete', self.cb_mission, qos_profile=sensor_qos)

        self.path_xy: List[Tuple[float, float]] = []
        self.path_s: List[float] = []
        self.odom: Optional[Odometry] = None
        self.speed_mps = 0.0
        self.obs_dist_m = float('inf')
        self.mission_done = False

        self.spi = 0.0
        self.prev_err = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_steer = 0.0

        gear = String(); gear.data = "forward"
        self.pub_gear.publish(gear)
        self.pub_hand.publish(Bool(data=False))

        self.create_timer(self.tau, self.control_loop)
        self.get_logger().info("Control node ready (Pure-Pursuit + PID).")

    def cb_path(self, msg: Path) -> None:
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(pts) < 2:
            self.get_logger().warn("nav_path too short.")
            return
        self.path_xy = pts
        self.path_s = [0.0]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            self.path_s.append(self.path_s[-1] + math.hypot(x1 - x0, y1 - y0))
        self.get_logger().info(f"Received nav_path with {len(self.path_xy)} pts.")

    def cb_odom(self, msg: Odometry) -> None:
        self.odom = msg

    def cb_speed(self, msg: Float32) -> None:
        self.speed_mps = msg.data * 0.277778

    def cb_obs(self, msg: Float32) -> None:
        self.obs_dist_m = msg.data

    def cb_mission(self, msg: Bool) -> None:
        if msg.data and not self.mission_done:
            self.get_logger().info("Mission complete flag received; braking to stop.")
        self.mission_done = msg.data

    def control_loop(self) -> None:
        if not self.path_xy or self.odom is None:
            return

        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        q = self.odom.pose.pose.orientation

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        nearest_idx = min(range(len(self.path_xy)),
                          key=lambda i: _dist2(self.path_xy[i], (px, py)))

        L = max(self.L_min, min(self.L_max, self.L_time * self.speed_mps))

        goal_idx = nearest_idx
        while goal_idx < len(self.path_s) - 1 and \
              self.path_s[goal_idx] - self.path_s[nearest_idx] < L:
            goal_idx += 1
        gx, gy = self.path_xy[goal_idx]

        dx = math.cos(-yaw) * (gx - px) - math.sin(-yaw) * (gy - py)
        dy = math.sin(-yaw) * (gx - px) + math.cos(-yaw) * (gy - py)

        if dx <= 0.1:
            steer_raw = 0.0
        else:
            curvature = 2 * dy / (dx * dx + dy * dy)
            steer_raw = math.atan(curvature * self.Lf)

        steer_cmd = self.prev_steer + \
                    max(-self.steer_rate_max, min(self.steer_rate_max,
                                                  steer_raw - self.prev_steer))
        steer_cmd = max(-1.0, min(1.0, steer_cmd))

        v_des = 0.0 if self.mission_done else self.speed_cruise
        if self.obs_dist_m < self.obs_start:
            factor = max(0.0,
                         (self.obs_dist_m - self.obs_full) /
                         (self.obs_start - self.obs_full))
            v_des *= factor
        v_des = min(v_des, self.speed_cap - 0.3)

        err = v_des - self.speed_mps
        self.spi += err * self.tau
        derr = (err - self.prev_err) / self.tau
        self.prev_err = err

        u_raw = self.k_p * err + self.k_i * self.spi + self.k_d * derr
        u_raw = max(-1.0, min(1.0, u_raw))

        throttle_des = max(0.0, u_raw)
        brake_des = max(0.0, -u_raw)

        throttle_cmd = self.prev_throttle + \
                       max(-self.t_jerk * self.tau, min(self.t_jerk * self.tau,
                                                        throttle_des - self.prev_throttle))
        brake_cmd = self.prev_brake + \
                    max(-self.b_jerk * self.tau, min(self.b_jerk * self.tau,
                                                     brake_des - self.prev_brake))

        throttle_cmd = self.alpha * throttle_cmd + (1 - self.alpha) * self.prev_throttle

        if brake_cmd > 0.05:
            throttle_cmd = 0.0
        if throttle_cmd > 0.05:
            brake_cmd = 0.0

        if self.speed_mps > self.speed_cap:
            throttle_cmd = 0.0
            brake_cmd = max(brake_cmd, 0.3)

        if self.mission_done and self.speed_mps < 0.3:
            throttle_cmd = 0.0
            brake_cmd = 1.0

        self.pub_steer.publish(Float64(data=steer_cmd))
        self.pub_throttle.publish(Float64(data=throttle_cmd))
        self.pub_brake.publish(Float64(data=brake_cmd))

        self.prev_throttle, self.prev_brake, self.prev_steer = \
            throttle_cmd, brake_cmd, steer_cmd

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_throttle.publish(Float64(data=0.0))
        node.pub_brake.publish(Float64(data=1.0))
        node.get_logger().info("Control node shut down — brakes applied.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
