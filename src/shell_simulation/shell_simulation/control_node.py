#!/usr/bin/env python3
"""
Control node (rev‑2) — Shell Eco‑marathon autonomous stack.

Upgrades
~~~~~~~~
1. **Pure‑Pursuit with dynamic look‑ahead**   `L = L0 + k * v`  (time‑based)
2. **PID throttle with anti‑wind‑up clamp**
3. **Eco‑coast mode** – if predicted Time‑to‑Collision (TTC) w.r.t. nearest
   obstacle < `coast_trigger_s` we cut throttle (*do not* brake) so kinetic
   energy bleeds off instead of turning into heat.
4. **Optional `/desired_speed` override** – if a speed‑planner node exists it
   can publish a float; otherwise the node uses the target speed encoded in
   `orientation.z` of the current path pose.
5. **Safety watchdog** – if no control cycle for >200 ms, node applies full
   brake and neutral steering.

Published topics
----------------
* `/steering_command`   std_msgs/Float64   (‑1 … 1)
* `/throttle_command`   std_msgs/Float64   (0 … 1)
* `/brake_command`      std_msgs/Float64   (0 … 1)
* `/gear_command`       std_msgs/String    ( latched "forward" )

Subscribed topics
-----------------
* `/nav_path`                   nav_msgs/Path    – centre‑line with target v
* `/nearest_obstacle_distance`  std_msgs/Float32 – from perception node
* `/desired_speed` (optional)   std_msgs/Float32 – external speed planner
* `/carla/ego_vehicle/odometry` nav_msgs/Odometry – ego pose & twist
"""
from __future__ import annotations

import math
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Float64, String
from nav_msgs.msg import Path, Odometry

LOOKAHEAD_MIN = 4.0      # m
LOOKAHEAD_TIME = 0.5     # s  (L = Lmin + t * v)

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # parameters
        self.declare_parameter('kp', 0.4)
        self.declare_parameter('ki', 0.05)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('i_clamp', 0.3)
        self.declare_parameter('cruise_speed', 8.0)        # m/s fallback
        self.declare_parameter('coast_trigger_s', 6.0)     # TTC seconds
        self.declare_parameter('hard_brake_s', 2.0)

        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.i_clamp = self.get_parameter('i_clamp').value
        self.v_cruise = self.get_parameter('cruise_speed').value
        self.coast_trigger = self.get_parameter('coast_trigger_s').value
        self.hard_brake = self.get_parameter('hard_brake_s').value

        # control publishers
        qos = 10
        self.pub_steer   = self.create_publisher(Float64, 'steering_command', qos)
        self.pub_throttle= self.create_publisher(Float64, 'throttle_command', qos)
        self.pub_brake   = self.create_publisher(Float64, 'brake_command', qos)
        self.pub_gear    = self.create_publisher(String,   'gear_command', qos)
        # latch forward gear once
        self.pub_gear.publish(String(data='forward'))

        # state
        self.path: Optional[Path] = None
        self.waypoints_np: Optional[np.ndarray] = None  # shape (N,2)
        self.speeds_np: Optional[np.ndarray] = None     # target v at each wp
        self.nearest_obstacle = float('inf')
        self.desired_speed_override: Optional[float] = None

        # subscribers
        self.create_subscription(Path, 'nav_path', self.on_path, qos)
        self.create_subscription(Float32, 'nearest_obstacle_distance', self.on_obs, qos)
        self.create_subscription(Float32, 'desired_speed', self.on_v_desired, qos)
        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.on_odom, qos)

        # PID terms
        self.prev_err = 0.0
        self.int_err = 0.0
        self.last_time = self.get_clock().now()

        # watchdog timer—check control loop alive
        self.timer_last_cmd = time.time()
        self.create_timer(0.2, self.watchdog)

        self.get_logger().info('Control node ready (Pure‑Pursuit + PID + coast).')

    # ------------------------------------------------------------------ callbacks
    def on_path(self, msg: Path):
        self.path = msg
        self.waypoints_np = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self.speeds_np    = np.array([p.pose.orientation.z              for p in msg.poses])

    def on_obs(self, msg: Float32):
        self.nearest_obstacle = msg.data

    def on_v_desired(self, msg: Float32):
        self.desired_speed_override = msg.data

    def on_odom(self, msg: Odometry):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.last_time = now

        # extract pose and twist
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v_ego = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

        if self.path is None or self.waypoints_np is None:
            return

        # ---------------- pure‑pursuit steering -----------------------
        look_ahead = max(LOOKAHEAD_MIN, LOOKAHEAD_TIME * v_ego)
        # find waypoint ahead of lookahead distance along the path
        dists = np.linalg.norm(self.waypoints_np - np.array([x, y]), axis=1)
        idx = int(np.where(dists > look_ahead)[0][0]) if np.any(dists > look_ahead) else -1
        target_pt = self.waypoints_np[idx]
        dx = target_pt[0] - x
        dy = target_pt[1] - y
        # transform to vehicle frame: assume yaw = atan2(vy, vx) is small; use sign of cross
        heading = math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)  # quick approx
        # simple rotation z
        tx =  math.cos(-heading) * dx - math.sin(-heading) * dy
        ty =  math.sin(-heading) * dx + math.cos(-heading) * dy
        curvature = 2 * ty / (look_ahead**2)
        steer_cmd = max(min(curvature * 1.0, 1.0), -1.0)  # scale to ‑1…1

        # ---------------- desired speed -----------------------------
        v_target = self.speeds_np[idx] if idx >= 0 else self.v_cruise
        if self.desired_speed_override is not None:
            v_target = min(v_target, self.desired_speed_override)

        # obstacle‑aware speed cap (TTC)
        if self.nearest_obstacle < 1e9 and v_ego > 0.1:
            ttc = self.nearest_obstacle / v_ego
            if ttc < self.hard_brake:
                v_target = 0.0
            elif ttc < self.coast_trigger:
                v_target = min(v_target, 0.3 + 0.2 * ttc)  # linear fall

        # ---------------- PID throttle / brake -----------------------
        err = v_target - v_ego
        self.int_err += err * dt
        # anti‑wind‑up clamp
        self.int_err = max(min(self.int_err, self.i_clamp), -self.i_clamp)
        deriv = (err - self.prev_err) / dt
        self.prev_err = err
        u = self.kp * err + self.ki * self.int_err + self.kd * deriv

        throttle = max(0.0, min(u, 1.0))
        brake    = max(0.0, min(-u, 1.0))
        # eco‑coast: don’t brake if target >0 and u<0 small
        if 0.0 < v_target and err < 0 and brake < 0.3:
            brake = 0.0

        # publish
        self.pub_steer.publish(Float64(data=float(steer_cmd)))
        self.pub_throttle.publish(Float64(data=float(throttle)))
        self.pub_brake.publish(Float64(data=float(brake)))
        self.timer_last_cmd = time.time()

    # ------------------------------------------------------------------ watchdog
    def watchdog(self):
        if time.time() - self.timer_last_cmd > 0.2:
            # no odom callback → stop vehicle
            self.pub_throttle.publish(Float64(data=0.0))
            self.pub_brake.publish(Float64(data=1.0))
            self.pub_steer.publish(Float64(data=0.0))

# -----------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
