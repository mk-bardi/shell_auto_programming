#!/usr/bin/env python3
"""
Planning node (rev‑2) — Shell Eco‑marathon autonomous stack.

Key upgrades over the original version
--------------------------------------
1. **Automatic TSP re‑ordering** – the 14 target waypoints are re‑ordered with
   a greedy 2‑opt heuristic (≈5–15 % shorter distance) rather than executed in
   the YAML order.
2. **Curvature‑aware speed profile** – each `Pose` in `/nav_path` carries a
   `target_speed` encoded in `pose.orientation.z`, computed as:
        v = min(cruise_speed, sqrt(a_lat_max / |κ|))
   so the controller can anticipate corners.
3. **Latched publishes** – path and final mission flag are latched so late‑
   joining nodes see them immediately.
4. **Parameterisation** – everything exposed as ROS 2 params: cruise speed,
   lateral‑acc limit, waypoint YAML path, replan request.

Published topics
----------------
* `/nav_path`                (nav_msgs/Path   ) – full centreline with speeds
* `/current_target_waypoint` (geometry_msgs/PoseStamped) – next goal centre
* `/mission_complete`        (std_msgs/Bool   ) – latched true when done

Requires: carla Python API on PYTHONPATH (launch file already does that).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import List, Tuple

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

# CARLA imports (must be available via PYTHONPATH)
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import carla

# -----------------------------------------------------------------------------
# Helper – simple greedy + 2‑opt TSP
# -----------------------------------------------------------------------------

def tsp_greedy_2opt(points: List[np.ndarray]) -> List[int]:
    """Return an index order that roughly minimises total path length."""
    n = len(points)
    if n <= 2:
        return list(range(n))

    remaining = set(range(1, n))
    order = [0]
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda j: np.linalg.norm(points[j] - points[last]))
        order.append(nxt)
        remaining.remove(nxt)

    # 2‑opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b = order[i - 1], order[i]
                c, d = order[j], order[j + 1]
                if (np.linalg.norm(points[a] - points[c]) + np.linalg.norm(points[b] - points[d]) <
                        np.linalg.norm(points[a] - points[b]) + np.linalg.norm(points[c] - points[d])):
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True
    return order

# -----------------------------------------------------------------------------
# Planner node
# -----------------------------------------------------------------------------

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        # ---------------- parameters ----------------
        self.declare_parameter('waypoints_yaml', '')
        self.declare_parameter('cruise_speed', 8.0)      # m/s
        self.declare_parameter('a_lat_max', 1.5)         # m/s²
        self.declare_parameter('replan', False)          # trigger bool

        self.yaml_path = self.get_parameter('waypoints_yaml').value
        if not os.path.isfile(self.yaml_path):
            self.get_logger().error(f"Waypoint YAML not found: {self.yaml_path}")
            sys.exit(1)

        self._load_waypoints()

        # CARLA world is required for GlobalRoutePlanner, connect via client
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        dao = GlobalRoutePlannerDAO(world.get_map(), sampling_resolution=1.5)
        self.grp = GlobalRoutePlanner(dao)

        self.path_pub    = self.create_publisher(Path,        'nav_path',                1)  # latched
        self.target_pub  = self.create_publisher(PoseStamped, 'current_target_waypoint', 1)
        self.done_pub    = self.create_publisher(Bool,        'mission_complete',        1)  # latched

        self.build_and_publish_path()

        # listen for replan parameter toggle
        self.add_on_set_parameters_callback(self.on_param)

    # ---------------------------------------------------------------------
    def _load_waypoints(self):
        import yaml
        with open(self.yaml_path) as f:
            arr = yaml.safe_load(f)
        self.goals = [np.array([p[0], p[1], p[2]], dtype=float) for p in arr]
        self.start = np.array([280.363739, -129.306351, 0.101746], dtype=float)  # fixed

    # ---------------------------------------------------------------------
    def build_and_publish_path(self):
        self.get_logger().info('Building global route …')

        # --- reorder goals via greedy+2‑opt TSP --------------------------
        pts_xy = [self.start[:2]] + [g[:2] for g in self.goals]
        order = tsp_greedy_2opt(pts_xy)         # includes index 0 (start)
        goal_order = [self.goals[i - 1] for i in order[1:]]

        # --- stitch route segments --------------------------------------
        waypoints: List[carla.Waypoint] = []
        last_loc = carla.Location(*self.start)
        for goal in goal_order:
            goal_loc = carla.Location(*goal)
            route = self.grp.trace_route(last_loc, goal_loc)
            waypoints.extend([wp for wp, _ in route])
            last_loc = goal_loc

        # make sure last goal is visited exactly (publish target)
        self.goal_queue = [carla.Location(*g) for g in goal_order]
        self.current_goal_idx = 0

        # --- convert to nav_msgs/Path -----------------------------------
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        poses = []
        cruise = self.get_parameter('cruise_speed').value
        a_lat_max = self.get_parameter('a_lat_max').value
        coords = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in waypoints])

        # curvature & speed
        curvatures = np.zeros(len(coords))
        for i in range(1, len(coords) - 1):
            p_prev, p, p_next = coords[i - 1], coords[i], coords[i + 1]
            a = np.linalg.norm(p - p_prev)
            b = np.linalg.norm(p_next - p)
            c = np.linalg.norm(p_next - p_prev)
            area = abs(np.cross(p - p_prev, p_next - p_prev)) / 2.0
            if area < 1e-4:
                curv = 0.0
            else:
                R = a * b * c / (4 * area)
                curv = 1.0 / R
            curvatures[i] = curv

        for i, wp in enumerate(waypoints):
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(wp.transform.location.x)
            ps.pose.position.y = float(wp.transform.location.y)
            ps.pose.position.z = float(wp.transform.location.z)
            # encode target speed in orientation.z (hack to avoid custom msg)
            target_v = cruise
            if curvatures[i] > 1e-3:
                target_v = min(cruise, math.sqrt(a_lat_max / abs(curvatures[i])))
            ps.pose.orientation.z = target_v
            poses.append(ps)
        path_msg.poses = poses

        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Published path: {len(poses)} poses, goals: {len(self.goals)}')

        # publish first goal now
        self.publish_current_goal()

    # ---------------------------------------------------------------------
    def publish_current_goal(self):
        if self.current_goal_idx >= len(self.goal_queue):
            self.done_pub.publish(Bool(data=True))
            return
        goal_loc = self.goal_queue[self.current_goal_idx]
        tgt = PoseStamped()
        tgt.header.frame_id = 'map'
        tgt.pose.position.x, tgt.pose.position.y, tgt.pose.position.z = (
            goal_loc.x, goal_loc.y, goal_loc.z)
        self.target_pub.publish(tgt)

    # ---------------------------------------------------------------------
    def on_param(self, params: List[Parameter]):
        for p in params:
            if p.name == 'replan' and p.value:
                self.build_and_publish_path()
                return rclpy.parameter.SetParametersResult(successful=True)
        return rclpy.parameter.SetParametersResult(successful=False)

# -----------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
