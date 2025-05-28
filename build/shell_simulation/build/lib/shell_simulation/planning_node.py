#!/usr/bin/env python3
"""
Planning node (rev-3b) — Shell Eco‑marathon autonomous stack.
Integrated goal sequencing based on /goal_reached_ack.
Relies on waypoints_yaml parameter for 14 target waypoints.
"""
from __future__ import annotations

import math
import os
import sys
from typing import List

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import yaml # For loading waypoints

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import carla

def tsp_greedy_2opt(points: List[np.ndarray]) -> List[int]:
    n = len(points)
    if n <= 2:
        return list(range(n))
    order = [0]
    remaining = set(range(1, n))
    while remaining:
        last_pt_coords = points[order[-1]]
        nxt_idx = min(remaining, key=lambda j: np.linalg.norm(points[j] - last_pt_coords))
        order.append(nxt_idx)
        remaining.remove(nxt_idx)

    if n > 3: # 2-opt needs at least 4 points for a meaningful swap
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1): 
                for j in range(i + 1, n):
                    if j + 1 >= n: continue
                    
                    pt_a, pt_b = points[order[i-1]], points[order[i]]
                    pt_c, pt_d = points[order[j]], points[order[j+1]]
                    
                    if (np.linalg.norm(pt_c - pt_a) + np.linalg.norm(pt_d - pt_b) <
                            np.linalg.norm(pt_b - pt_a) + np.linalg.norm(pt_d - pt_c) - 1e-9):
                        order[i : j+1] = order[i : j+1][::-1]
                        improved = True
    return order

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        self.declare_parameter('waypoints_yaml', '')
        self.declare_parameter('cruise_speed', 8.0)
        self.declare_parameter('a_lat_max', 1.5)
        self.declare_parameter('replan', False)
        self.declare_parameter('num_goals_to_pass', 15) # Total goals: 1 start + 14 YAML

        self.yaml_path = self.get_parameter('waypoints_yaml').value
        if not self.yaml_path or not os.path.isfile(self.yaml_path):
            self.get_logger().fatal(f"Waypoint YAML not found or path not set: {self.yaml_path}")
            sys.exit(1)
        
        self.num_goals_to_pass = self.get_parameter('num_goals_to_pass').get_parameter_value().integer_value
        self._load_waypoints() # Sets self.goals_from_yaml and self.start_pose_fixed

        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0) # Your original timeout
        world = client.get_world()
        self.map = world.get_map()
        dao = GlobalRoutePlannerDAO(self.map, sampling_resolution=1.5)
        self.grp = GlobalRoutePlanner(dao)

        self.path_pub    = self.create_publisher(Path, 'nav_path', 1)
        self.target_pub  = self.create_publisher(PoseStamped, 'current_target_waypoint', 1)
        self.done_pub    = self.create_publisher(Bool, 'mission_complete', 1)
        self.done_pub.publish(Bool(data=False))

        self.create_subscription(Bool, '/goal_reached_ack', self.on_goal_reached_ack, 10)

        self.goals_passed_count = 0
        self.current_goal_idx_for_ack = 0 # Index for self.goal_queue_for_ack
        self.goal_queue_for_ack: List[carla.Location] = [] 

        self.build_and_publish_path()
        self.add_on_set_parameters_callback(self.on_param_change)
        self.get_logger().info(f"Planning node ready. Expecting {self.num_goals_to_pass} goals.")

    def _load_waypoints(self):
        with open(self.yaml_path) as f:
            loaded_yaml_goals_raw = yaml.safe_load(f)
        # These are the 14 target waypoints from the competition spec
        self.goals_from_yaml = [np.array([p[0], p[1], p[2]], dtype=float) for p in loaded_yaml_goals_raw]
        # Fixed start position ("green dot")
        self.start_pose_fixed = np.array([280.363739, -129.306351, 0.101746], dtype=float)

    def build_and_publish_path(self):
        self.get_logger().info('Building global route…')

        # TSP uses fixed start as anchor + 14 YAML goals
        pts_xy_for_tsp = [self.start_pose_fixed[:2]] + [g[:2] for g in self.goals_from_yaml]
        order_indices = tsp_greedy_2opt(pts_xy_for_tsp)
        
        # ordered_yaml_goals are the 14 YAML goals in TSP sequence
        ordered_yaml_goals = []
        if len(order_indices) > 1:
            ordered_yaml_goals = [self.goals_from_yaml[i - 1] for i in order_indices[1:]]

        # Stitch route segments for the detailed nav_path
        detailed_path_carla_wps: List[carla.Waypoint] = []
        last_loc_for_routing = carla.Location(*self.start_pose_fixed)
        for goal_np in ordered_yaml_goals:
            goal_loc_for_routing = carla.Location(*goal_np)
            route_segment = self.grp.trace_route(last_loc_for_routing, goal_loc_for_routing)
            if route_segment:
                detailed_path_carla_wps.extend([wp for wp, _ in route_segment])
                last_loc_for_routing = route_segment[-1][0].transform.location
            else:
                self.get_logger().warn(f"GRP failed for segment: {last_loc_for_routing} to {goal_loc_for_routing}")

        # Create the queue for goal acknowledgement (15 goals total)
        # Start pose is the first goal to be acknowledged.
        self.goal_queue_for_ack = [carla.Location(*self.start_pose_fixed)] + \
                                  [carla.Location(*g) for g in ordered_yaml_goals]
        self.current_goal_idx_for_ack = 0
        self.goals_passed_count = 0 # Reset for new plan

        # Convert to nav_msgs/Path and calculate speeds
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        poses = []
        cruise = self.get_parameter('cruise_speed').value
        a_lat_max = self.get_parameter('a_lat_max').value
        
        if not detailed_path_carla_wps:
            self.get_logger().error("No detailed_path_carla_wps generated, cannot publish path.")
            self.path_pub.publish(path_msg) # Publish empty path
            # Publish no current target if path is empty
            no_target_msg = PoseStamped()
            no_target_msg.header.frame_id = 'map'
            no_target_msg.header.stamp = self.get_clock().now().to_msg()
            self.target_pub.publish(no_target_msg)
            return

        coords = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in detailed_path_carla_wps])
        curvatures = np.zeros(len(coords))

        if len(coords) > 2:
            for i in range(1, len(coords) - 1):
                p_prev, p, p_next = coords[i-1], coords[i], coords[i+1]
                a = np.linalg.norm(p - p_prev)
                b = np.linalg.norm(p_next - p)
                c = np.linalg.norm(p_next - p_prev)
                area = abs(np.cross(p - p_prev, p_next - p_prev)) / 2.0
                if area < 1e-6 or a < 1e-6 or b < 1e-6 or c < 1e-6: # Adjusted epsilon
                    curv = 0.0
                else:
                    R = (a * b * c) / (4 * area)
                    curv = 1.0 / R if R > 1e-6 else 0.0
                curvatures[i] = curv

        for i, carla_wp in enumerate(detailed_path_carla_wps):
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp = path_msg.header.stamp
            ps.pose.position.x = float(carla_wp.transform.location.x)
            ps.pose.position.y = float(carla_wp.transform.location.y)
            ps.pose.position.z = float(carla_wp.transform.location.z)
            
            target_v = cruise
            current_curvature_val = abs(curvatures[i])
            if current_curvature_val > 1e-4: # Your original threshold was 1e-3
                val_under_sqrt = a_lat_max / current_curvature_val
                if val_under_sqrt >= 0:
                     target_v = min(cruise, math.sqrt(val_under_sqrt))
            
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = float(target_v) # Speed encoded here
            ps.pose.orientation.w = 1.0 
            poses.append(ps)
        
        path_msg.poses = poses
        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Published path: {len(poses)} poses.')

        self.publish_current_high_level_goal_for_ack()

    def publish_current_high_level_goal_for_ack(self):
        if self.current_goal_idx_for_ack < len(self.goal_queue_for_ack):
            goal_loc = self.goal_queue_for_ack[self.current_goal_idx_for_ack]
            tgt = PoseStamped()
            tgt.header.frame_id = 'map'
            tgt.header.stamp = self.get_clock().now().to_msg()
            tgt.pose.position.x, tgt.pose.position.y, tgt.pose.position.z = (
                float(goal_loc.x), float(goal_loc.y), float(goal_loc.z))
            tgt.pose.orientation.w = 1.0 # Identity orientation
            self.target_pub.publish(tgt)
            self.get_logger().info(f"Published high-level goal for ACK ({self.current_goal_idx_for_ack + 1}/{len(self.goal_queue_for_ack)})")
        # else: No more goals in queue to publish

    def on_goal_reached_ack(self, msg: Bool):
        if not msg.data: return

        self.goals_passed_count += 1
        self.get_logger().info(f"Goal {self.current_goal_idx_for_ack +1} ACKED. Total passed: {self.goals_passed_count}/{self.num_goals_to_pass}")

        if self.goals_passed_count >= self.num_goals_to_pass:
            self.get_logger().info(f"All {self.num_goals_to_pass} goals passed. Mission Complete.")
            self.done_pub.publish(Bool(data=True))
            return

        self.current_goal_idx_for_ack += 1
        if self.current_goal_idx_for_ack < len(self.goal_queue_for_ack):
            self.publish_current_high_level_goal_for_ack()
        else: # Should not happen if num_goals_to_pass is correct
            self.get_logger().warn("Goal queue exhausted but mission not yet complete by count.")
            if self.goals_passed_count < self.num_goals_to_pass:
                self.done_pub.publish(Bool(data=False)) # Mission failed if queue ends early


    def on_param_change(self, params: List[Parameter]): # Renamed from on_param
        for p in params:
            if p.name == 'replan' and p.value:
                self.get_logger().info("Replan triggered by parameter.")
                self.done_pub.publish(Bool(data=False)) # Reset mission status
                self.build_and_publish_path() # This will reset goal counts and publish first goal
                return rclpy.parameter.SetParametersResult(successful=True)
        return rclpy.parameter.SetParametersResult(successful=False)

def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down planning node.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()