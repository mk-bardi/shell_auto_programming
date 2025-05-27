#!/usr/bin/env python3

"""
Planning node for Shell Eco-marathon APC 2025
Creates a right-hand-traffic-legal path through a TSP-ordered waypoint list and
feeds it to the controller.

Publishes (all with TRANSIENT_LOCAL durability):
/nav_path            nav_msgs/Path
/current_target_waypoint geometry_msgs/PointStamped
/mission_complete     std_msgs/Bool

Subscribes:
  /carla/ego_vehicle/odometry nav_msgs/Odometry
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from std_msgs.msg import Bool

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


class PlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('planning_node')

        self.declare_parameter('waypoints_yaml', 'config/waypoints.yaml')
        self.declare_parameter('waypoint_threshold', 3.0)
        self.declare_parameter('sampling_resolution', 2.0)
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)

        self.waypoint_threshold = self.get_parameter('waypoint_threshold').value
        sampling_res = self.get_parameter('sampling_resolution').value
        carla_host = self.get_parameter('carla_host').value
        carla_port = self.get_parameter('carla_port').value
        way_yaml = Path(self.get_parameter('waypoints_yaml').value).expanduser()

        self.goal_points: List[Point] = self._load_yaml_waypoints(way_yaml)
        if not self.goal_points:
            self.get_logger().fatal("No waypoints loaded — shutting down.")
            raise SystemExit

        client = carla.Client(carla_host, carla_port)
        client.set_timeout(4.0)
        world = client.get_world()
        carla_map = world.get_map()
        dao = GlobalRoutePlannerDAO(carla_map, sampling_res)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        self.get_logger().info(f"GlobalRoutePlanner ready (res = {sampling_res:.1f} m)")

        self.full_path: List[carla.Waypoint] = []
        for start, goal in zip(self.goal_points[:-1], self.goal_points[1:]):
            seg = grp.trace_route(
                carla.Location(x=float(start.x), y=float(start.y), z=0.0),
                carla.Location(x=float(goal.x), y=float(goal.y), z=0.0)
            )
            seg_wps = [wp for wp, _ in seg if wp.lane_type == carla.LaneType.Driving and wp.lane_id > 0]
            self.full_path.extend(seg_wps)

        self.get_logger().info(f"Generated centre-line path with {len(self.full_path)} points.")

        latched_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.path_pub = self.create_publisher(PathMsg, '/nav_path', latched_qos)
        self.waypoint_pub = self.create_publisher(PointStamped, '/current_target_waypoint', latched_qos)
        self.complete_pub = self.create_publisher(Bool, '/mission_complete', latched_qos)

        sensor_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.odom_cb, sensor_qos)

        self.current_goal_idx = 0
        self.all_goals_done = False

        self._publish_nav_path()
        self._publish_current_target()
        self.create_timer(1.0, self._republish_target)

        self.get_logger().info("Planner node ready.")

    def _load_yaml_waypoints(self, yaml_file: Path) -> List[Point]:
        import yaml
        with yaml_file.open('r') as f:
            data = yaml.safe_load(f)
        try:
            wps = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in data['ordered_waypoints']]
            self.get_logger().info(f"Loaded {len(wps)} goal points from YAML.")
            return wps
        except Exception as exc:
            self.get_logger().error(f"Failed to parse {yaml_file}: {exc}")
            return []

    def _publish_nav_path(self) -> None:
        path = PathMsg()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        for wp in self.full_path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = wp.transform.location.x
            pose.pose.position.y = wp.transform.location.y
            pose.pose.position.z = wp.transform.location.z
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _publish_current_target(self) -> None:
        target = self.goal_points[self.current_goal_idx]
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.point = target
        self.waypoint_pub.publish(msg)
        self.get_logger().info(f"New target #{self.current_goal_idx}: {target.x:.1f}, {target.y:.1f}")

    def _republish_target(self) -> None:
        self.waypoint_pub.publish(PointStamped(
            header=self.get_clock().now().to_msg(),
            point=self.goal_points[self.current_goal_idx]
        ))

    def odom_cb(self, odom: Odometry) -> None:
        if self.all_goals_done:
            return

        pos = odom.pose.pose.position
        goal = self.goal_points[self.current_goal_idx]
        dist2 = (goal.x - pos.x) ** 2 + (goal.y - pos.y) ** 2
        if dist2 < self.waypoint_threshold ** 2:
            self.get_logger().info(f"Goal {self.current_goal_idx} reached.")
            if self.current_goal_idx < len(self.goal_points) - 1:
                self.current_goal_idx += 1
                self._publish_current_target()
            else:
                self.all_goals_done = True
                self.complete_pub.publish(Bool(data=True))
                self.get_logger().info("All goals reached — mission complete!")


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Planner node shut down.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
