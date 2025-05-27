#!/usr/bin/env python3
"""
Perception node for Shell Eco-marathon APC 2025
Detects the nearest obstacle in front of the ego vehicle using either
the depth camera or the front LiDAR, if available.
Publishes:
  – /nearest_obstacle_distance (std_msgs/Float32, latched)
  – /obstacle_alert (std_msgs/Bool, latched)
"""

import math
from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct

class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__('perception_node')

        self.declare_parameter('obstacle_distance_threshold', 3.0)
        self.declare_parameter('roi_x_start_ratio', 0.38)
        self.declare_parameter('roi_x_end_ratio', 0.62)
        self.declare_parameter('roi_y_start_ratio', 0.45)
        self.declare_parameter('roi_y_end_ratio', 0.80)
        self.declare_parameter('use_lidar', True)
        self.declare_parameter('lidar_max_fwd_angle_deg', 30.0)
        self.declare_parameter('alert_latch_sec', 0.5)

        self.threshold = self.get_parameter('obstacle_distance_threshold').value
        self.roi_x_start_ratio = self.get_parameter('roi_x_start_ratio').value
        self.roi_x_end_ratio = self.get_parameter('roi_x_end_ratio').value
        self.roi_y_start_ratio = self.get_parameter('roi_y_start_ratio').value
        self.roi_y_end_ratio = self.get_parameter('roi_y_end_ratio').value
        self.use_lidar = self.get_parameter('use_lidar').value
        self.lidar_fwd_rad = math.radians(self.get_parameter('lidar_max_fwd_angle_deg').value)
        self.alert_latch_sec = self.get_parameter('alert_latch_sec').value

        latched_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.alert_pub = self.create_publisher(Bool, '/obstacle_alert', latched_qos)
        self.dist_pub = self.create_publisher(Float32, '/nearest_obstacle_distance', latched_qos)

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image,
            '/carla/ego_vehicle/depth_middle/image',
            self.depth_callback,
            sensor_qos
        )
        if self.use_lidar:
            self.lidar_sub = self.create_subscription(
                PointCloud2,
                '/carla/ego_vehicle/vlp16_1',
                self.lidar_callback,
                sensor_qos
            )
            self.get_logger().info("LiDAR mode enabled. Depth camera subscription active but processing will be skipped.")
        else:
            self.lidar_sub = None
            self.get_logger().info("Depth-camera mode enabled.")

        self._img_shape: Optional[tuple[int, int]] = None
        self._roi_px: Optional[tuple[int, int, int, int]] = None
        self._last_alert_time = self.get_clock().now()
        self.get_logger().info(
            f"Perception node ready (threshold = {self.threshold:.1f} m, "
            f"{'LiDAR' if self.use_lidar else 'depth-camera'} mode)"
        )

    def depth_callback(self, msg: Image) -> None:
        if self.use_lidar:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge failed: {exc}')
            return

        if self._roi_px is None:
            h, w = cv_image.shape
            if self._img_shape is None:
                 self._img_shape = (h, w)

            self._roi_px = (
                max(0, int(w * self.roi_x_start_ratio)),
                min(w, int(w * self.roi_x_end_ratio)),
                max(0, int(h * self.roi_y_start_ratio)),
                min(h, int(h * self.roi_y_end_ratio)),
            )
            self.get_logger().info(f"Computed ROI pixels = {self._roi_px} from image shape {h}x{w}")

        x0, x1, y0, y1 = self._roi_px
        if x1 <= x0 or y1 <= y0:
            if not hasattr(self, '_roi_error_logged'):
                self.get_logger().error("ROI ratios produce empty region — node disabled for depth processing.")
                self._roi_error_logged = True 
            return

        roi = cv_image[y0:y1, x0:x1]
        valid = roi[np.isfinite(roi) & (roi > 0.01)]
        if valid.size == 0:
            self._publish_obstacle(False, float('inf'))
            return

        dist = float(np.percentile(valid, 10.0))
        self._publish_obstacle(dist < self.threshold, dist)

    def lidar_callback(self, cloud: PointCloud2) -> None:
        step = cloud.point_step
        data = cloud.data
        nearest = float('inf')
        cos_fov = math.cos(self.lidar_fwd_rad)

        for i in range(0, len(data), step):
            try:
                x, y, z = struct.unpack_from('fff', data, i)
            except struct.error as e:
                self.get_logger().warn_once(f"Failed to unpack point: {e}. Point cloud format might be unexpected. Data offset: {i}, step: {step}, len(data): {len(data)}")
                continue

            if x <= 0.1:
                continue

            dist_sq = x*x + y*y + z*z
            if dist_sq == 0:
                continue
            inv_len = 1.0 / math.sqrt(dist_sq)
            if (x * inv_len) < cos_fov:
                continue

            dist = math.sqrt(dist_sq)
            if dist < nearest:
                nearest = dist
                if nearest < 0.5:
                    break

        if nearest == float('inf'):
            self._publish_obstacle(False, float('inf'))
        else:
            self._publish_obstacle(nearest < self.threshold, nearest)

    def _publish_obstacle(self, detected: bool, distance: float) -> None:
        now = self.get_clock().now()
        if detected:
            self._last_alert_time = now
        else:
            if (now - self._last_alert_time).nanoseconds * 1e-9 < self.alert_latch_sec:
                detected = True

        self.alert_pub.publish(Bool(data=detected))
        self.dist_pub.publish(Float32(data=distance))

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Perception node shut down.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()