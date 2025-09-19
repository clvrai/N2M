#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

class PointCloudFrameConverter(Node):
    def __init__(self):
        super().__init__('point_cloud_frame_converter')
        
        # Create subscriber
        self.subscription = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.cloud_callback,
            1)
        
        # Create publisher
        self.publisher = self.create_publisher(
            PointCloud2,
            '/head_cam/point_cloud',
            1)
        
        self.get_logger().info('Point Cloud Frame Converter Node has started')
        
    def cloud_callback(self, msg):
        # Change the frame_id to "head_cam"
        msg.header.frame_id = 'head_cam'
        
        # Publish the modified point cloud
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = PointCloudFrameConverter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
