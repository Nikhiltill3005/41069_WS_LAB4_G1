#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class imageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.get_logger().info('Python Node Started')

def main(args=None):
    rclpy.init(args=args)
    node = imageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
