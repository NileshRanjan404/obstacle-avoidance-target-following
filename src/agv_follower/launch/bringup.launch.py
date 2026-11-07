"""bringup.launch.py - starts the whole AGV.

  ros2 launch agv_follower bringup.launch.py                 # everything, robot follows the ball
  ros2 launch agv_follower bringup.launch.py follow:=false   # sensors + base only (teleop, calibration)
  ros2 launch agv_follower bringup.launch.py lidar:=false camera:=false follow:=false   # base only

The lidar driver is started here with the same parameter file that
`ros2 launch ydlidar_ros2_driver ydlidar_launch.py` uses, so do not run both at once.
The lidar and camera mounting transforms come from config/agv.yaml.
"""
import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def static_tf(name, parent, child, x, y, z, yaw_deg, condition=None):
    return Node(
        package='tf2_ros', executable='static_transform_publisher', name=name,
        arguments=['--x', str(x), '--y', str(y), '--z', str(z), '--yaw', str(math.radians(yaw_deg)),
                   '--frame-id', parent, '--child-frame-id', child],
        condition=condition,
    )


def setup(context):
    config = LaunchConfiguration('config').perform(context)
    lidar_params = LaunchConfiguration('lidar_params').perform(context)
    with open(config) as f:
        shared = yaml.safe_load(f).get('/**', {}).get('ros__parameters', {})

    def on(name):
        return IfCondition(LaunchConfiguration(name))

    # A lidar mounted upside down / clockwise is handled in follower via laser_angle_sign;
    # for rviz only the yaw matters.
    return [
        Node(package='ydlidar_ros2_driver', executable='ydlidar_ros2_driver_node',
             name='ydlidar_ros2_driver_node', output='screen', emulate_tty=True,
             parameters=[lidar_params], condition=on('lidar')),
        static_tf('tf_laser', 'base_link', 'laser_frame', shared.get('laser_x', 0.0),
                  shared.get('laser_y', 0.0), shared.get('laser_z', 0.0), shared.get('laser_yaw_deg', 0.0)),
        static_tf('tf_camera', 'base_link', 'camera_link', shared.get('camera_x', 0.0),
                  shared.get('camera_y', 0.0), shared.get('camera_z', 0.0), shared.get('camera_yaw_deg', 0.0)),
        Node(package='agv_follower', executable='base_driver', name='base_driver', output='screen',
             parameters=[config], condition=on('base')),
        Node(package='agv_follower', executable='ball_detector', name='ball_detector', output='screen',
             parameters=[config], condition=on('camera')),
        Node(package='agv_follower', executable='follower', name='follower', output='screen',
             parameters=[config], condition=on('follow')),
    ]


def generate_launch_description():
    share = get_package_share_directory('agv_follower')
    lidar_share = get_package_share_directory('ydlidar_ros2_driver')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(share, 'config', 'agv.yaml')),
        DeclareLaunchArgument('lidar_params', default_value=os.path.join(lidar_share, 'params', 'ydlidar.yaml')),
        DeclareLaunchArgument('lidar', default_value='true'),
        DeclareLaunchArgument('camera', default_value='true'),
        DeclareLaunchArgument('base', default_value='true'),
        DeclareLaunchArgument('follow', default_value='true'),
        OpaqueFunction(function=setup),
    ])
