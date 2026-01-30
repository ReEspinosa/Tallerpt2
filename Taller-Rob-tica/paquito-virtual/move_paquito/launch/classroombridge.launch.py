"""Launch Gazebo server and client with command line arguments."""

import os
from ament_index_python.packages import get_package_share_directory, get_package_prefix

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ros_gz_bridge.actions import RosGzBridge


# Packages
package_name = 'move_paquito'
models_package = 'paquito_models'

# World
world_file_name = 'classroom-world.sdf'

# .-. Environment variables por Gazebo -.-

# Resources

os.environ['GZ_SIM_RESOURCE_PATH'] = \
    os.path.join( get_package_share_directory(models_package),
        'worlds'
    ) + \
    os.pathsep + \
    os.path.join(
        get_package_share_directory(models_package),
        'models'
    )

# Local drivers

os.environ['GZ_SIM_SYSTEM_PLUGIN_PATH'] = \
    os.path.join( get_package_prefix('gz_mecanum_drive_controller'),
        'lib/gz_mecanum_drive_controller/'
    )


def generate_launch_description():
    verbose = LaunchConfiguration('v', default='1')
    pkg_launch_arg = DeclareLaunchArgument(
        'v',
        default_value='1'
    )

    # World
    world = os.path.join(
        get_package_share_directory(models_package),
        'worlds',
        world_file_name
    )

    # Bridge
    #bridge_name = 'ros_gz_bridge'
    #config_file = 'simplebridge.yaml'
    bridge_config_file = os.path.join(
        get_package_share_directory(package_name),
        'resource',
        'simplebridge.yaml'
    )
    #ros_topic_name='/keyboard/keypress'
    #ros_msg_type='std_msgs/msg/Int32'
    #gz_msg_type='gz.msgs.Int32'

    # Launch description
    ld = LaunchDescription([
        pkg_launch_arg,
        ExecuteProcess(
            cmd=['gz', 'sim', '-v', verbose, world_file_name],
            output='screen',
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            #arguments=[ ros_topic_name + '@' + ros_msg_type + '@' + gz_msg_type],
            arguments=[
                '--ros-args',
                '-p',
                f'config_file:={bridge_config_file}',
            ],
            output='screen',
        ),
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen'
        ),
        Node(
            package='game_control',
            executable='game_control_node',
            name='game_control',
            output='screen',
        ),
        ExecuteProcess(
            cmd=['ros2', 'topic', 'echo', '/keyboard/keypress'],
            output='screen'
        ),
    ])

    print("\033[33mEn Gazebo abre el plugin Key Publisher\033[0m")

    return ld

