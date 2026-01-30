"""Launch Gazebo server and client with command line arguments."""

import os
from ament_index_python.packages import get_package_share_directory, get_package_prefix

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

models_package = 'paquito_models'

# Resources

os.environ['GZ_SIM_RESOURCE_PATH'] = \
    os.path.join( get_package_share_directory(models_package),
        'worlds'
    ) + \
    os.pathsep + os.path.join(
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

    world_file_name = 'simple-world.sdf'

    world = os.path.join(
        get_package_share_directory(models_package),
        'worlds',
        world_file_name
    )

    return LaunchDescription([
        pkg_launch_arg,
        ExecuteProcess(
            cmd=['gz', 'sim', '-v', verbose, world_file_name],
            output='screen'),
    ])

