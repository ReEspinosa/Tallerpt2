# Virtual Paquito
Gazebo virtual work with a paquito robot

# Requirements

## Firewall

If the ```uwf``` firewall is active it is necessary to allow ```multicast```. [Instalation troubleshooting. Enable multicast.](https://docs.ros.org/en/rolling/How-To-Guides/Installation-Troubleshooting.html#enable-multicast)

```
sudo ufw allow in proto udp to 224.0.0.0/4
sudo ufw allow in proto udp from 224.0.0.0/4
```


## Gazebo Bridge

Install Gazebo bridge:

```
sudo apt install ros-jazzy-ros-gz
```


## Mecanum Driver

Clone ROS package from [gz_mecanum_drive_controller](https://github.com/veroarriola/gz_mecanum_drive_controller).  Make sure to have the corresponding Gazebo version installed.


## Packages

Created with:

```
ros2 pkg create --build-type ament_cmake --license GPL-3.0-only virtual_paquito
ros2 pkg create --build-type ament_python --license GPL-3.0-only --node-name move_node move_paquito
```

Compile (terminal 1):

```
colcon build --symlink-install
```

Run (terminal 2):

```
source install/local_setup.bash
```


# Demos

## World with robot

```
ros2 launch paquito_models simpleworld.launch.py
```

## World and bridge
```
ros2 launch move_paquito simplebridge.launch.py
```

## World and game control

Requires Gazebo mecanum drive controller.

```
ros2 launch move_paquito classroombridge.launch.py
```

Press play to start simulation
