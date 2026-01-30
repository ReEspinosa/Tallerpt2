# gz_mecanum_drive_controller

** Driver for mecanum wheels for Gazebo **

## References

* [Mecanum drive](https://github.com/gazebosim/gz-sim/tree/gz-sim8/src/systems/mecanum_drive)
* [Gazebo custom plugin](https://www.linkedin.com/pulse/gazebo-custom-plugin-ros2-step-by-step-guide-shantanu-parab)

## Requires

Gazeboo Harmonic.  Following [Gazebo Plugin](https://gazebosim.org/api/plugin/2/installation.html):

```
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update
sudo apt install libgz-plugin2-dev
sudo apt install gz-harmonic
```

To compile:
```
colcon build --symlink-install
```
