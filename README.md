DO NOT FORGET to start your environment (should happen automatically in bashrc):

source /opt/ros/jazzy/setup.bash


After building with make (colcon build):

sudo ip link set can0 up type can bitrate 500000  # init CANbus interface
source install/setup.bash  # to get packages in environment
ros2 launch scout_base scout_base.launch.py  # to launch robot chassis module

---

To run main combo, use `run.sh`. This will launch the drivers for the chassis, lidar, camera, and our custom modules.

---

Requirements (ubuntu):

`sudo apt install libasio-dev`

```
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.realsenseai.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.realsenseai.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update

sudo apt-get install librealsense2-dev

sudo apt install ros-jazzy-gz-ros2-control ros-jazzy-diff-drive-controller
```
