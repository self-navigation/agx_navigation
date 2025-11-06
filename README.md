DO NOT FORGET to start your environment (should happen automatically in bashrc):

source /opt/ros/jazzy/setup.bash


After building with make (colcon build):

sudo ip link set can0 up type can bitrate 500000  # init CANbus interface
source install/setup.bash  # to get packages in environment
ros2 launch scout_base scout_base.launch.py  # to launch robot chassis module

---

To run main combo, use `run.sh`. This will launch the drivers for the chassis, lidar, camera, and our custom modules.