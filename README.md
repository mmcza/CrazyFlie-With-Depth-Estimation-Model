# CrazyFlie With Depth Image Model
 
## Downloading the repository

Clone the repository into `~/crazyflie_sim_shared` directory:

```Shell
git clone https://github.com/mmcza/CrazyFlie-With-Depth-Image-Model/ ~/crazyflie_sim_shared
```

## Build the Docker image

```Shell
docker build . -t crazyflie_simulator
```

## Start the container

To start the container simply run:
```Shell
bash start_container.sh
```

To enter the container from another terminal you can use:
```Shell
docker exec -ti crazyflie-sim bash
```

## Fly around with the drone

```Shell
cd Shared/crazyflie_mapping_demo/ros2_ws/
```

```Shell
colcon build --symlink-install && source install/setup.bash
```

```Shell
ros2 launch crazyflie_ros2_multiranger_bringup simple_mapper_simulation.launch.py
```

In second terminal
```Shell
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```