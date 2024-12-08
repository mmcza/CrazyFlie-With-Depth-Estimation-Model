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

## Set position of the drone

```Shell
gz service -s /world/empty/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 300 -r "name: 'crazyflie', position: {x: -1.0, y: -1.0, z: 1.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}"
```

[Link](https://github.com/gazebosim/gz-msgs/blob/gz-msgs11/proto/gz/msgs/pose.proto) to Pose message declaration

## Collect training data

To run in `the world_cafe_1.sdf` (adjust the `num_of_files` to the desired number of pictures)

```Shell
ros2 run crazyflie_data_collector data_collector --ros-args -p min_x:=-4.75 -p max_x:=4.0 -p min_y:=-10.5 -p max_y:=11.5 -p min_z:=0.1 -p max_z:=2.50 -p num_of_files:=10 -p output_path:="/root/Shared/crazyflie_images/"
```

## To run the `data_viewer.py` script, use the following command in the terminal:

```bash
python data_viewer.py
```

## Model training

### Install the required packages

Install the required packages by running the following command:

```Shell
pip install -r requirements.txt
```

Check the CUDA version of the GPU and install the appropriate version of PyTorch from the [official website](https://pytorch.org/get-started/locally/).

### Start the training

To start the training, run the following command:

```Shell
python main.py
```