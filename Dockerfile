# Base image with NVIDIA CUDA
ARG BASE_IMAGE=osrf/ros:humble-desktop
FROM ${BASE_IMAGE} AS downloader

# Determine Webots version and download it
ARG WEBOTS_VERSION=R2023b
ARG WEBOTS_PACKAGE_PREFIX=
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --yes wget bzip2 && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://github.com/cyberbotics/webots/releases/download/$WEBOTS_VERSION/webots-$WEBOTS_VERSION-x86-64$WEBOTS_PACKAGE_PREFIX.tar.bz2 && \
    tar xjf webots-*.tar.bz2 && rm webots-*.tar.bz2

# Start a new stage from base image
FROM ${BASE_IMAGE}

# Install tools and dependencies
RUN apt-get update && \
    apt-get install --yes \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    wget \
    libboost-program-options-dev \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install cflib transform3D 

# Disable interactive dialogs
ENV DEBIAN_FRONTEND=noninteractive

# Install Webots runtime dependencies
RUN apt-get update && \
    apt-get install --yes wget xvfb locales && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://raw.githubusercontent.com/cyberbotics/webots/master/scripts/install/linux_runtime_dependencies.sh && \
    chmod +x linux_runtime_dependencies.sh && ./linux_runtime_dependencies.sh && \
    rm ./linux_runtime_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Install Webots
WORKDIR /usr/local
COPY --from=downloader /webots /usr/local/webots/
ENV QTWEBENGINE_DISABLE_SANDBOX=1
ENV WEBOTS_HOME /usr/local/webots
ENV PATH /usr/local/webots:${PATH}

# Enable OpenGL capabilities
ENV NVIDIA_DRIVER_CAPABILITIES graphics,compute,utility

# Set user name to fix a warning
ENV USER root

# Set the locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# Source ROS 2 environment
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Install Gazebo
RUN apt-get update && \
    apt-get install -y curl \
    lsb-release \
    gnupg && \
    curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update && \
    apt-get install -y gz-harmonic && \
    rm -rf /var/lib/apt/lists/*

# Install ROS 2 dependencies
RUN apt-get update && \
    apt-get install -y \
    ros-humble-motion-capture-tracking \
    ros-humble-tf-transformations \
    ros-humble-teleop-twist-keyboard \
    ros-humble-ros-gzharmonic

# Clone Crazyflie simulator repository
WORKDIR /root
RUN git clone https://github.com/bitcraze/crazyflie-simulation.git
RUN git clone https://github.com/bitcraze/crazyflie-lib-python.git && \
    cd crazyflie-lib-python && \
    pip3 install -e .

# Download the bresenham package
RUN pip install bresenham

# Use a shell script to find the line number and replace the line
RUN line_number=$(grep -n "sys.path.append('../../../../../../python/crazyflie-lib-python/examples/multiranger/wall_following')" /root/crazyflie-simulation/simulator_files/webots/controllers/crazyflie_controller_py_wallfollowing/crazyflie_controller_py_wallfollowing.py | cut -d: -f1) && \
    if [ -n "$line_number" ]; then \
        sed -i "${line_number}s|.*|sys.path.append('/root/crazyflie-lib-python/examples/multiranger/wall_following')|" /root/crazyflie-simulation/simulator_files/webots/controllers/crazyflie_controller_py_wallfollowing/crazyflie_controller_py_wallfollowing.py; \
    else \
        echo "Text not found in the file."; \
    fi

# Crazyradio 2.0 settings
# Add plugdev group and add root to it
RUN getent group plugdev || groupadd plugdev && usermod -a -G plugdev root

# Create udev rules for Crazyflie and Crazyradio
RUN mkdir -p /etc/udev/rules.d && \
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"' > /etc/udev/rules.d/99-bitcraze.rules && \
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="0101", MODE="0664", GROUP="plugdev"' >> /etc/udev/rules.d/99-bitcraze.rules && \
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0664", GROUP="plugdev"' >> /etc/udev/rules.d/99-bitcraze.rules

# Build Crazyflie controller
WORKDIR /root/crazyflie-simulation/simulator_files/webots/controllers/crazyflie_controller_c
RUN make

# # Set environment for Crazyflie simulation
# WORKDIR /root
# RUN export GZ_SIM_RESOURCE_PATH="/root/crazyflie-simulation/simulator_files/gazebo/"

# Create workspaces for both simulation and ROS
RUN mkdir -p /root/crazyflie_mapping_demo/simulation_ws && \
    mkdir -p /root/crazyflie_mapping_demo/ros2_ws/src

# Clone the simulation repository
WORKDIR /root/crazyflie_mapping_demo/simulation_ws
RUN git clone https://github.com/bitcraze/crazyflie-simulation.git

# Clone the ROS2 projects
WORKDIR /root/crazyflie_mapping_demo/ros2_ws/src
RUN git clone https://github.com/knmcguire/crazyflie_ros2_multiranger.git && \
    git clone https://github.com/knmcguire/ros_gz_crazyflie && \
    git clone --recursive https://github.com/IMRCLab/crazyswarm2.git

# Build the ROS2 workspace
WORKDIR /root/crazyflie_mapping_demo/ros2_ws
RUN source /opt/ros/humble/setup.bash && \
    colcon build --cmake-args -DBUILD_TESTING=ON
WORKDIR /root

# Source the ROS2 workspace
RUN echo "source /root/crazyflie_mapping_demo/ros2_ws/install/setup.bash" >> ~/.bashrc
ENV GZ_SIM_RESOURCE_PATH="/root/crazyflie_mapping_demo/simulation_ws/crazyflie-simulation/simulator_files/gazebo/"