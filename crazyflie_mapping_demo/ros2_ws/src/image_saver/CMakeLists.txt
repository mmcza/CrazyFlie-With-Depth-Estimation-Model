cmake_minimum_required(VERSION 3.5)
project(image_saver)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

# Add executable
add_executable(image_saver src/image_saver.cpp)

# Link libraries
ament_target_dependencies(image_saver
    rclcpp
    sensor_msgs
    cv_bridge
    OpenCV
)

# Install targets
install(TARGETS image_saver
    DESTINATION lib/${PROJECT_NAME}
)

# Ament package
ament_package()