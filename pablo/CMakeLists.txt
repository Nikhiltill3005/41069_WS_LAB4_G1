cmake_minimum_required(VERSION 3.5)
project(pablo)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# Add C++ executable
add_executable(my_cpp_node src/my_cpp_node.cpp)
ament_target_dependencies(my_cpp_node rclcpp)

# Install C++ executable
install(TARGETS
  my_cpp_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(scripts)

# Install Python scripts
install(PROGRAMS
  scripts/image_processor.py
  DESTINATION lib/${PROJECT_NAME}
)

# Ament package
ament_package()