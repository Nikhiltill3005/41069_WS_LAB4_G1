cmake_minimum_required(VERSION 3.5)
project(pablo)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# Add C++ executable
add_executable(ur3e_move src/ur3e_move.cpp)
ament_target_dependencies(ur3e_move rclcpp)

add_executable(path_planning src/path_planning.cpp)
ament_target_dependencies(path_planning rclcpp)

# Install C++ executable
install(TARGETS
  ur3e_move
  path_planning
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