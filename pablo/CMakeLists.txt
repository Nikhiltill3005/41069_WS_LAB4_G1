cmake_minimum_required(VERSION 3.5)
project(pablo)

#------------- Find dependencies -------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

#------------- C++ executables -------------
add_executable(ur3e_control src/ur3e_control.cpp)
ament_target_dependencies(ur3e_control rclcpp std_msgs moveit_ros_planning_interface)
add_executable(path_planning src/path_planning.cpp)
ament_target_dependencies(path_planning rclcpp std_msgs OpenCV)

include_directories(include)

install(TARGETS
  ur3e_control
  path_planning
  DESTINATION lib/${PROJECT_NAME}
)

#------------- Python modules and scripts -------------
ament_python_install_package(scripts)
install(PROGRAMS
  scripts/image_processor.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package() # Ament package