cmake_minimum_required(VERSION 3.5)
project(pablo)

#------------- Find dependencies -------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

#------------- C++ executables -------------
add_executable(ur3e_move src/ur3e_move.cpp)
ament_target_dependencies(ur3e_move rclcpp std_msgs)
add_executable(path_planning src/path_planning.cpp)
ament_target_dependencies(path_planning rclcpp std_msgs)

install(TARGETS
  ur3e_move
  path_planning
  DESTINATION lib/${PROJECT_NAME}
)

#------------- Python modules and scripts -------------
ament_python_install_package(scripts)
install(PROGRAMS
  scripts/image_processor.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package() # Ament package