cmake_minimum_required(VERSION 3.5)
project(pablo)

#------------- Find dependencies -------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)


# Find OpenCV without ximgproc first
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Try to find opencv_contrib separately
find_package(OpenCV QUIET COMPONENTS ximgproc)
if(OpenCV_ximgproc_FOUND)
    message(STATUS "OpenCV ximgproc found")
    set(OPENCV_COMPONENTS ${OpenCV_LIBS})
else()
    message(WARNING "OpenCV ximgproc not found - will use fallback implementation")
    set(OPENCV_COMPONENTS ${OpenCV_LIBS})
endif()

find_package(moveit_ros_planning_interface REQUIRED)
# Find GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
# Find X11 for window capture
find_package(X11 REQUIRED)

#------------- C++ executables -------------
add_executable(ur3e_control src/ur3e_control.cpp)
ament_target_dependencies(ur3e_control rclcpp std_msgs moveit_ros_planning_interface)
add_executable(path_planning src/path_planning.cpp)
ament_target_dependencies(path_planning rclcpp std_msgs OpenCV)


# New streaming node
add_executable(rviz_streamer src/rviz_streamer.cpp)
ament_target_dependencies(rviz_streamer rclcpp std_msgs)
target_link_libraries(rviz_streamer 
    ${GSTREAMER_LIBRARIES}
    ${X11_LIBRARIES}
)
target_include_directories(rviz_streamer PRIVATE 
    ${GSTREAMER_INCLUDE_DIRS}
    ${X11_INCLUDE_DIRS}
)


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