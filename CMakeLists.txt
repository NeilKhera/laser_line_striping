cmake_minimum_required(VERSION 3.1)
project(laser_line_striping)

find_package(catkin REQUIRED COMPONENTS rospy roscpp sensor_msgs nav_msgs cv_bridge)
find_package(OpenCV REQUIRED)
catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(line_striping_node src/line_striping.cpp)
target_link_libraries(line_striping_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
