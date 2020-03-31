cmake_minimum_required(VERSION 2.8.3)
project(laser_line_striping)

find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs nav_msgs cv_bridge)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(line_striping_node src/line_striping.cpp)
target_link_libraries(line_striping_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}  
)
