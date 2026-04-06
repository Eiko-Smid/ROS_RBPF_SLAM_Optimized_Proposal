cmake_minimum_required(VERSION 3.0.2)
project(rvc_simulation_environment)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  roscpp
  rospy
  std_msgs
)

## Depend on system install of gazebo
find_package(gazebo REQUIRED)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rvc_simulation_environment
  CATKIN_DEPENDS gazebo_ros roscpp rospy std_msgs
#  DEPENDS system_lib
)


link_directories(${GAZEBO_LIBRARY_DIRS})

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${src}
  ${GAZEBO_INCLUDE_DIRS}
)

# Gazebo set flags
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
