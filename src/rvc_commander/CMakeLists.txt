cmake_minimum_required(VERSION 3.0.2)
project(rvc_commander)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  nav_msgs
  message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Measurement.msg
  LogOddsMap.msg
  WheelEncoder.msg
  Float64Array.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs   
    sensor_msgs   
    nav_msgs
)


###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rvc_commander
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy sensor_msgs std_msgs message_runtime
#  DEPENDS system_lib
)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${src}
)


#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
scripts/laser_scan_data_transformation.py
scripts/testing.py
scripts/log_odds_map_to_occupancy_grid.py
scripts/wheel_encoder_simulation.py
scripts/monte_carlo_localization_with_beam_model_modified.py
scripts/monte_carlo_localization_with_likelihood_field.py
scripts/monte_carlo_localization_with_likelihood_field_pre_computation.py
scripts/test_map_change.py
scripts/final_occupancy_grid_mapping_with_map_extension.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
