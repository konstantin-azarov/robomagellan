cmake_minimum_required(VERSION 2.8.3)
project(controller)

SET(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

find_package(
  catkin REQUIRED 
  COMPONENTS 
  roscpp 
  geometry_msgs
  message_generation 
  tf)

find_package(Eigen3 REQUIRED)

###########
## Build ##
###########
add_definitions(-std=c++11 -g -Wno-deprecated-register)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)


add_message_files(
  FILES
  Parameters.msg
)

generate_messages()

catkin_package()

## Declare a C++ executable
add_executable(controller src/controller.cc)
target_link_libraries(controller ${catkin_LIBRARIES})
add_dependencies(controller trex_dmc01_generate_messages_cpp)

add_executable(motion_planner src/motion_planner.cc)
target_link_libraries(motion_planner ${catkin_LIBRARIES})
add_dependencies(controller trex_dmc01_generate_messages_cpp)

