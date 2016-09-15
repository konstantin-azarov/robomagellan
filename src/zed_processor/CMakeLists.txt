cmake_minimum_required(VERSION 2.8.7)

project(zed_processor)

# if CMAKE_BUILD_TYPE is not specified, take 'Release' as default
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_BUILD_TYPE)

find_package(ZED 1.1 REQUIRED)

##For Jetson, OpenCV4Tegra is based on OpenCV2.4
exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64" ) # X1
    SET(OCV_VERSION "2.4")
    SET(CUDA_VERSION "7.0")
elseif(CMAKE_SYSTEM_NAME2 MATCHES "armv7l" ) # K1
    SET(OCV_VERSION "2.4")
    SET(CUDA_VERSION "6.5")
else() # Desktop
    SET(OCV_VERSION "3.1")
    SET(CUDA_VERSION "7.5")
endif()

include_directories("/usr/include/libusb-1.0")

find_package(catkin REQUIRED COMPONENTS
  rosbag
  roscpp
  rosconsole
  sensor_msgs
  tf2_ros
  cmake_modules
)

find_package(OpenCV ${OCV_VERSION} COMPONENTS 
  core 
  highgui 
  imgproc 
  calib3d 
  xfeatures2d
  REQUIRED)

find_package(Ceres REQUIRED)

find_package(CUDA ${CUDA_VERSION} REQUIRED)

find_package(Eigen REQUIRED)

find_package(Boost COMPONENTS program_options)

find_package(libuvc REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rosconsole
    sensor_msgs
    tf2_ros
)

###############################################################################
# INCLUDES

# Specify locations of header files.
include_directories(
	${catkin_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${ZED_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

###############################################################################

###############################################################################
# EXECUTABLE
add_definitions(-std=c++11)# -m64) #-Wall)

add_executable(
  process
  src/process.cc
  src/bag_video_reader.cc
  src/calibration_data.cc
  src/clique.cc
  src/cross_frame_processor.cc
  src/debug_renderer.cc
  src/frame_processor.cc
  src/math3d.cc
  src/raw_video_reader.cc
  src/reprojection_estimator.cc
  src/rigid_estimator.cc
  src/utils.cc
)

target_link_libraries(
  process
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${CERES_LIBRARIES}
)

add_executable(
  capture
  src/capture.cc
  src/camera.cc
  src/utils.cc
  src/frame_buffer_queue.cc
)

target_link_libraries(
  capture
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${libuvc_LIBRARIES}
  libdw.so
)

add_executable(
  zed_capture
  src/zed_capture.cc)

target_link_libraries(
  zed_capture
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  ${ZED_LIBRARIES}
)

###############################################################################

#Add all files in subdirectories of the project in
# a dummy_target so qtcreator have access to all files
FILE(GLOB_RECURSE extra_files ${CMAKE_SOURCE_DIR}/*)
add_custom_target(dummy_${PROJECT_NAME} SOURCES ${extra_files})
