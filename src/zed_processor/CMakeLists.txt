cmake_minimum_required(VERSION 2.8.7)

project(zed_processor)

SET(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# if CMAKE_BUILD_TYPE is not specified, take 'Release' as default
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_BUILD_TYPE)

#find_package(ZED 1.0 REQUIRED)

##For Jetson, OpenCV4Tegra is based on OpenCV2.4
exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64" ) # X1
    SET(OCV_VERSION "2.4")
    SET(CUDA_VERSION "7.0")
elseif(CMAKE_SYSTEM_NAME2 MATCHES "armv7l" ) # K1
    SET(OCV_VERSION "3.1")
    SET(CUDA_VERSION "6.5")
else() # Desktop
    SET(OCV_VERSION "3.1")
    SET(CUDA_VERSION "8.0")
endif()

include_directories("/usr/include/libusb-1.0")

## 3rd party libs
include_directories("../../3rdparty/benchmark/include")
include_directories("../../3rdparty/cub")
link_directories("../../3rdparty/benchmark/build/src")

find_package(catkin REQUIRED COMPONENTS
  rosbag
  roscpp
  rosconsole
  sensor_msgs
  tf2_ros
  cmake_modules
  trex_dmc01
)

find_package(OpenCV ${OCV_VERSION} REQUIRED COMPONENTS 
  core 
  highgui 
  imgproc 
  calib3d 
  xfeatures2d
  cudawarping
  cudafeatures2d
  PATHS /usr/local/opencv3/
  NO_DEFAULT_PATH
  )

find_package(Ceres REQUIRED)

find_package(CUDA ${CUDA_VERSION} REQUIRED)

set(CUDA_NVCC_FLAGS 
  -gencode arch=compute_30,code=sm_32
  -gencode arch=compute_50,code=sm_50
  -lineinfo 
  -std=c++11
  -O2
  -g)

find_package(Eigen3 REQUIRED)

find_package(Boost COMPONENTS program_options filesystem system)

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
  #	${ZED_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
)

#link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

###############################################################################

###############################################################################
# EXECUTABLE
add_definitions(-std=c++11 -g -Wno-deprecated-register)# -m64) #-Wall)

#file(GLOB LIBVISO_SRC_FILES src/libviso/*.cpp)

#include_directories(src/libviso)

#add_library(libviso ${LIBVISO_SRC_FILES})

#target_link_libraries(libviso)

cuda_add_library(
  freak_gpu
  src/freak_base.cc
  src/freak_gpu.cu
)

target_link_libraries(
  freak_gpu
  ${OpenCV_LIBS}
)

catkin_add_gtest(
  freak_gpu_test
  src/freak_gpu_test.cc)

target_link_libraries(
  freak_gpu_test
  freak_gpu
  fast_gpu
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  benchmark
  dw)

cuda_add_library(
  fast_gpu
  src/fast_gpu.cu)

catkin_add_gtest(
  fast_gpu_test
  src/fast_gpu_test.cc)

target_link_libraries(
  fast_gpu_test
  fast_gpu
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  benchmark
  dw)

cuda_add_library(
  stereo_matcher
  src/stereo_matcher.cu)

catkin_add_gtest(
  stereo_matcher_test
  src/stereo_matcher_test.cc)

target_link_libraries(
  stereo_matcher_test
  stereo_matcher
  math3d
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  benchmark
  dw)

cuda_add_library(
  descriptor_tools
  src/descriptor_tools.cu)

catkin_add_gtest(
  descriptor_tools_test
  src/descriptor_tools_test.cc)

target_link_libraries(
  descriptor_tools_test
  descriptor_tools
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  benchmark
  dw)

add_library(
  math3d
  src/math3d.cc)

catkin_add_gtest(
  math3d_test
  src/math3d_test.cc)

target_link_libraries(
  math3d_test
  math3d)

target_link_libraries(
  math3d
  ${OpenCV_LIBS})

add_executable(
  calibrate
  src/calibrate.cc
  src/calibration_data.cc)

target_link_libraries(
  calibrate
  math3d
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  dw)

add_executable(
  mission_editor
  src/mission_editor.cc
  src/calibration_data.cc
  src/direction_target.cc)

target_link_libraries(
  mission_editor
  math3d
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  dw)

add_library(
  reprojection_estimator
  src/reprojection_estimator.cc
  src/math3d.cc)

target_link_libraries(
  reprojection_estimator
  ${CERES_LIBRARIES}
  ${OpenCV_LIBS})

catkin_add_gtest(
  reprojection_estimator_test
  src/reprojection_estimator_test.cc)

target_link_libraries(
  reprojection_estimator_test
  reprojection_estimator)
  
add_library(
  clique
  src/clique.cc)

catkin_add_gtest(
  clique_test
  src/clique_test.cc)

target_link_libraries(
  clique_test
  clique)

add_executable(
  process
  src/process.cc
  src/bag_video_reader.cc
  src/calibration_data.cc
  src/camera.cc
  src/cone_tracker.cc
  src/cross_frame_processor.cc
  src/debug_renderer.cc
  src/direction_target.cc
  src/direction_tracker.cc
  src/frame_buffer_queue.cc
  src/frame_processor.cc
  src/kitti_video_reader.cc
  src/raw_video_reader.cc
  src/rigid_estimator.cc
  src/utils.cc
)

target_link_libraries(
  process
  clique
  descriptor_tools
  math3d
  reprojection_estimator
  freak_gpu
  fast_gpu
  stereo_matcher
  libdw.so
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY} nvToolsExt
  ${OpenCV_LIBS}
  ${libuvc_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(
  process_race
  src/process_race.cc
  src/bag_video_reader.cc
  src/calibration_data.cc
  src/camera.cc
  src/frame_buffer_queue.cc
  src/utils.cc
)

target_link_libraries(
  process_race
  math3d
  libdw.so
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY} nvToolsExt
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${libuvc_LIBRARIES}
  ${EIGEN_LIBRARIES}
)

add_dependencies(process_race trex_dmc01_generate_messages_cpp)

#add_executable(
#  process_viso
#  src/process_viso.cc
#  src/calibration_data.cc
#  src/bag_video_reader.cc
#)

#target_link_libraries(
#  process_viso
#  libviso
#  png
#  ${catkin_LIBRARIES}
#  ${OpenCV_LIBS}
#  gflags
#)

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

#add_executable(
#  zed_capture
#  src/zed_capture.cc)

#target_link_libraries(
#  zed_capture
#  ${OpenCV_LIBS}
#  ${catkin_LIBRARIES}
#  ${ZED_LIBRARIES}
#)

#add_executable(
#  zed_test
#  src/zed_test.cc)

#target_link_libraries(
#  zed_test
#  ${OpenCV_LIBS}
#  ${catkin_LIBRARIES}
#  ${ZED_LIBRARIES}
#)
###############################################################################

#Add all files in subdirectories of the project in
# a dummy_target so qtcreator have access to all files
FILE(GLOB_RECURSE extra_files ${CMAKE_SOURCE_DIR}/*)
add_custom_target(dummy_${PROJECT_NAME} SOURCES ${extra_files})
