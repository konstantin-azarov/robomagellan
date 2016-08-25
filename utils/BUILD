COPTS = []
LINKOPTS = []

cc_library(
    name="catch",
    hdrs=["catch.hpp"])

cc_library(
    name="backward",
    hdrs=["backward.hpp"],
    copts=COPTS)

cc_library(
    name="utils_lib",
    srcs=["utils.cc"],
    hdrs=["utils.hpp"])

cc_library(
    name="calibration_data_lib",
    srcs=["calibration_data.cc"],
    hdrs=["calibration_data.hpp"],
    deps=["@opencv3//:opencv2"],
    copts=COPTS)

cc_library(
    name="clique_lib",
    srcs=["clique.cc"],
    hdrs=["clique.hpp"],
    copts=COPTS)

cc_test(
    name="clique_test",
    srcs=["clique_test.cc"],
    deps=[":clique_lib", ":catch"],
    copts=COPTS)

cc_library(
    name="cross_frame_processor_lib",
    srcs=["cross_frame_processor.cc"],
    hdrs=["cross_frame_processor.hpp"],
    copts=COPTS,
    deps=[":clique_lib",
          ":reprojection_estimator_lib",
          ":frame_processor_lib",
          ":rigid_estimator_lib", 
          "@opencv3//:opencv2"])

cc_library(
    name="debug_renderer_lib",
    srcs=["debug_renderer.cc"],
    hdrs=["debug_renderer.hpp"],
    copts=COPTS,
    deps=[":frame_processor_lib", 
          ":cross_frame_processor_lib",
          "@opencv3//:opencv2"])

cc_library(
    name="frame_processor_lib",
    srcs=["frame_processor.cc"],
    hdrs=["frame_processor.hpp"],
    copts=COPTS,
    deps=[
        ":utils_lib", 
        ":calibration_data_lib", 
        "math3d_lib", 
        "@opencv3//:opencv2"])

cc_library(
    name="math3d_lib",
    srcs=["math3d.cc"],
    hdrs=["math3d.hpp"],
    copts=COPTS,
    deps=[":calibration_data_lib", "@opencv3//:opencv2"])

cc_test(
    name="math3d_test",
    srcs=["math3d_test.cc"],
    copts=COPTS,
    deps=[":math3d_lib", ":catch"])

cc_library(
    name="reprojection_estimator_lib",
    srcs=["reprojection_estimator.cc"],
    hdrs=["reprojection_estimator.hpp"],
    deps=[":math3d_lib", "@opencv3//:opencv2", "@local//:ceres"],
    copts=["-I/usr/include/eigen3"] + COPTS,
    linkopts=LINKOPTS)

cc_test(
    name="reprojection_estimator_test",
    srcs=["reprojection_estimator_test.cc"],
    deps=[":reprojection_estimator_lib",
          ":catch"],
    copts=COPTS)

cc_library(
    name="rigid_estimator_lib",
    srcs=["rigid_estimator.cc"],
    hdrs=["rigid_estimator.hpp"],
    deps=[":math3d_lib", "@opencv3//:opencv2"],
    linkopts=LINKOPTS,
    copts=COPTS)

cc_test(
    name="rigid_estimator_test",
    srcs=["rigid_estimator_test.cc"],
    deps=[":rigid_estimator_lib", 
          ":catch"],
    copts=COPTS)

cc_binary(
    name="process",
    srcs=["process.cc",
          "utils.cc",
          "average.hpp",
          "fps_meter.hpp",
          "timer.hpp",
          "raw_video_reader.cc",
          "raw_video_reader.h"],
    deps=[":backward",
          ":calibration_data_lib",
          ":clique_lib", 
          ":cross_frame_processor_lib",
          ":debug_renderer_lib",
          ":frame_processor_lib",
          ":rigid_estimator_lib", 
          ":reprojection_estimator_lib",
          "@opencv3//:opencv2"],
    copts=COPTS,
    linkopts=LINKOPTS + ["-lboost_program_options"])

cc_binary(
    name="capture",
    srcs=["capture.cc",
          "frame_buffer_queue.hpp",
          "frame_buffer_queue.cc",
          "camera.hpp",
          "camera.cc"],
    deps=[":utils_lib", "@opencv3//:opencv2", "@local//:uvc"],
    copts=COPTS,
    linkopts=LINKOPTS + ["-lboost_program_options"])

cc_binary(
    name="distance",
    srcs=["distance.cc"],
    deps=[
        ":calibration_data_lib",
        ":backward",
        "@opencv3//:opencv2"],
    copts=COPTS,
    linkopts=LINKOPTS + ["-lboost_program_options", "-ldw"])

cc_binary(
    name="calibrate",
    srcs=["calibrate.cc"],
    deps=[
        ":backward",
        ":math3d_lib",
        ":utils_lib", 
        ":calibration_data_lib",
        "@opencv3//:opencv2",
        "@cuda//:cuda"],
    copts=COPTS,
    linkopts=LINKOPTS + [
      "-lboost_system", 
      "-lboost_program_options", 
      "-lboost_filesystem",
      "-ldw"])
