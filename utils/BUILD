COPTS = ["-g"]

cc_library(
    name="catch",
    hdrs=["catch.hpp"])

cc_library(
    name="backward",
    hdrs=["backward.hpp"],
    copts=COPTS)

cc_library(
    name="calibration_data_lib",
    srcs=["calibration_data.cc"],
    hdrs=["calibration_data.hpp"],
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
    name="frame_processor_lib",
    srcs=["frame_processor.cc"],
    hdrs=["frame_processor.hpp"],
    copts=COPTS,
    linkopts=["-lopencv_core"])

cc_library(
    name="math3d_lib",
    srcs=["math3d.cc"],
    hdrs=["math3d.hpp"],
    copts=COPTS,
    linkopts=["-lopencv_core"]);

cc_library(
    name="reprojection_estimator_lib",
    srcs=["reprojection_estimator.cc"],
    hdrs=["reprojection_estimator.hpp"],
    deps=[":math3d_lib"],
    copts=["-I/usr/local/include/eigen3"] + COPTS,
    linkopts=["-lceres", "-lopencv_core"])

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
    deps=[":math3d_lib"],
    linkopts=["-lopencv_core"],
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
          "raw_video_reader.cc"],
    deps=[":backward",
          ":calibration_data_lib",
          ":clique_lib", 
          ":frame_processor_lib",
          ":rigid_estimator_lib", 
          ":reprojection_estimator_lib"],
    copts=COPTS,
    linkopts=["-lboost_program_options",
              "-lopencv_features2d",
              "-lopencv_nonfree"])
