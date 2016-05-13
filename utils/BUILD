cc_library(
    name="catch",
    hdrs=["catch.hpp"])

cc_library(
    name="reprojection_estimator_lib",
    srcs=["reprojection_estimator.cc"],
    hdrs=["reprojection_estimator.hpp",
          "util.hpp"],
    copts=["-I/usr/local/include/eigen3"],
    linkopts=["-lceres", "-lopencv_core"])

cc_test(
    name="reprojection_estimator_test",
    srcs=["reprojection_estimator_test.cc"],
    deps=[":reprojection_estimator_lib",
          ":catch"])

