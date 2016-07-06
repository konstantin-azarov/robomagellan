cc_library(
    name="ceres",
    srcs=glob(["lib/cerec.a"]),
    hdrs=glob(["include/ceres/**/*.h"]),
    includes=["include"],
    visibility=["//visibility:public"])

cc_library(
    name="uvc",
    srcs=["lib/libuvc.so"],
    hdrs=glob(["include/libuvc/*.h"]),
    includes=["include"],
    visibility=["//visibility:public"])
