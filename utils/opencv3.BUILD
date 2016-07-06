cc_library(
    name="opencv2",
    srcs=glob([
      "lib/libopencv_core.so",
      "lib/libopencv_imgproc.so",
      "lib/libopencv_imgcodecs.so",
      "lib/libopencv_calib3d.so",
      "lib/libopencv_features2d.so",
      "lib/libopencv_highgui.so",
      "lib/libopencv_flann.so"]),
    hdrs=glob(["include/**/*.hpp", "include/**/*.h"]),
    deps=["@cuda//:cuda"],
    includes=["include"],
    visibility=["//visibility:public"])
