import cv2
import numpy
import os

CHESSBOARD_WIDTH = 7
CHESSBOARD_HEIGHT = 5
CHESSBOARD_STEP = 24.5 # mm

N_IMAGES = 12

def trueCorners(cnt):
    res = []
    for i in xrange(0, CHESSBOARD_HEIGHT):
        for j in xrange(0, CHESSBOARD_WIDTH):
            res.append((j*CHESSBOARD_STEP, i*CHESSBOARD_STEP, 0))
    return [numpy.array(res, numpy.float32)]*cnt

def extractCornersFromImage(filename):
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

    res, corners = cv2.findChessboardCorners(
            img, 
            (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), 
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not res:
        return None, None

    term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01 )
    cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), term)

    cv2.drawChessboardCorners(img, (6, 9), corners, True)

    return img, corners

all_left_corners = []
all_right_corners = []
matching_left_corners = []
matching_right_corners = []

snapshots_dir = "snapshots/calibration"

i = 1
while os.path.isfile(os.path.join(snapshots_dir, "left_%d.png" % i)):
    img_left, corners_left = extractCornersFromImage(os.path.join(snapshots_dir, "left_%d.png" % i))
    img_right, corners_right = extractCornersFromImage(os.path.join(snapshots_dir, "right_%d.png" % i))

    if corners_left is not None and corners_right is not None:
        matching_left_corners.append(corners_left)
        matching_right_corners.append(corners_right)

    print "%d:" % i,

    if corners_left is not None:
        print "left",
        all_left_corners.append(corners_left)

    if corners_right is not None:
        print "right",
        all_right_corners.append(corners_right)

    print

    i += 1

calib_flags = (cv2.CALIB_RATIONAL_MODEL +
              cv2.CALIB_ZERO_TANGENT_DIST +
              cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5)

# res, Ml, dl, rvecs, tvecs = cv2.calibrateCamera(
#     trueCorners(len(all_left_corners)),
#     all_left_corners,
#     (640, 480),
#     None, numpy.zeros(8), None, None,
#     calib_flags,
#     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1E-5))
# print "Left RMS = ", res
#
# res, Mr, dr, rvecs, tvecs = cv2.calibrateCamera(
#     trueCorners(len(all_right_corners)),
#     all_right_corners,
#     (640, 480),
#     None, numpy.zeros(8), None, None,
#     calib_flags,
#     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1E-5))
# print "Right RMS = ", res

calib_flags = (cv2.CALIB_RATIONAL_MODEL +
              cv2.CALIB_ZERO_TANGENT_DIST +
            #   cv2.CALIB_FIX_ASPECT_RATIO +
            #   cv2.CALIB_SAME_FOCAL_LENGTH +
              cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5)

res, Ml, dl, Mr, dr, R, T, E, F = cv2.stereoCalibrate(
    trueCorners(len(matching_left_corners)),
    matching_left_corners,
    matching_right_corners,
    (640, 480),
    #Ml, dl, Mr, dr,
    #numpy.eye(3, 3), numpy.zeros(8), numpy.eye(3, 3), numpy.zeros(8),
    flags=calib_flags,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1E-5))

# Compute reprojection error
err = 0
cnt = 0
for i in xrange(len(matching_left_corners)):
    left_corners_und = cv2.undistortPoints(matching_left_corners[i], Ml, dl, None, Ml)
    right_corners_und = cv2.undistortPoints(matching_right_corners[i], Mr, dr, None, Mr)

    left_epilines = cv2.computeCorrespondEpilines(left_corners_und, 1, F)
    right_epilines = cv2.computeCorrespondEpilines(right_corners_und, 2, F)

    for i in xrange(len(left_epilines)):
        l = left_epilines[i].reshape(3)
        p = right_corners_und[i].reshape(2)
        err += abs(l[0]*p[0] + l[1]*p[1] + l[2])

    for i in xrange(len(right_epilines)):
        l = right_epilines[i].reshape(3)
        p = left_corners_und[i].reshape(2)
        err += abs(l[0]*p[0] + l[1]*p[1] + l[2])

    cnt += len(left_epilines)
print "Reprojection error = ", err/cnt


def savePyData(f):
    numpy.set_printoptions(precision=16)
    print >>f, "from numpy import array"
    print >>f, "Ml = ", repr(Ml)
    print >>f, "dl = ", repr(dl)
    print >>f, "Mr = ", repr(Mr)
    print >>f, "dr = ", repr(dr)
    print >>f, "R = ", repr(R)
    print >>f, "T = ", repr(T)

with open("data/calibration_constants.py", "w") as f:
    savePyData(f)
