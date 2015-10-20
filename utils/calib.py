import cv2
import numpy
import os

CHESSBOARD_WIDTH = 6
CHESSBOARD_HEIGHT = 9
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

    res, corners = cv2.findChessboardCorners(img, (6, 9))
    if not res:
        return None, None

    term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
    cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

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

res, Ml, dl, rvecs, tvecs = cv2.calibrateCamera(
    trueCorners(len(all_left_corners)),
    all_left_corners,
    (640, 480),
    None, None, None, None)

res, Mr, dr, rvecs, tvecs = cv2.calibrateCamera(
    trueCorners(len(all_right_corners)),
    all_right_corners,
    (640, 480),
    None, None, None, None)

res, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    trueCorners(len(matching_left_corners)),
    matching_left_corners,
    matching_right_corners,
    Ml, dl, Mr, dr,
    (640, 480),
    cv2.CALIB_FIX_INTRINSIC)

def savePyData(f):
    print >>f, "from numpy import array"
    print >>f, "Ml = ", repr(Ml)
    print >>f, "dl = ", repr(dl)
    print >>f, "Mr = ", repr(Mr)
    print >>f, "dr = ", repr(dr)
    print >>f, "R = ", repr(R)
    print >>f, "T = ", repr(T)

with open("data/calibration_constants.py", "w") as f:
    savePyData(f)
