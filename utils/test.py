import cv2
import numpy as np
from data.calibration_constants import *
import sys
import random
import time

# dl.reshape(dl.size)[4] = 0
# dr.reshape(dr.size)[4] = 0

Rl, Rr, Pl, Pr, Q, poi1, poi2 = cv2.stereoRectify(
    Ml, dl,
    Mr, dr,
    (640, 480), R, T,
    None, None, None, None, None,
    cv2.CALIB_ZERO_DISPARITY, 0, (640, 480))

print Ml
print Pl
print Mr
print Pr
print Q
print poi1
print poi2

mlx, mly = cv2.initUndistortRectifyMap(Ml, dl, Rl, Pl, (640, 480), cv2.CV_16SC2)
mrx, mry = cv2.initUndistortRectifyMap(Mr, dr, Rr, Pr, (640, 480), cv2.CV_16SC2)



def speedTest():
    pairs = []
    i = 1
    while True:
        left = cv2.imread("utils/snapshots/test/left_%d.png" % i)
        right = cv2.imread("utils/snapshots/test/right_%d.png" % i)
        if (left is None or right is None):
            break
        pairs.append((left, right))
        i += 1

    t0 = time.time()
    for (left, right) in pairs:
        left1 = cv2.remap(left, mlx, mly, cv2.INTER_LINEAR)
        right1 = cv2.remap(right, mrx, mry, cv2.INTER_LINEAR)

        # sift = cv2.xfeatures2d.SIFT_create()
        # kpsl, descsl = sift.detectAndCompute(left1,None)
        #
        # sift = cv2.xfeatures2d.SIFT_create()
        # kpsr, descsr = sift.detectAndCompute(right1,None)
    t1 = time.time()
    print "dt = ", t1 - t0

# speedTest()
# sys.exit(0)


def descDist(d1, d2):
    return np.linalg.norm(d1 - d2)

def processImagePair(index):
    left = cv2.imread("utils/snapshots/test/left_%d.png" % index)
    right = cv2.imread("utils/snapshots/test/right_%d.png" % index)
    if left is None or right is None:
        return False

    #showStereoImage(left, right)

    left1 = cv2.remap(left, mlx, mly, cv2.INTER_LINEAR)
    right1 = cv2.remap(right, mrx, mry, cv2.INTER_LINEAR)

    sift = cv2.xfeatures2d.SURF_create()
    kpsl, descsl = sift.detectAndCompute(left1,None)

    sift = cv2.xfeatures2d.SURF_create()
    kpsr, descsr = sift.detectAndCompute(right1,None)

    def findKpIndex(kps, x, y, delta):
        candidates = []
        for (kp, i) in zip(kps, range(len(kps))):
            if (abs(kp.pt[0] - x) < delta and abs(kp.pt[1] - y) < delta):
                candidates.append(i)
        return candidates

    matches = []
    res = []

    for (kpl, descl, i) in zip(kpsl, descsl, xrange(len(kpsl))):
        candidates = []
        for (kpr, descr, j) in zip(kpsr, descsr, xrange(len(kpsr))):
            if abs(kpl.pt[1] - kpr.pt[1]) < 2 and abs(kpr.pt[0] - kpr.pt[0]) < 100:
                candidates.append((descDist(descl, descr), kpr, j))
        if len(candidates):
            candidates.sort()
            if len(candidates) > 1:
                ratio = candidates[0][0]/candidates[1][0]
            else:
                ratio = 1.0

            if ratio < 0.8:
                c = candidates[0][1]
                pt = np.array((kpl.pt[0], kpl.pt[1], kpl.pt[0] - c.pt[0]), ndmin=3)
                X = cv2.perspectiveTransform(
                    pt,
                    Q).reshape(3)
                #print kpl.pt, ":", ratio, c.pt[1] - kpl.pt[1], X

                matches.append((i, candidates[0][2]))
                res.append((X, descl, descsr[candidates[0][2]]))

    # def rndPt(p):
    #     return (int(round(p[0])), int(round(p[1])))
    #
    # img = np.concatenate((left1, right1), 1)
    # for (i, j) in matches:
    #     color = (random.randint(0, 255),
    #              random.randint(0, 255),
    #              random.randint(0, 255))
    #     cv2.drawKeypoints(
    #         img,
    #         [kpsl[i]],
    #         img,
    #         color=color,
    #         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    #     kpr = cv2.KeyPoint(
    #         kpsr[j].pt[0]+640,
    #         kpsr[j].pt[1],
    #         kpsr[j].size,
    #         kpsr[j].angle)
    #     cv2.drawKeypoints(
    #         img,
    #         [kpr],
    #         img,
    #         color=color,
    #         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     #cv2.line(img, rndPt(kpsl[i].pt), rndPt(kpr.pt), color)
    # sys.stdout.flush()
    #
    # cv2.imshow("main", img)
    # return cv2.waitKey(0) != 27

    return res


pts1 = processImagePair(1)
pts2 = processImagePair(2)

matches = []
for (X1, dl1, dr1) in pts1:
    best = -1
    second = -1
    best_X2 = None
    for (X2, dl2, dr2) in pts2:
        dist = (descDist(dl1, dl2) +
                descDist(dl1, dr2) +
                descDist(dr1, dl2) +
                descDist(dr1, dr2))
        if (best == -1 or dist < best):
            second = best
            best = dist
            best_X2 = X2
        elif (second == -1 or dist < second):
            second = dist
    if (best != -1):
        if (second == -1):
            ratio = 1
        else:
            ratio = best/second
        if (ratio < 0.8):
            print X1, "->", ratio, best_X2
            matches.append((X1, best_X2))

print len(matches)

W = np.zeros((len(matches), len(matches)))
for i in xrange(len(matches)):
    for j in xrange(len(matches)):
        a1, b1 = matches[i]
        a2, b2 = matches[j]
        W[i, j] = abs((np.linalg.norm(a1 - a2) - np.linalg.norm(b1 - b2))) < 30



print W
print np.sum(W, 0)
# i = 1
# while processImagePair(i):
#     i+=1
