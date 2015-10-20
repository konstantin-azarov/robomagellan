import cv2
import numpy as np


M = np.array([[ 436.06674696,    0.        ,  311.98848644],
       [   0.        ,  437.07646328,  257.75458457],
       [   0.        ,    0.        ,    1.        ]])
d = np.array([[ -4.08805035e-01,   2.19300208e-01,   1.81827461e-03,
         -8.89952977e-05,  -7.29948083e-02]])

for i in xrange(1, 11):
    img = cv2.imread("snapshots/left_%d.png" % i)
    res = cv2.undistort(img, M, d)

    cv2.imshow("res", np.concatenate((img, res), 1))

    cv2.waitKey(0)

cv2.destroyAllWindows()
