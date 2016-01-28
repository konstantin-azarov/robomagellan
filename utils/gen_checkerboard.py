import cv2
import numpy as np

w = 640*5
h = 480*5
sz = 80*5

img = np.zeros((h, w))

for i in xrange(h/sz):
    for j in xrange(w/sz):
        if (i+j)%2:
            img[i*sz:(i+1)*sz, j*sz:(j+1)*sz] = 255


cv2.imwrite("test.png", img)
