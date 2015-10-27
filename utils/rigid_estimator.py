import numpy as np

import random
import math

# Compute R and t minimizing that
# sum( (R*xi + t - yi)^2 )
# X and Y are 3xn matrices
# returns (R, t)
def estimate_rotation(X, Y):
    n = X.shape[1]
    cX = np.sum(X, 1).reshape(3, -1)/n
    cY = np.sum(Y, 1).reshape(3, -1)/n

    print "cX = ", cX
    print "cY = ", cY

    X1 = X - np.tile(cX, n)
    Y1 = Y - np.tile(cY, n)

    # print "X1 = ", X1
    # print "Y1 = ", Y1

    S = np.dot(X1, np.transpose(Y1))

    # print "S = ", S

    U, S, Vt = np.linalg.svd(S)
    # print "U = ", U
    # print "V = ", Vt
    # print "USV", np.dot(np.dot(U, np.diag(S)), Vt)
    R = np.dot(U, Vt).transpose()
    t = cY - np.dot(R, cX)
    return (R, t)

def rotX(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]])

def rotY(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]])

def rotZ(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]])


if __name__ == "__main__":
    X = np.random.rand(3, 10)

    R = np.dot(
            np.dot(
                rotX(random.random()*math.pi),
                rotY(random.random()*math.pi)),
            rotZ(random.random()*math.pi))
    t = np.random.rand(3, 1)*random.random()*100
    Y = np.dot(R, X) + np.tile(t, X.shape[1])

    R1, t1 = estimate_rotation(X, Y)
    print "R1_err = ", np.abs(R1 - R)
    print "t1_err = ", np.abs(t1 - t)
