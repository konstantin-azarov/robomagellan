import numpy as np

# Kl Kr B
p = [ 2.7, 2.7, 0.3  ]

cmds = np.array([[1.0, 1.0]] * 30 + [[1.0, 0.85]] * 60 + [[0.85, 1.0]] * 60)

speeds = np.concatenate(
        (np.reshape((cmds[:, 0] * p[0] + cmds[:, 1] * p[1])/2, (-1, 1)) + 
            np.random.randn(cmds.shape[0], 1) * 0.05,
        np.reshape((cmds[:, 0] * p[0] - cmds[:, 1] * p[1])/p[2], (-1, 1)) +
            np.random.randn(cmds.shape[0], 1) * 0.01), axis=1)

# v, w, cl, cr
xs = np.concatenate((speeds, cmds), axis=1)

s = np.mat([ 2, 2, 0.35 ])
P = np.mat(np.diag([0.5, 0.5, 0.2]))

for x in xs:
#    print x
#    print s
    M = np.mat(
            [[x[2]/2, x[3]/2, 0], 
             [x[2]/s[0, 2], -x[3]/s[0, 2], (s[0, 1]*x[3] - s[0, 0]*x[2])/(s[0, 2]**2)]])
    L = np.mat(np.diag([0.5, 0.5]))

    K = P * M.T * ( M * P * M.T + L).I

    f = np.mat([
        (s[0, 0] * x[2] + s[0, 1] * x[3])/2 - x[0],
        (s[0, 0] * x[2] - s[0, 1] * x[3]) / s[0, 2] - x[1]])

    s = (s.T - K * f.T).T
    P = np.mat(np.identity(3) - K * M) * P
    print s
