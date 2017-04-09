import numpy as np
import matplotlib.pyplot as plt

f = open("/tmp/trace.txt", "r")
gt_f = open("/home/konstantin/kitti/dataset/poses/00.txt")

gt_ts = []
odo_ts = []

for l in f:
    gt_t = np.fromstring(gt_f.readline(), sep= ' ', dtype=float)
    v_t = np.fromstring(l, sep=' ', dtype=float)

    gt_ts.append(gt_t)
    odo_ts.append(v_t)

odo_ts = np.array(odo_ts)
gt_ts = np.array(gt_ts)
odo_ts[:, [3, 7, 11]] /= 1000

def plot_traj(ts, label, c, mul=1):
    traj = ts[:, [3, 11]]
    plt.plot(traj[:, 0]*mul, traj[:, 1]*mul, label=label, c=c)

def rebase(ts, tr0):
    m0 = tr0.reshape((3, 4))
    r0 = m0[:3, :3].T
    t0 = m0[:3, 3]

    res = []

    for t in ts:
        m = t.reshape((3, 4))
        r = np.concatenate((r0.dot(m[:3, :3]), r0.dot((m[:3, 3] - t0).T).reshape(3, 1)), axis=1)
        res.append(r.reshape(12))

    res = np.array(res)
    return np.array(res)

d = np.r_[0, np.linalg.norm(gt_ts[1:, [3, 7, 11]] - gt_ts[:-1, [3, 7, 11]], axis=1)]
d = np.cumsum(d)

n = d.size
i = 0
j = 0

l = 100

while i < n and d[i] < l:
    i += 1


errs = []
while i < n:
    while (d[i] - d[j+1] >= l):
        j += 1

    o = rebase(odo_ts[i:i+1, :], odo_ts[j])
    g = rebase(gt_ts[i:i+1, :], gt_ts[j])
    e = np.linalg.norm(o[:, [3, 7, 11]] - g[:, [3, 7, 11]])
    errs.append(e / (d[i] - d[j]))

    i += 1

#plt.plot(errs)
#plt.show()

    


s = 0
n = 1000

plot_traj(rebase(odo_ts[s:s+n, :], odo_ts[s]), label="odometry", c="blue")
plot_traj(rebase(gt_ts[s:s+n, :], gt_ts[s]), label="ground truth", c="green")
plt.xlabel('x')
plt.ylabel('z')
plt.legend();

plt.show()
