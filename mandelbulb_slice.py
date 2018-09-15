import numpy as np
import matplotlib.pyplot as plt
from power3d import pow3d

BAILOUT = 64
ORDER = 4
MAX_ITERS = 16

IMAGE_WIDTH = 100
IMAGE_HEIGHT = IMAGE_WIDTH

xs = np.linspace(-1.5, 1.5, IMAGE_WIDTH)
ys = np.linspace(-1.5, 1.5, IMAGE_HEIGHT)

escape_times = -np.ones((IMAGE_WIDTH, IMAGE_HEIGHT))
residues = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))
for j, y in enumerate(ys):
    for i, x in enumerate(xs):
        v = np.array([x, y, 0.1])
        z = v
        for k in range(MAX_ITERS):
            l = (z**2).sum()
            if l > BAILOUT:
                escape_times[i, j] = k
                break
            z = pow3d(z, ORDER) + v
        residues[i, j] = l

mu = np.log(residues)
mu[escape_times < 0] = 1
mu = np.log(mu)
norm = np.log(np.log(BAILOUT) * ORDER) - np.log(np.log(BAILOUT))
plt.imshow(escape_times - mu / norm)
plt.show()
