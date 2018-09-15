import numpy as np
from power3d import pow3d
import matplotlib.pyplot as plt
import scipy.optimize

IMAGE_WIDTH = 100
IMAGE_HEIGHT = IMAGE_WIDTH

BAILOUT = 64
ORDER = 5
MAX_ITERS = 6

PATIENCE = 120
ACCURACY = 0.02
MIN_DISTANCE = 2.9

CAMERA_POS = np.array([-0.1, 0.2, -4])
LOOK_XS = np.linspace(-0.1, 1.2, IMAGE_WIDTH)
LOOK_YS = np.linspace(-0.1, 1.2, IMAGE_HEIGHT)
LOOK_Z = 0

MU_NORM = np.log(np.log(BAILOUT) * ORDER) - np.log(np.log(BAILOUT))


def distance(v):
    """
    Estimated distance to the body of the Mandelbuld
    """
    z = v
    for k in range(MAX_ITERS):
        l = (z**2).sum()
        if l > BAILOUT:
            escape_time = k
            break
        z = pow3d(z, ORDER) + v
    else:
        return 0
    return np.log(np.log(l)) / MU_NORM - escape_time + MAX_ITERS - 2


min_distance = float("inf")
max_distance = float("-inf")
max_patience = 0
distances = -np.ones((IMAGE_WIDTH, IMAGE_HEIGHT))
for j, y in enumerate(LOOK_YS):
    print(j, min_distance, max_patience)
    for i, x in enumerate(LOOK_XS):
        source = CAMERA_POS
        jitter_x = np.random.rand() * (LOOK_XS[1] - LOOK_XS[0])
        jitter_y = np.random.rand() * (LOOK_YS[1] - LOOK_YS[0])
        direction = np.array([x + 0.01 * jitter_x, y + 0.01 * jitter_y, LOOK_Z]) - source
        direction /= np.sqrt((direction**2).sum())
        f = lambda t: distance(source + t * direction) - 0.5
        t = MIN_DISTANCE
        f_b = f(t)
        for k in range(PATIENCE):
            f_a = f_b
            t += ACCURACY
            f_b = f(t)
            if f_a * f_b < 0:
                root = scipy.optimize.brentq(f, t - ACCURACY, t)
                distances[j, i] = root
                if root < min_distance:
                    min_distance = root
                if root > max_distance:
                    max_distance = root
                if k > max_patience:
                    max_patience = k


print("Minimum distance", min_distance)
print("Maximum distance", max_distance)
print("Maximum patience", max_patience)
np.save("distances.npy", distances)
distances[distances < 0] = min_distance
plt.imshow(distances)
plt.show()
