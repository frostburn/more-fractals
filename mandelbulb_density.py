import numpy as np
import scipy.misc
import multiprocessing
from power3d import pow3d

t = 0

def project_cloud(seed):
    np.random.seed(seed)
    v = np.random.randn(2000000, 3) * 0.5
    v += np.array([-0.1, 0.2, 0.03])

    results = []
    z = v
    for i in range(6):
        z = pow3d(z, 2) + v
        results.append(z)

    # width = 640
    # height = 360
    width = 1920
    height = 1080

    imgs = []
    for z in results:
        z *= 0.15
        img = np.histogram2d(
            z[:, 1] + 0.1 * z[:, 0], z[:, 2] - 0.2 * z[:, 0],
            range=[(-height/width, height/width), (-1, 1)],
            bins=[height, width]
        )[0]
        imgs.append(img)
    return imgs


num_frames = 1
for n in range(num_frames):
    t = n / num_frames
    n_proc = multiprocessing.cpu_count() - 2
    p = multiprocessing.Pool(n_proc)  # New pool to sync global t
    # Save memory by reducing while mapping
    imgss = None
    remaining = 130
    while remaining > 0:
        print("Runs remaining:", remaining)
        imgss_ = p.map(project_cloud, range(remaining, remaining + n_proc))
        remaining -= n_proc
        if imgss is None:
            imgss = imgss_
        else:
            for i in range(len(imgss_)):
                for j in range(len(imgss_[i])):
                    imgss[i][j] += imgss_[i][j]
    p.close()

    r = 0
    g = 0
    b = 0
    for imgs in imgss:
        r += imgs[-6] + imgs[-4] * 0.5 + imgs[-5] * 0.5
        g += imgs[-3] + imgs[-2] * 0.5 + imgs[-4] * 0.5
        b += imgs[-1] + imgs[-2] * 0.5 + imgs[-5] * 0.5

    weight = (r.mean() + g.mean() + b.mean()) * 5

    r /= weight
    g /= weight
    b /= weight

    r **= 0.9
    g **= 0.8
    b **= 0.7

    img = np.array([r, g, b])

    filename = 'imgs/density_{0:05d}.png'.format(n)
    print('Saving', filename)
    scipy.misc.toimage(img, cmin=0, cmax=1).save(filename)
