import numpy as np

EPSILON = 1e-12


def _pow_rank1(v, n):
    """
    Raise the 3D vector v into "power" n.
    """
    r2 = v[0] + v[1] * 1j
    l2 = abs(r2)

    r1 = l2 + v[2] * 1j

    r2 /= (l2 + (l2 < EPSILON))

    r1 **= n
    r2 **= n

    return np.array([r1.real * r2.real, r1.real * r2.imag, r1.imag])


def _pow_rank2(v, n):
    """
    Raise an array of 3D vectors into "power" n.
    """
    r2 = v[:,0] + v[:,1] * 1j
    l2 = abs(r2)

    r1 = l2 + v[:,2] * 1j

    r2 /= (l2 + (l2 < EPSILON))

    r1 **= n
    r2 **= n

    return np.vstack((r1.real * r2.real, r1.real * r2.imag, r1.imag)).T


def pow3d(v, n):
    """
    Raise the 3D vector v into "power" n.
    """
    v = np.array(v)
    rank = len(v.shape)
    if rank == 1:
        return _pow_rank1(v, n)
    elif rank == 2:
        return _pow_rank2(v, n)
    raise ValueError("Input shape not understood")
