import numpy as np

EPSILON = 1e-12


def _pow(x, y, z, n):
    """
    Raise the 3D vector v into "power" n.
    """
    r2 = x + y * 1j
    l2 = abs(r2)

    r1 = l2 + z * 1j

    r2 /= (l2 + (l2 < EPSILON))

    r1 **= n
    r2 **= n

    return r1.real * r2.real, r1.real * r2.imag, r1.imag


def pow3d(x, y, z, n):
    """
    Raise the 3D vector v into "power" n.
    """
    if n in (2, 4):
        x2 = x**2
        y2 = y**2
        z2 = z**2
        x2y2 = x2 + y2
        if n == 2:
            u = (x2y2 - z2) / (x2y2 + (x2y2 < EPSILON))
            y = 2*x*y*u
            x = (x2 - y2)*u
            z = 2*z*np.sqrt(x2y2)
            return x, y, z
        if n == 4:
            u = x2y2**2
            u = (z2**2 - 6*z2*x2y2 + u)/(u + (u < EPSILON))
            y = 4*x*y*(x2 - y2)*u
            x = (x2**2 - 6*x2*y2 + y2**2)*u
            z = 4*z*np.sqrt(x2y2)*(x2y2 - z2)
        return x, y, z

    return _pow(x, y, z, n)
