# Don't bother making this installable for now.
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from power3d import pow3d, _pow

def test_identity():
    x, y, z = np.random.randn(3)
    rx, ry, rz = pow3d(x, y, z, 1)
    assert np.isclose([x, y, z], [rx, ry, rz]).all()


def test_zero():
    for i in range(1, 5):
        assert np.isclose(pow3d(0, 0, 0, i), [0, 0, 0]).all()


def test_known():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0.5, 1])

    assert np.isclose(pow3d(1, 0, 0, 2), [1, 0, 0]).all()
    assert np.isclose(pow3d(0, 1, 0, 2), [-1, 0, 0]).all()
    assert np.isclose(pow3d(0, 0.5, 1, 2), [0.75, 0, 1]).all()

    assert np.isclose(pow3d(1, 0, 0, 3), [1, 0, 0]).all()
    assert np.isclose(pow3d(0, 1, 0, 3), [0, -1, 0]).all()
    assert np.isclose(pow3d(0, 0.5, 1, 3), [0, 1.375, -0.25]).all()


def test_optimized():
    x, y, z = np.random.randn(3)
    assert np.isclose(pow3d(x, y, z, 2), _pow(x, y, z, 2)).all()
    assert np.isclose(pow3d(x, y, z, 4), _pow(x, y, z, 4)).all()
