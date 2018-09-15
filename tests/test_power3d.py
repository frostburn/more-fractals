# Don't bother making this installable for now.
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from power3d import pow3d

def test_identity():
    v = np.random.randn(3)
    res = pow3d(v, 1)
    assert np.isclose(v, res).all()


def test_ranks():
    vs = np.random.randn(5, 3)
    rs = pow3d(vs, 3)
    for v, r in zip(vs, rs):
        res = pow3d(v, 3)
        assert np.isclose(r, res).all()


def test_zero():
    v = np.array([0, 0, 0])
    for i in range(1, 5):
        assert np.isclose(pow3d(v, i), v).all()


def test_known():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0.5, 1])

    assert np.isclose(pow3d(v1, 2), [1, 0, 0]).all()
    assert np.isclose(pow3d(v2, 2), [-1, 0, 0]).all()
    assert np.isclose(pow3d(v3, 2), [0.75, 0, 1]).all()

    assert np.isclose(pow3d(v1, 3), [1, 0, 0]).all()
    assert np.isclose(pow3d(v2, 3), [0, -1, 0]).all()
    assert np.isclose(pow3d(v3, 3), [0, 1.375, -0.25]).all()