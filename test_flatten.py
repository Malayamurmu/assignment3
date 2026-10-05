"""pytest / plain-python tests (no reshape inside the algorithm)."""
import numpy as np
from flatten_core import *
from datasets import synthetic_fmap


def test_index_roundtrip_all_addresses():
    C, H, W = 5, 4, 7
    seen = set()
    for c in range(C):
        for h in range(H):
            for w in range(W):
                j = flat_index(c, h, w, C, H, W)
                assert unflat_index(j, C, H, W) == (c, h, w)
                seen.add(j)
    assert seen == set(range(C * H * W))          # bijection, no gaps/collisions


def test_manual_example():
    assert flat_index(1, 0, 2, 2, 2, 3) == 8      # 1*6 + 0*3 + 2


def test_roundtrip_exact():
    for C in (1, 3, 8, 17):
        I = synthetic_fmap(2, C, 5, 6)
        R = flat_to_bchw(bchw_to_bchw_flat(I), C, 5, 6)
        assert errors(I, R) == (0.0, 0.0)


def test_matches_numpy_reference():               # reference only
    I = synthetic_fmap(3, 4, 5, 6)
    assert np.array_equal(bchw_to_bchw_flat(I), I.reshape(3, -1))


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
