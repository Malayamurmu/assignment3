""
Core algorithms: BCHW  <->  B x (C*H*W), implemented from scratch.

NO reshape()/view()/flatten()/ravel() anywhere in this file.
Only explicit loops + index arithmetic. NumPy is used purely as a
container (allocate an empty array, read/write single elements).

Row-major (C-order) address mapping, for a fixed batch index b:

    j = c*(H*W) + h*W + w                    (forward / flatten)

    c = j // (H*W)                           (inverse / reconstruct)
    r = j %  (H*W)
    h = r // W
    w = r %  W
""
import numpy as np


def flat_index(c, h, w, C, H, W):
    """Column index j inside the flattened row for element (c, h, w)."""
    return c * (H * W) + h * W + w


def unflat_index(j, C, H, W):
    """Inverse of flat_index: recover (c, h, w) from column index j."""
    hw = H * W
    c = j // hw
    r = j % hw
    h = r // W
    w = r % W
    return c, h, w


def bchw_to_bchw_flat(I):
    """B x C x H x W  ->  B x (C*H*W)   (explicit 4-level loop)."""
    B, C, H, W = I.shape
    F = np.empty((B, C * H * W), dtype=I.dtype)      # allocation only
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    j = flat_index(c, h, w, C, H, W)
                    F[b, j] = I[b, c, h, w]
    return F


def flat_to_bchw(F, C, H, W):
    """B x (C*H*W)  ->  B x C x H x W   (loop over flat address, invert it)."""
    B, N = F.shape
    assert N == C * H * W, f"row length {N} != C*H*W = {C*H*W}"
    R = np.empty((B, C, H, W), dtype=F.dtype)        # allocation only
    for b in range(B):
        for j in range(N):
            c, h, w = unflat_index(j, C, H, W)
            R[b, c, h, w] = F[b, j]
    return R


def errors(I, R):
    """Return (E_max, MAE) using explicit loops (no library reductions)."""
    B, C, H, W = I.shape
    e_max, total = 0.0, 0.0
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    d = abs(float(I[b, c, h, w]) - float(R[b, c, h, w]))
                    total += d
                    if d > e_max:
                        e_max = d
    return e_max, total / (B * C * H * W)
