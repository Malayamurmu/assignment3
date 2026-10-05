""Prints the hand-worked mapping for B=1, C=2, H=2, W=3.""
import numpy as np
from flatten_core import flat_index, bchw_to_bchw_flat

B, C, H, W = 1, 2, 2, 3
I = np.arange(B * C * H * W, dtype=np.float32) + 1          # values 1..12
# build the tensor WITHOUT reshape: fill element by element with a label value
I = np.empty((B, C, H, W), dtype=np.float32)
v = 1
for b in range(B):
    for c in range(C):
        for h in range(H):
            for w in range(W):
                I[b, c, h, w] = v; v += 1

print(f"B={B}, C={C}, H={H}, W={W}  ->  row length C*H*W = {C*H*W}\n")
print("j = c*(H*W) + h*W + w   with H*W =", H * W, "and W =", W, "\n")
print(f"{'(b,c,h,w)':<14}{'arithmetic':<26}{'j':<4}{'value'}")
for c in range(C):
    for h in range(H):
        for w in range(W):
            j = flat_index(c, h, w, C, H, W)
            print(f"{str((0,c,h,w)):<14}{f'{c}*{H*W} + {h}*{W} + {w}':<26}{j:<4}{I[0,c,h,w]:.0f}")
print("\nFlattened row:", bchw_to_bchw_flat(I)[0].astype(int).tolist())
