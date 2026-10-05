"""Runs all experiments, prints + saves Table 1 (results.csv / results.md)."""
import csv, time
import numpy as np
from flatten_core import bchw_to_bchw_flat, flat_to_bchw, errors
from datasets import load_mnist_sample, load_cifar_sample, synthetic_fmap


def build_cases():
    cases = []
    t, n = load_mnist_sample(); cases.append((n, t))
    t, n2 = load_cifar_sample(); cases.append((n2, t))
    for k, C in enumerate([1, 3, 8, 16, 32, 64, 128, 256, 500], 1):
        cases.append((f"Synthetic fmap{k}", synthetic_fmap(2, C, 16, 16)))
    return cases


def main():
    rows = []
    for name, I in build_cases():
        B, C, H, W = I.shape
        t0 = time.perf_counter()
        F = bchw_to_bchw_flat(I)
        R = flat_to_bchw(F, C, H, W)
        dt = time.perf_counter() - t0
        e_max, mae = errors(I, R)
        # ---- REFERENCE CHECK ONLY (library reshape; not part of the algorithm)
        ref_ok = bool(np.array_equal(F, I.reshape(B, C * H * W)))
        # ----------------------------------------------------------------------
        rows.append([name, B, C, H, W, f"{e_max:.1e}", f"{mae:.1e}", ref_ok, f"{dt:.2f}"])
        print(f"{name:<26} B={B} C={C:<3} H={H} W={W}  Emax={e_max:.1e} MAE={mae:.1e} "
              f"matches_ref={ref_ok}  {dt:.2f}s")

    hdr = ["Dataset/Input", "B", "C", "H", "W", "Emax", "MAE", "matches_numpy_ref", "time_s"]
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    with open("results.md", "w") as f:
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows: f.write("| " + " | ".join(map(str, r)) + " |\n")


if __name__ == "__main__":
    main()
