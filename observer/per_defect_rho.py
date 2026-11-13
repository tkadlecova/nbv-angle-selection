"""Per-defect Spearman correlation between V and the observer d'.

For every defect, the correlation of V and d' across the 36 angles; reported
as the distribution over the whole observer set, next to the correlation
pooled over all defect x angle pairs. Spearman is invariant to monotone
rescaling, so the gap between the two measures how much the mapping V -> d'
varies between objects, not a difference in object difficulty.

d' matrices are cached to <data>_dprime_i0<I0>_r<real>.npz.

    python per_defect_rho.py --data cho_structured.h5 cho_realistic.h5 cho_random.h5 --i0 5000 --real 1000
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import os

import h5py
import numpy as np
from scipy.stats import spearmanr

from cho_validation import calibrate, load_cho, dog_channels, BODY_L
from cho_compare import d_cho_crn
from train_svr import ACTION_SIZE


def dprime_matrices(path, scale, U, i0, n_real, n):
    cache = path.replace(".h5", f"_dprime_i0{int(i0)}_r{n_real}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"  d' loaded from {cache}")
        return [z[k] for k in sorted(z.files, key=lambda s: int(s[1:]))]
    g_full, g_abl, _, ndef, _, _ = load_cho(path, n, scale)
    out = []
    for i in range(len(g_full)):
        k = int(ndef[i])
        out.append(np.zeros((0, ACTION_SIZE)) if k == 0 else
                   d_cho_crn(g_full[i], g_abl[i], k, i0, U, n_real, 10_000 * i))
    np.savez(cache, **{f"o{i}": m for i, m in enumerate(out)})
    print(f"  d' computed and saved to {cache}")
    return out


def main(a):
    U = dog_channels()
    scale, med = calibrate(a.data, a.body_L)
    print(f"Calibration: median body path {med:.1f} -> {a.body_L} "
          f"(factor {scale:.3e})\n")

    all_rho, pooled_V, pooled_D = [], [], []
    # same sample as the per-defect distribution (defects with constant V or d' excluded)
    sub_V, sub_D = [], []

    for path in a.data:
        print(f"{path}")
        with h5py.File(path, "r") as f:
            D_all, ndef = f["D"][:a.n], f["n_def"][:a.n]
        DPs = dprime_matrices(path, scale, U, a.i0, a.real, a.n)

        rhos, undef, flat = [], 0, 0
        for i in range(len(D_all)):
            k = int(ndef[i])
            if k == 0:
                continue
            V = D_all[i][:k].astype(np.float64)
            M = DPs[i]
            for d in range(k):
                pooled_V.append(V[d]); pooled_D.append(M[d])
                if V[d].std() == 0 or M[d].std() == 0:
                    undef += 1
                    if V[d].std() == 0:
                        flat += 1
                    continue
                rhos.append(spearmanr(V[d], M[d]).statistic)
                sub_V.append(V[d]); sub_D.append(M[d])
        r = np.array(rhos)
        all_rho.append(r)
        print(f"  defects with a defined correlation: {len(r)}"
              + (f"  (excluded {undef}, of which {flat} with constant V)"
                 if undef else ""))
        print(f"  median {np.median(r):+.3f}   IQR [{np.percentile(r,25):+.3f}, "
              f"{np.percentile(r,75):+.3f}]   share > 0.5: "
              f"{(r > 0.5).mean()*100:.0f} %\n")

    R = np.concatenate(all_rho)
    pV, pD = np.concatenate(pooled_V), np.concatenate(pooled_D)
    sV, sD = np.concatenate(sub_V), np.concatenate(sub_D)
    pooled_all = spearmanr(pV, pD).statistic
    pooled_sub = spearmanr(sV, sD).statistic

    print("=" * 62)
    print(f"PER DEFECT  (n = {len(R)} defects, each over 36 angles)")
    print(f"  median {np.median(R):+.3f}   IQR [{np.percentile(R,25):+.3f}, "
          f"{np.percentile(R,75):+.3f}]")
    print(f"  share of defects with rho > 0.5: {(R > 0.5).mean()*100:.0f} %"
          f"   |  rho < 0: {(R < 0).mean()*100:.0f} %")
    print("\nPOOLED  (all defect x angle pairs together)")
    print(f"  same sample as above ({len(sV)} pairs, {len(R)} defects): "
          f"rho = {pooled_sub:+.3f}")
    print(f"  including defects with constant V ({len(pV)} pairs): "
          f"rho = {pooled_all:+.3f}")
    print(f"\nGap per-defect median - pooled (same sample): "
          f"{np.median(R) - pooled_sub:+.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--i0", type=float, default=5000)
    p.add_argument("--real", type=int, default=1000)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--body-L", type=float, default=BODY_L)
    main(p.parse_args())
