"""Factor-wise comparison of V with the observer d'.

(1) Spearman correlation of each factor of V (CNR*mu_d, P_metal, P_dist,
    P_unique) and of V itself with d', pooled over all defect x angle pairs.
(2) With --svr: angular spread of the selected sets (std of the gaps between
    the four views; 0 = evenly spaced).

The factors are recomputed step by step as in
phantom_utils._defect_visibility_single.

    python v_factor_analysis.py --dist realistic --seed-offset 950000 --cho cho_realistic.h5 \
        --svr svr_ens1.pt svr_ens2.pt svr_ens3.pt
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

import phantom_utils as PU
import generate_vmatrix_dataset as G
from phantom_utils import (PhantomGenerator, ACTION_SIZE, IMAGE_SIZE,
                           ANGLES_RANGE, generate_sinogram_transmission,
                           generate_sinogram_raw)
from cho_validation import calibrate, load_cho, dog_channels, BODY_L
from cho_compare import d_cho_crn, FIXED

MAXD = G.MAX_DEF
FACTORS = ["CNR*mu_d", "P_metal", "P_dist", "P_unique"]


def factors_at_angle(phantom, defect_mask, metal_map, angle_idx):
    angle = ANGLES_RANGE[angle_idx]
    proj = generate_sinogram_transmission(phantom, [angle])[0]
    metal_proj = generate_sinogram_transmission(
        metal_map.astype(np.float32), [angle])[0]
    metal_binary = (metal_proj < 1)
    dist_map = distance_transform_edt(~metal_binary)

    ids = sorted(int(d) for d in np.unique(defect_mask) if d > 0)
    sinos = {d: generate_sinogram_raw(
        (defect_mask == d).astype(np.float32), [angle])[0] for d in ids}

    out = {}
    for d in ids:
        ds = sinos[d]
        dr = ds > 0.01
        if ds.sum() < 1e-6 or not dr.any():
            out[d] = dict.fromkeys(FACTORS, np.nan)
            continue
        bg = ~dr & ~metal_binary
        if bg.sum() < 5:
            out[d] = dict.fromkeys(FACTORS, np.nan)
            continue
        noise = max(float(proj[bg].std()), PU.NOISE_FLOOR)
        CNR = min(abs(1.0 - proj[dr].mean()) / noise, PU.CNR_CLIP)
        Pu = 1.0
        for o in ids:
            if o != d and (dr & (sinos[o] > 0.01)).sum() > 0.5 * dr.sum():
                Pu *= 0.5
        out[d] = {"CNR*mu_d": CNR * ds[dr].mean(),
                  "P_metal": 1.0 - float(metal_binary[dr].mean()),
                  "P_dist": float(np.clip(dist_map[dr].mean() / 50.0, 0, 1)),
                  "P_unique": Pu}
    return [out[d] for d in ids]


def spread(angles_idx, scout):
    deg = np.sort([(a * 5.0) % 180 for a in [scout] + list(angles_idx)])
    gaps = np.diff(np.concatenate([deg, [deg[0] + 180]]))
    return float(gaps.std())


def main(a):
    svr = None
    if a.svr:
        from paired_stats import load, svr_oneshot
        from train_svr import SetVisibilityRanker
        svr = load(SetVisibilityRanker, a.svr)

    U = dog_channels()
    scale, med = calibrate([a.cho], a.body_L)
    print(f"Calibration: median body path {med:.1f} -> {a.body_L}")
    g_full, g_abl, D_ref, ndef, scout, sino = load_cho(a.cho, a.n, scale)

    probs = G.ORIENTATION_PRESETS[a.dist]
    rng_seed = np.random.default_rng(12345 + a.seed_offset)

    fac = {k: [] for k in FACTORS}
    dp, Vv = [], []
    spr = {k: [] for k in list(FIXED)}
    if svr is not None:
        spr["SVR one-shot"] = []

    made, attempt_seed, mism = 0, a.seed_offset, 0
    while made < len(g_full):
        attempt_seed += 1
        gen = PhantomGenerator(seed=attempt_seed, image_size=IMAGE_SIZE,
                               orientation_probs=probs)
        mu, dmask, _ = gen.generate_object()[:3]
        ids = sorted(int(d) for d in np.unique(dmask) if d > 0)
        if len(ids) == 0 or len(ids) > MAXD:
            continue
        metal = mu > 1.5
        Dm = G.visibility_matrix(mu, dmask, metal)
        Dm = np.asarray(Dm, float)
        _ = int(rng_seed.integers(0, ACTION_SIZE))

        n, sc = int(ndef[made]), int(scout[made])
        ref_D = np.asarray(D_ref[made][:n], float)
        if Dm.shape != ref_D.shape or np.abs(Dm - ref_D).max() > 1e-3:
            mism += 1
        M = d_cho_crn(g_full[made], g_abl[made], n, a.i0, U, a.real,
                      10_000 * made)

        for ang in range(ACTION_SIZE):
            per = factors_at_angle(mu, dmask, metal, ang)
            for d in range(min(n, len(per))):
                for k in FACTORS:
                    fac[k].append(per[d][k])
                dp.append(M[d, ang])
                Vv.append(Dm[d, ang])

        sel = {k: [(sc + o) % ACTION_SIZE for o in off]
               for k, off in FIXED.items()}
        if svr is not None:
            sel["SVR one-shot"] = svr_oneshot(svr, sino[made], sc)
        for k, ang in sel.items():
            spr[k].append(spread(ang, sc))
        made += 1

    if mism:
        raise SystemExit(f"{mism} objects do not match {a.cho}.")
    print(f"Checksum: all {made} phantoms match {a.cho}.")

    dp = np.array(dp)
    ok = ~np.isnan(dp)
    print(f"\n(1) Spearman of the factors of V against d'   "
          f"(n = {ok.sum()} defect x angle pairs, I0 = {a.i0:g})")
    print("  {:<14} {:>10} {:>10}   counterpart in the CHO".format(
        "factor", "rho vs d'", "rho vs V"))
    print("  " + "-" * 62)
    note = {"CNR*mu_d": "yes (SNR term)", "P_metal": "partial",
            "P_dist": "none", "P_unique": "none"}
    for k in FACTORS:
        v = np.array(fac[k])
        m = ok & ~np.isnan(v)
        r1 = spearmanr(v[m], dp[m]).statistic
        r2 = spearmanr(v[m], np.array(Vv)[m]).statistic
        print(f"  {k:<14} {r1:>+10.3f} {r2:>+10.3f}   {note[k]}")
    print(f"  {'V total':<14} {spearmanr(np.array(Vv)[ok], dp[ok]).statistic:>+10.3f} "
          f"{'—':>10}")

    if svr is not None:
        print(f"\n(2) Angular spread of the selected sets "
              f"(std of the gaps between the 4 views, 0 = evenly spaced)")
        for k, v in spr.items():
            print(f"  {k:<22} {np.mean(v):>6.1f}° ± {np.std(v):>4.1f}°")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dist", required=True,
                   choices=["structured", "realistic", "random"])
    p.add_argument("--seed-offset", type=int, required=True)
    p.add_argument("--cho", required=True)
    p.add_argument("--svr", nargs="*", default=[])
    p.add_argument("--i0", type=float, default=5000)
    p.add_argument("--real", type=int, default=1000)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--body-L", type=float, default=BODY_L)
    main(p.parse_args())
