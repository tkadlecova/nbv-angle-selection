"""Is there anything to optimise under the observer? (split-half headroom)

The maximum over 6,545 noisy S_CHO estimates is biased upwards, so the naive
headroom above the fixed protocol is inflated. The optimum is therefore
searched on one half of the noise realisations (A) and evaluated on the
other (B); only the reproducible part of the headroom is what a ranker could
exploit. Both numbers are printed.

Phantoms are regenerated from their seeds (to get V and the optimum under S)
and checked against the V matrix stored in the observer file.

    python cho_headroom.py --dist realistic --seed-offset 950000 --cho cho_realistic.h5 \
        --i0 5000 --real 1000 --svr svr_ens1.pt svr_ens2.pt svr_ens3.pt
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import itertools

import numpy as np

import generate_vmatrix_dataset as G
from phantom_utils import PhantomGenerator, ACTION_SIZE, IMAGE_SIZE
from cho_validation import (auc_cho, d_prime, dog_channels, calibrate,
                            load_cho, BODY_L)
from cho_compare import FIXED, paired
from train_svr import score_sets_from_D

MAXD = G.MAX_DEF


def d_cho_halves(g_full, g_abl, n, i0, U, n_real, seed0):
    """Two independent d' matrices from disjoint noise streams."""
    A = np.empty((n, ACTION_SIZE))
    B = np.empty((n, ACTION_SIZE))
    for d in range(n):
        for a in range(ACTION_SIZE):
            ra = np.random.default_rng(seed0 + 7919 * d)
            rb = np.random.default_rng(seed0 + 7919 * d + 104729)
            A[d, a] = auc_cho(g_full[a], g_abl[d, a], i0, U, ra, n_real)
            B[d, a] = auc_cho(g_full[a], g_abl[d, a], i0, U, rb, n_real)
    return np.clip(d_prime(A), 0, None), np.clip(d_prime(B), 0, None)


def combos_of(sc):
    rem = [a for a in range(ACTION_SIZE) if a != sc]
    return np.array(list(itertools.combinations(rem, 3)), dtype=np.int64)


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

    keys = ["worst set", "median set"] + list(FIXED) + \
           ["A* (S)", "A*_CHO in-sample", "A*_CHO out-of-sample"]
    if svr is not None:
        keys.append("SVR one-shot")
    val = {k: [] for k in keys}
    pct = {k: [] for k in keys}
    rng_sets = []

    # same acceptance loop and scout draw as generate_vmatrix_dataset.generate()
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

        MA, MB = d_cho_halves(g_full[made], g_abl[made], n, a.i0, U,
                              a.real // 2, 10_000 * made)
        cb = combos_of(sc)
        SA = score_sets_from_D(MA, sc, [], cb)
        SB = score_sets_from_D(MB, sc, [], cb)
        rank = SB.argsort().argsort() / (len(SB) - 1) * 100
        idx = {tuple(sorted(c)): i for i, c in enumerate(cb.tolist())}

        def put(k, i):
            val[k].append(float(SB[i])); pct[k].append(float(rank[i]))

        put("worst set", int(SB.argmin()))
        put("median set", int(np.argsort(SB)[len(SB) // 2]))
        for k, off in FIXED.items():
            put(k, idx[tuple(sorted((sc + o) % ACTION_SIZE for o in off))])
        put("A* (S)", int(score_sets_from_D(Dm, sc, [], cb).argmax()))
        put("A*_CHO in-sample", int(SB.argmax()))
        put("A*_CHO out-of-sample", int(SA.argmax()))
        if svr is not None:
            put("SVR one-shot",
                idx[tuple(sorted(svr_oneshot(svr, sino[made], sc)))])
        rng_sets.append(float(SB.max() - SB.min()))
        made += 1

    if mism:
        raise SystemExit(f"{mism} objects do not match {a.cho}.")
    print(f"Checksum: all {made} phantoms match {a.cho}.")

    val = {k: np.array(v) for k, v in val.items()}
    best_fixed = max(FIXED, key=lambda k: val[k].mean())
    ref = val[best_fixed]

    print(f"\n{a.cho}  (n = {made}, I0 = {a.i0:g}, {a.real//2} realisations/half)")
    print("  {:<24} {:>8} {:>10} {:>9} {:>18}".format(
        "selection", "S_CHO", "percentile", "vs fixed", "95% CI"))
    print("  " + "-" * 74)
    for k in keys:
        d = val[k] - ref
        if k == best_fixed:
            print(f"  {k + ' (ref)':<24} {val[k].mean():>8.3f} "
                  f"{np.mean(pct[k]):>9.1f}% {'-':>9} {'-':>18}")
            continue
        _, lo, hi, _ = paired(val[k], ref)
        print(f"  {k:<24} {val[k].mean():>8.3f} {np.mean(pct[k]):>9.1f}% "
              f"{d.mean():>+9.3f} [{lo:>+7.3f}, {hi:>+7.3f}]")

    naive = val["A*_CHO in-sample"].mean() - ref.mean()
    real = val["A*_CHO out-of-sample"].mean() - ref.mean()
    print(f"\n  range worst-best set:                    {np.mean(rng_sets):.3f}")
    print(f"  headroom above fixed, naive (in-sample): {naive:+.3f}")
    print(f"  headroom above fixed, reproducible:      {real:+.3f}")
    if naive > 0:
        print(f"  bias of the maximum over noise:          {naive - real:+.3f}"
              f"  ({(1 - real/naive)*100:.0f} % of the naive headroom is noise)")


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
