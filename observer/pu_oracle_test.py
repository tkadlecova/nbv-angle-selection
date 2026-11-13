"""Exhaustive selections under the observer (model-observer table, one I0 per run).

Separates disagreement of the CRITERION from errors of the ranker by using
selections without any estimation error:
  A* (S)                  argmax S(A) from the full V matrix
  A* (S without P_unique) argmax S(A) with the overlap factor switched off
  fixed protocols, and SVR one-shot with --svr
All are scored by S_CHO (observer d', common random numbers) and compared
pairwise with the strongest of the three fixed protocols at this I0
(one-sided Wilcoxon). Run once per photon count I0.

P_unique is switched off by temporarily replacing
phantom_utils._defect_visibility_single, so every other part of V is
bit-identical.

    python pu_oracle_test.py --dist realistic --seed-offset 950000 --cho cho_realistic.h5 \
        --i0 5000 --real 1000 --svr svr_ens1.pt svr_ens2.pt svr_ens3.pt
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import itertools

import numpy as np

import phantom_utils as PU
import generate_vmatrix_dataset as G
from phantom_utils import PhantomGenerator, ACTION_SIZE, IMAGE_SIZE
from cho_validation import calibrate, load_cho, dog_channels, BODY_L
from cho_compare import d_cho_crn, FIXED, paired
from train_svr import score_sets_from_D

MAXD = G.MAX_DEF


def _single_no_unique(projection, defect_sino, metal_binary, dist_map,
                      all_defect_sinos, defect_id):
    """phantom_utils._defect_visibility_single without P_unique."""
    if defect_sino.sum() < 1e-6:
        return 0.0
    defect_region = defect_sino > 0.01
    if not defect_region.any():
        return 0.0
    bg_region = ~defect_region & ~metal_binary
    if bg_region.sum() < 5:
        return 0.0
    signal = projection[defect_region].mean()
    noise_std = max(float(projection[bg_region].std()), PU.NOISE_FLOOR)
    CNR = min(abs(1.0 - signal) / noise_std, PU.CNR_CLIP)
    mu_d = defect_sino[defect_region].mean()
    P_metal = 1.0 - float(metal_binary[defect_region].mean())
    P_dist = float(np.clip(dist_map[defect_region].mean() / 50.0, 0, 1))
    return float(CNR * mu_d * P_metal * P_dist) * 100.0


def d_without_unique(mu, dmask, metal):
    orig = PU._defect_visibility_single
    PU._defect_visibility_single = _single_no_unique
    try:
        D = G.visibility_matrix(mu, dmask, metal)
    finally:
        PU._defect_visibility_single = orig
    return D


def bf_set(M, sc):
    rem = [a for a in range(ACTION_SIZE) if a != sc]
    combos = np.array(list(itertools.combinations(rem, 3)), dtype=np.int64)
    S = score_sets_from_D(M, sc, [], combos)
    return [int(x) for x in combos[int(S.argmax())]]


def score_of(M, sc, ang):
    return float(score_sets_from_D(
        M, sc, [], np.array([sorted(ang)], dtype=np.int64))[0])


def main(a):
    svr = None
    if a.svr:
        from paired_stats import load, svr_oneshot
        from train_svr import SetVisibilityRanker
        svr = load(SetVisibilityRanker, a.svr)

    U = dog_channels()
    scale, med = calibrate([a.cho], a.body_L)
    print(f"Calibration: median body path {med:.1f} -> {a.body_L} "
          f"(factor {scale:.3e})")
    g_full, g_abl, D_ref, ndef, scout, sino = load_cho(a.cho, a.n, scale)

    probs = G.ORIENTATION_PRESETS[a.dist]
    rng_seed = np.random.default_rng(12345 + a.seed_offset)

    rows = {k: [] for k in list(FIXED)}
    rows["A* (S)"] = []
    rows["A* (S without P_unique)"] = []
    if svr is not None:
        rows["SVR one-shot"] = []

    # same acceptance loop and scout draw as generate_vmatrix_dataset.generate()
    made, attempt_seed, mismatch = 0, a.seed_offset, 0
    while made < len(g_full):
        attempt_seed += 1
        gen = PhantomGenerator(seed=attempt_seed, image_size=IMAGE_SIZE,
                               orientation_probs=probs)
        mu, dmask, _ = gen.generate_object()[:3]
        ids = sorted(int(d) for d in np.unique(dmask) if d > 0)
        if len(ids) == 0 or len(ids) > MAXD:
            continue
        metal = mu > 1.5
        D_full = G.visibility_matrix(mu, dmask, metal)
        _ = int(rng_seed.integers(0, ACTION_SIZE))

        n, sc = int(ndef[made]), int(scout[made])

        ref_D = np.asarray(D_ref[made][:n], float)
        Df = np.asarray(D_full, float)
        if Df.shape != ref_D.shape or np.abs(Df - ref_D).max() > 1e-3:
            mismatch += 1
        D_noq = d_without_unique(mu, dmask, metal)

        M = d_cho_crn(g_full[made], g_abl[made], n, a.i0, U, a.real,
                      10_000 * made)

        sel = {k: [(sc + o) % ACTION_SIZE for o in off]
               for k, off in FIXED.items()}
        sel["A* (S)"] = bf_set(np.asarray(D_full, float), sc)
        sel["A* (S without P_unique)"] = bf_set(np.asarray(D_noq, float), sc)
        if svr is not None:
            sel["SVR one-shot"] = svr_oneshot(svr, sino[made], sc)
        for k, ang in sel.items():
            rows[k].append(score_of(M, sc, ang))
        made += 1

    if mismatch:
        raise SystemExit(
            f"{mismatch} objects do not match {a.cho}; the regenerated "
            "phantoms differ and the results would not be valid.")
    print(f"Checksum: all {made} regenerated phantoms match {a.cho}.")

    rows = {k: np.array(v) for k, v in rows.items()}
    best_fixed = max(FIXED, key=lambda k: rows[k].mean())
    ref = rows[best_fixed]

    print(f"\n{a.cho}  (n = {made}, I0 = {a.i0:g})   "
          f"reference is {best_fixed}")
    print(f"  {'selection':<24} {'S_CHO':>8} {'Δ':>8} {'95% CI':>18} {'p':>9}")
    print("  " + "-" * 72)
    for k, v in rows.items():
        if k == best_fixed:
            print(f"  {k + ' (ref)':<24} {v.mean():>8.3f} {'—':>8} "
                  f"{'—':>18} {'—':>9}")
            continue
        d, lo, hi, p = paired(v, ref)
        print(f"  {k:<24} {v.mean():>8.3f} {d:>+8.3f} "
              f"[{lo:>+7.3f}, {hi:>+7.3f}] {p:>9.3g}")


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
