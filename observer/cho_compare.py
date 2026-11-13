"""Model-observer comparison with common random numbers (CRN).

d'(d, a) of the CHO for every defect and angle, with the same noise stream
for every angle of a defect, so differences between angles are not noise.
Reports the flatness of S and of S_CHO near the optimum and, with --svr,
raw S_CHO of SVR one-shot vs the best fixed protocol (paired), split by
whether defect footprints overlap.

Also the library for the other observer scripts (d_cho_crn, FIXED, paired).

    python cho_compare.py --data cho_realistic.h5 --i0 5000 --real 1000 \
        --svr svr_ens1.pt svr_ens2.pt svr_ens3.pt
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import itertools

import numpy as np
from scipy.stats import wilcoxon

from cho_validation import (auc_cho, d_prime, dog_channels, calibrate,
                            load_cho, BODY_L)
from train_svr import ACTION_SIZE, PROJ_LEN, score_sets_from_D

FIXED = {"+60/120/150": (12, 24, 30),
         "+45/90/135": (9, 18, 27),
         "+30/90/150": (6, 18, 30)}


def d_cho_crn(g_full, g_abl, n, i0, U, n_real, seed0=0):
    """d' matrix (n_defects x 36), clipped at 0; one noise seed per defect."""
    M = np.empty((n, ACTION_SIZE))
    for d in range(n):
        for a in range(ACTION_SIZE):
            rng = np.random.default_rng(seed0 + 7919 * d)
            M[d, a] = auc_cho(g_full[a], g_abl[d, a], i0, U, rng, n_real)
    return np.clip(d_prime(M), 0, None)


def all_sets(sc):
    rem = [a for a in range(ACTION_SIZE) if a != sc]
    return np.array(list(itertools.combinations(rem, 3)), dtype=np.int64)


def score_of(M, sc, angles):
    return float(score_sets_from_D(
        M, sc, [], np.array([sorted(angles)], dtype=np.int64))[0])


def flatness(M, sc, rng, reps=20, shift=3):
    """(share of sets within 5 % of the optimum, score retained at +-shift steps)."""
    combos = all_sets(sc)
    S = score_sets_from_D(M, sc, [], combos)
    best = float(S.max())
    if best <= 0:
        return np.nan, np.nan
    near = float((S >= 0.95 * best).mean())
    opt = combos[int(S.argmax())]
    keep = []
    for _ in range(reps):
        p = [(int(a) + shift * rng.choice([-1, 1])) % ACTION_SIZE for a in opt]
        if len(set(p)) < 3 or sc in p:
            continue
        keep.append(score_of(M, sc, p) / best)
    return near, float(np.mean(keep)) if keep else np.nan


def overlap_flag(g_full, g_abl, n, thr=0.2):
    """True if at some angle one defect's signal footprint covers > 50 % of another's."""
    F = np.zeros((n, ACTION_SIZE, PROJ_LEN), bool)
    for d in range(n):
        for a in range(ACTION_SIZE):
            s = np.abs(g_full[a] - g_abl[d, a])
            m = s.max()
            F[d, a] = s > thr * m if m > 0 else False
    for a in range(ACTION_SIZE):
        for d1 in range(n):
            own = F[d1, a].sum()
            if own == 0:
                continue
            for d2 in range(n):
                if d1 != d2 and (F[d1, a] & F[d2, a]).sum() > 0.5 * own:
                    return True
    return False


def paired(x, y):
    """Mean paired difference, bootstrap 95 % CI, one-sided Wilcoxon (x > y)."""
    d = x - y
    rng = np.random.default_rng(0)
    bs = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = wilcoxon(x, y, alternative="greater").pvalue if np.any(d != 0) else np.nan
    return d.mean(), lo, hi, p


def main(a):
    from paired_stats import load, svr_oneshot
    from train_svr import SetVisibilityRanker
    svr = load(SetVisibilityRanker, a.svr) if a.svr else None
    U = dog_channels()
    scale, med = calibrate(a.data, a.body_L)
    print(f"Calibration: median body path {med:.1f} -> {a.body_L} "
          f"(factor {scale:.3e})")

    for path in a.data:
        g_full, g_abl, D_all, ndef, scout, sino = load_cho(path, a.n, scale)
        rng = np.random.default_rng(0)
        raw = {k: [] for k in FIXED}
        if svr is not None:
            raw["SVR one-shot"] = []
        fl_S, fl_C, ov = [], [], []

        for i in range(len(g_full)):
            n = int(ndef[i])
            if n == 0:
                continue
            sc = int(scout[i])
            D = D_all[i][:n].astype(np.float64)
            M = d_cho_crn(g_full[i], g_abl[i], n, a.i0, U, a.real, 10_000 * i)

            sel = {k: [(sc + o) % ACTION_SIZE for o in off]
                   for k, off in FIXED.items()}
            if svr is not None:
                sel["SVR one-shot"] = svr_oneshot(svr, sino[i], sc)
            for k, ang in sel.items():
                raw[k].append(score_of(M, sc, ang))

            fl_S.append(flatness(D, sc, rng))
            fl_C.append(flatness(M, sc, rng))
            ov.append(overlap_flag(g_full[i], g_abl[i], n))

        raw = {k: np.array(v) for k, v in raw.items()}
        ov = np.array(ov)
        fS, fC = np.array(fl_S), np.array(fl_C)

        print(f"\n{path}   (n = {len(ov)}, I0 = {a.i0:g}, {a.real} realisations, CRN)")
        print(f"  flatness   {'':<14} {'within 5 % of opt.':>18} {'score at ±15°':>16}")
        print(f"  {'original S':<26} {np.nanmean(fS[:,0])*100:>17.1f}% "
              f"{np.nanmean(fS[:,1])*100:>15.1f}%")
        print(f"  {'S_CHO':<26} {np.nanmean(fC[:,0])*100:>17.1f}% "
              f"{np.nanmean(fC[:,1])*100:>15.1f}%")

        if svr is None:
            continue
        best_fixed = max(FIXED, key=lambda k: raw[k].mean())
        print(f"\n  raw S_CHO, paired against {best_fixed}:")
        print(f"  {'subset':<20} {'n':>4} {'SVR':>8} {'fixed':>8} "
              f"{'Δ':>8} {'95% CI':>16} {'p':>9}")
        for name, mask in (("all objects", np.ones_like(ov, bool)),
                           ("no defect overlap", ~ov),
                           ("defect overlap", ov)):
            if mask.sum() < 5:
                print(f"  {name:<20} {mask.sum():>4}  (too few objects)")
                continue
            x, y = raw["SVR one-shot"][mask], raw[best_fixed][mask]
            d, lo, hi, p = paired(x, y)
            print(f"  {name:<20} {mask.sum():>4} {x.mean():>8.2f} "
                  f"{y.mean():>8.2f} {d:>+8.2f} [{lo:>+6.2f}, {hi:>+6.2f}] "
                  f"{p:>9.3g}")

    print("\np = one-sided paired Wilcoxon (SVR > best fixed) on RAW S_CHO,")
    print("without the optimum in the denominator.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--svr", nargs="*", default=[])
    p.add_argument("--i0", type=float, default=5000)
    p.add_argument("--real", type=int, default=1000)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--body-L", type=float, default=BODY_L)
    main(p.parse_args())
