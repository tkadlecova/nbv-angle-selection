"""Channelized Hotelling observer (CHO) used to validate the visibility criterion.

Core functions shared by the observer scripts: physical rescaling of the
stored transmission, difference-of-Gaussians channels, split-half AUC with a
tie-corrected Mann-Whitney statistic, and d' = sqrt(2) * Phi^-1(AUC).

Why the rescaling: the dataset stores transmission normalised per angle by
its own maximum line integral, which couples signal amplitude to the metal
path at that angle and makes Poisson noise on it unphysical. calibrate()
maps the median body line integral to BODY_L (0.17, roughly 10 mm PA 6.6
around 120 kVp) with one global factor.

    python cho_validation.py --pilot --data cho_realistic.h5
    python cho_validation.py --data cho_realistic.h5 --i0 5000 --svr svr_ens1.pt svr_ens2.pt svr_ens3.pt
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import itertools

import h5py
import numpy as np
from scipy.stats import norm, spearmanr

from train_svr import ACTION_SIZE, score_sets_from_D

BODY_L = 0.17
ROI_W = 64
N_REAL = 200
N_CHANNELS = 10
DOG_ALPHA = 1.4
DOG_SIGMA0 = 1.5
RIDGE = 1e-6

FIXED = {"+60/120/150": (12, 24, 30),
         "+45/90/135": (9, 18, 27),
         "+30/90/150": (6, 18, 30)}


def raw_L(g, denom):
    """Undo the per-angle normalisation: line integrals in simulation units."""
    return -np.log(np.clip(g, 1e-12, None)) * denom[..., None]


def calibrate(paths, body_L, probe=20):
    """Global factor mapping the median body line integral to body_L."""
    med = []
    for p in paths:
        with h5py.File(p, "r") as f:
            if "denom" not in f:
                return None, None
            L = raw_L(f["g_full"][:probe], f["denom"][:probe])
        med.append(float(np.median(L[L > 1e-6 * L.max()])))
    m = float(np.median(med))
    return body_L / m, m


def to_phys(g, denom, scale):
    return np.exp(-raw_L(g, denom) * scale).astype(np.float32)


def load_cho(path, n=None, scale=None):
    with h5py.File(path, "r") as f:
        gf, ga = f["g_full"][:n], f["g_ablated"][:n]
        D, nd, sc = f["D"][:n], f["n_def"][:n], f["scout_idx"][:n]
        sino = f["sino_all"][:n] if "sino_all" in f else None
        dn = f["denom"][:n] if "denom" in f else None
    if scale is None or dn is None:
        print("  WARNING: running in the per-angle normalised domain, not the "
              "physical one (see calibrate()); results are not physically interpretable.")
        return gf, ga, D, nd, sc, sino
    return (to_phys(gf, dn, scale), to_phys(ga, dn[:, None], scale),
            D, nd, sc, sino)


def dog_channels(width=ROI_W, n=N_CHANNELS, alpha=DOG_ALPHA, sigma0=DOG_SIGMA0):
    """1-D difference-of-Gaussians channels, zero mean, unit norm."""
    r = np.arange(width) - (width - 1) / 2.0
    U = np.empty((width, n))
    for j in range(n):
        s = sigma0 * alpha ** j
        U[:, j] = (np.exp(-r ** 2 / (2 * s ** 2))
                   - np.exp(-r ** 2 / (2 * (alpha * s) ** 2)))
    U -= U.mean(axis=0, keepdims=True)
    U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
    return U


def _roi(profile, centre, width=ROI_W):
    lo = int(np.clip(centre - width // 2, 0, len(profile) - width))
    return profile[lo:lo + width]


def _poisson(profile, i0, n, rng):
    return rng.poisson(i0 * profile[None, :], size=(n, len(profile))) / i0


def auc_cho(g_present, g_absent, i0, U, rng, n_real=N_REAL):
    """AUC of the CHO; template estimated on one half of the noise
    realisations, AUC measured on the other half."""
    s = g_present - g_absent
    if np.abs(s).max() < 1e-12:
        return 0.5
    c = int(np.argmax(np.abs(s)))
    p_roi, a_roi = _roi(g_present, c), _roi(g_absent, c)

    vp = _poisson(p_roi, i0, n_real, rng) @ U
    va = _poisson(a_roi, i0, n_real, rng) @ U

    h = n_real // 2
    dv = vp[:h].mean(0) - va[:h].mean(0)
    K = 0.5 * (np.cov(vp[:h], rowvar=False) + np.cov(va[:h], rowvar=False))
    K += RIDGE * np.trace(K) / len(K) * np.eye(len(K))
    try:
        w = np.linalg.solve(K, dv)
    except np.linalg.LinAlgError:
        return 0.5

    lp, la = vp[h:] @ w, va[h:] @ w

    # Mann-Whitney AUC with average ranks for ties
    allv = np.concatenate([lp, la])
    ranks = allv.argsort().argsort().astype(float) + 1

    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n1, n2 = len(lp), len(la)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def d_prime(auc, eps=1e-3):
    return np.sqrt(2) * norm.ppf(np.clip(auc, eps, 1 - eps))


def d_auc_matrix(g_full, g_abl, n_def, i0, U, rng, metric="dprime"):
    M = np.empty((n_def, ACTION_SIZE))
    for d in range(n_def):
        for a in range(ACTION_SIZE):
            M[d, a] = auc_cho(g_full[a], g_abl[d, a], i0, U, rng)
    return d_prime(M) if metric == "dprime" else M


def bf_from(M, sc):
    rem = [a for a in range(ACTION_SIZE) if a != sc]
    combos = np.array(list(itertools.combinations(rem, 3)), dtype=np.int64)
    return combos, score_sets_from_D(M, sc, [], combos)


def eff(M, sc, angles, best):
    s = float(score_sets_from_D(
        M, sc, [], np.array([sorted(angles)], dtype=np.int64))[0])
    return s / best if best > 0 else np.nan


def pilot(args, U):
    """Find an I0 where the AUC spans a useful range (does not saturate at 1)."""
    rng = np.random.default_rng(0)
    print("Searching for an I0 where the AUC spans a useful range.\n")
    print(f"  {'I0':>8} {'mean':>7} {'p10':>7} {'p90':>7} {'>0.99':>8} {'<0.55':>8}")
    print("  " + "-" * 50)
    scale, med = calibrate(args.data[:1], args.body_L)
    if scale:
        print(f"Calibration: median body path {med:.1f} sim. units "
              f"-> {args.body_L} (factor {scale:.3e})\n")
    g_full, g_abl, _, ndef, _, _ = load_cho(args.data[0], 10, scale)
    for i0 in args.i0_grid:
        vals = []
        for i in range(len(g_full)):
            for d in range(int(ndef[i])):
                for a in range(0, ACTION_SIZE, 3):
                    vals.append(auc_cho(g_full[i][a], g_abl[i][d, a],
                                        i0, U, rng, n_real=100))
        v = np.array(vals)
        print(f"  {i0:>8.0f} {v.mean():>7.3f} {np.percentile(v,10):>7.3f} "
              f"{np.percentile(v,90):>7.3f} {(v>0.99).mean()*100:>7.1f}% "
              f"{(v<0.55).mean()*100:>7.1f}%")
    print("\nPick an I0 where the share of AUC > 0.99 is small (below ~15 %)")
    print("and the share of AUC < 0.55 does not dominate.")


def run(args, U):
    """[1] V vs observer per (defect, angle); [2] S vs S_CHO over the 6545 sets;
    [3] efficiency of the selected sets under S and under the observer."""
    from paired_stats import load, svr_oneshot
    from train_svr import SetVisibilityRanker
    svr = load(SetVisibilityRanker, args.svr) if args.svr else None

    scale, med = calibrate(args.data, args.body_L)
    if scale:
        print(f"Calibration: median body path {med:.1f} sim. units "
              f"-> body_L = {args.body_L} (factor {scale:.3e})")
    for path in args.data:
        g_full, g_abl, D_all, ndef, scout, sino = load_cho(path, args.n, scale)
        n_obj = len(g_full)
        rng = np.random.default_rng(0)

        pair_D, pair_A = [], []
        rho_sets = []
        n_degenerate = 0
        effs = {k: [] for k in FIXED}
        if svr is not None:
            effs["SVR one-shot"] = []
        eff_ref = {k: [] for k in effs}

        for i in range(n_obj):
            n = int(ndef[i])
            if n == 0:
                continue
            sc = int(scout[i])
            D = D_all[i][:n].astype(np.float64)
            M = d_auc_matrix(g_full[i], g_abl[i], n, args.i0, U, rng,
                             metric=args.metric)

            pair_D.append(D.ravel())
            pair_A.append(M.ravel())

            combos, s_orig = bf_from(D, sc)
            _, s_cho = bf_from(M, sc)
            if s_orig.std() > 0 and s_cho.std() > 0:
                rho_sets.append(spearmanr(s_orig, s_cho).statistic)
            else:
                n_degenerate += 1

            best_cho, best_orig = float(s_cho.max()), float(s_orig.max())
            sel = {k: [(sc + o) % ACTION_SIZE for o in off]
                   for k, off in FIXED.items()}
            if svr is not None:
                sel["SVR one-shot"] = svr_oneshot(svr, sino[i], sc)
            for k, ang in sel.items():
                effs[k].append(eff(M, sc, ang, best_cho))
                eff_ref[k].append(eff(D, sc, ang, best_orig))

        rho_ang = spearmanr(np.concatenate(pair_D),
                            np.concatenate(pair_A)).statistic
        rho_sets = np.array(rho_sets)

        print(f"\n{path}   (n = {n_obj} objects, I0 = {args.i0:g}, "
              f"metric = {args.metric})")
        print(f"  [1] Spearman ρ between V(d,a) and the observer, pooled over "
              f"all pairs: {rho_ang:.3f}")
        print(f"  [2] Spearman ρ between S(A) and S_CHO(A) over the 6545 sets, "
              f"per object: median {np.median(rho_sets):.3f}, "
              f"IQR [{np.percentile(rho_sets,25):.3f}, "
              f"{np.percentile(rho_sets,75):.3f}]"
              + (f"  ({n_degenerate} objects skipped, constant score)"
                 if n_degenerate else ""))
        print(f"  [3] efficiency of the selected sets under S and under the CHO:")
        print(f"      {'method':<22} {'S':>8} {'CHO':>8} {'Δ':>8}")
        for k in effs:
            a, b = np.mean(eff_ref[k]) * 100, np.mean(effs[k]) * 100
            print(f"      {k:<22} {a:>7.1f}% {b:>7.1f}% {b-a:>+7.1f}")

        np.savez(path.replace(".h5", "_cho.npz"),
                 rho_angle=rho_ang, rho_sets=rho_sets,
                 **{f"eff_cho_{k}": np.array(v) for k, v in effs.items()},
                 **{f"eff_S_{k}": np.array(v) for k, v in eff_ref.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--svr", nargs="*", default=[])
    ap.add_argument("--i0", type=float, default=300.0)
    ap.add_argument("--i0-grid", nargs="+", type=float,
                    default=[30, 100, 300, 1000, 3000, 10000])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--metric", choices=["dprime", "auc"], default="dprime")
    ap.add_argument("--body-L", type=float, default=BODY_L,
                    help="line integral of a typical path through the body; "
                         "0.17 ~ 10 mm PA 6.6 around 120 kVp")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    U = dog_channels()
    pilot(a, U) if a.pilot else run(a, U)
