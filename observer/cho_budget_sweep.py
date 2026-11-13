"""Headroom under the observer for budgets of 1, 2 and 3 angles.

For each budget k: percentile of the equally spaced protocol among all
k-angle candidates, naive and reproducible (split-half) headroom above it,
and, with --svr, SVR against the equally spaced protocol (paired, raw S_CHO).
The comparison protocol is scout + j * 180 / (k + 1), j = 1..k; for k = 3
it coincides with +45/+90/+135.

    python cho_budget_sweep.py --dist realistic --seed-offset 950000 --cho cho_realistic.h5 \
        --svr svr_ens1.pt svr_ens2.pt svr_ens3.pt
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import itertools

import numpy as np
import torch

import generate_vmatrix_dataset as G
from phantom_utils import PhantomGenerator, ACTION_SIZE, IMAGE_SIZE
from cho_validation import calibrate, load_cho, dog_channels, BODY_L
from cho_headroom import d_cho_halves
from cho_compare import paired
from train_svr import score_sets_from_D, PROJ_LEN, MAX_CTX, device

MAXD = G.MAX_DEF


def equal_spacing(scout, k):
    """scout + j * 180 / (k + 1); for k = 3 gives +45/90/135."""
    step = 180.0 / (k + 1)
    return sorted((scout + int(round(step * j / 5.0))) % ACTION_SIZE
                  for j in range(1, k + 1))


def combos_k(scout, k):
    rem = [a for a in range(ACTION_SIZE) if a != scout]
    return np.array(list(itertools.combinations(rem, k)), dtype=np.int64)


@torch.no_grad()
def svr_pick(models, sino, sc, k):
    """SVR one-shot over k-angle completions (k < 3 is outside the training setting)."""
    from train_svr import fbp_coarse
    cb = combos_k(sc, k)
    pad = np.full((len(cb), 3), -1, dtype=np.int64)
    pad[:, :k] = cb
    projs = sino[[sc]].astype(np.float32)
    geo = fbp_coarse(projs, [sc])[None]
    ctx_p = np.zeros((MAX_CTX, PROJ_LEN), dtype=np.float32)
    ctx_a = np.full(MAX_CTX, -1, dtype=np.int64)
    ctx_p[0], ctx_a[0] = projs[0], sc
    t = lambda x: torch.as_tensor(x, device=device).unsqueeze(0)
    zs = []
    for m in models:
        s = m(t(ctx_p), t(ctx_a), t(geo).float(), t(pad))[0]
        zs.append((s - s.mean()) / (s.std() + 1e-6))
    s = torch.stack(zs).mean(0).cpu().numpy()
    return sorted(int(x) for x in cb[int(s.argmax())])


def main(a):
    models = None
    if a.svr:
        from paired_stats import load
        from train_svr import SetVisibilityRanker
        models = load(SetVisibilityRanker, a.svr)

    U = dog_channels()
    scale, med = calibrate([a.cho], a.body_L)
    print(f"Calibration: median body path {med:.1f} -> {a.body_L}")
    g_full, g_abl, D_ref, ndef, scout, sino = load_cho(a.cho, a.n, scale)

    probs = G.ORIENTATION_PRESETS[a.dist]
    rng_seed = np.random.default_rng(12345 + a.seed_offset)
    res = {k: {"fix": [], "pct": [], "naive": [], "oos": [], "rng": [],
               "svr": []} for k in a.budgets}

    made, attempt_seed, mism = 0, a.seed_offset, 0
    while made < len(g_full):
        attempt_seed += 1
        gen = PhantomGenerator(seed=attempt_seed, image_size=IMAGE_SIZE,
                               orientation_probs=probs)
        mu, dmask, _ = gen.generate_object()[:3]
        ids = sorted(int(d) for d in np.unique(dmask) if d > 0)
        if len(ids) == 0 or len(ids) > MAXD:
            continue
        Dm = G.visibility_matrix(mu, dmask, mu > 1.5)
        Dm = np.asarray(Dm, float)
        _ = int(rng_seed.integers(0, ACTION_SIZE))
        n, sc = int(ndef[made]), int(scout[made])
        ref_D = np.asarray(D_ref[made][:n], float)
        if Dm.shape != ref_D.shape or np.abs(Dm - ref_D).max() > 1e-3:
            mism += 1

        MA, MB = d_cho_halves(g_full[made], g_abl[made], n, a.i0, U,
                              a.real // 2, 10_000 * made)

        for k in a.budgets:
            cb = combos_k(sc, k)
            SA = score_sets_from_D(MA, sc, [], cb)
            SB = score_sets_from_D(MB, sc, [], cb)
            idx = {tuple(c): i for i, c in enumerate(cb.tolist())}
            rank = SB.argsort().argsort() / (len(SB) - 1) * 100

            i_fix = idx[tuple(equal_spacing(sc, k))]
            res[k]["fix"].append(float(SB[i_fix]))
            res[k]["pct"].append(float(rank[i_fix]))
            res[k]["naive"].append(float(SB.max() - SB[i_fix]))
            res[k]["oos"].append(float(SB[int(SA.argmax())] - SB[i_fix]))
            res[k]["rng"].append(float(SB.max() - SB.min()))
            if models is not None:
                res[k]["svr"].append(
                    float(SB[idx[tuple(svr_pick(models, sino[made], sc, k))]]))
        made += 1

    if mism:
        raise SystemExit(f"{mism} objects do not match {a.cho}.")
    print(f"Checksum: all {made} phantoms match {a.cho}.\n")

    print(f"{a.cho}  (n = {made}, I0 = {a.i0:g})")
    print("  {:>7} {:>9} {:>10} {:>11} {:>12} {:>10} {:>9}".format(
        "budget", "candid.", "range", "fix pctl", "naive head.",
        "reproduc.", "% noise"))
    print("  " + "-" * 74)
    for k in a.budgets:
        r = res[k]
        nv, oo = np.mean(r["naive"]), np.mean(r["oos"])
        print(f"  {k:>7} {len(combos_k(0, k)):>9} {np.mean(r['rng']):>10.3f} "
              f"{np.mean(r['pct']):>10.1f}% {nv:>12.3f} {oo:>11.3f} "
              f"{(1 - oo / nv) * 100 if nv > 0 else float('nan'):>8.0f}%")

    if models is not None:
        print("\n  SVR vs the equally spaced protocol (paired, raw S_CHO):")
        for k in a.budgets:
            r = res[k]
            v, f = np.array(r["svr"]), np.array(r["fix"])
            d, lo, hi, p = paired(v, f)
            print(f"    budget {k}: {v.mean():>6.3f} vs {f.mean():>6.3f}  "
                  f"delta {d:>+6.3f} [{lo:>+6.3f}, {hi:>+6.3f}]  p = {p:.3g}")

    print("\n  'reproduc.' = optimum found on one half of the realisations and")
    print("  measured on the other; 'fix pctl' = percentile of the equally")
    print("  spaced protocol among all candidates.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dist", required=True,
                   choices=["structured", "realistic", "random"])
    p.add_argument("--seed-offset", type=int, required=True)
    p.add_argument("--cho", required=True)
    p.add_argument("--svr", nargs="*", default=[])
    p.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--i0", type=float, default=5000)
    p.add_argument("--real", type=int, default=1000)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--body-L", type=float, default=BODY_L)
    main(p.parse_args())
