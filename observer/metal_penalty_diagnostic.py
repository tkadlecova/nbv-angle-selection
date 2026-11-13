"""Binary vs graded metal penalty, compared against the observer d'.

V is rebuilt from its factors without any division, so every variant is
evaluated on the same set of defect x angle pairs:
  binary P_metal (as in V), sqrt(T_metal), linear T_metal, and V without a
  metal factor; T_metal is the mean transmission through the metal over the
  defect footprint for a metal/polymer ratio of --ratio.
The decomposition is checked against the stored V (relative error < 1e-5).

    python metal_penalty_diagnostic.py --dist realistic --seed-offset 950000 --cho cho_realistic.h5
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
from cho_compare import d_cho_crn

MAXD = G.MAX_DEF
BODY_PX = 400.0


def per_angle(phantom, dmask, metal, ang, mu_metal_px):
    """Per defect: (P_metal binary, metal path, sqrt(T_metal), T_metal, V without P_metal)."""
    angle = ANGLES_RANGE[ang]
    proj = generate_sinogram_transmission(phantom, [angle])[0]
    mp = generate_sinogram_transmission(metal.astype(np.float32), [angle])[0]
    metal_binary = (mp < 1)
    dist_map = distance_transform_edt(~metal_binary)
    metal_path = generate_sinogram_raw(metal.astype(np.float32), [angle])[0]

    ids = sorted(int(d) for d in np.unique(dmask) if d > 0)
    sinos = {d: generate_sinogram_raw((dmask == d).astype(np.float32),
                                      [angle])[0] for d in ids}
    nan5 = (np.nan,) * 5
    out = []
    for d in ids:
        ds = sinos[d]
        dr = ds > 0.01
        if ds.sum() < 1e-6 or not dr.any():
            out.append(nan5); continue
        bg = ~dr & ~metal_binary
        if bg.sum() < 5:
            out.append(nan5); continue

        noise = max(float(proj[bg].std()), PU.NOISE_FLOOR)
        CNR = min(abs(1.0 - proj[dr].mean()) / noise, PU.CNR_CLIP)
        mu_d = ds[dr].mean()
        p_bin = 1.0 - float(metal_binary[dr].mean())
        p_dist = float(np.clip(dist_map[dr].mean() / 50.0, 0, 1))
        p_uniq = 1.0
        for o in ids:
            if o != d and (dr & (sinos[o] > 0.01)).sum() > 0.5 * dr.sum():
                p_uniq *= 0.5

        path = float(metal_path[dr].mean())

        # mean of exp(-path), not exp(-mean path)
        t_metal = float(np.exp(-metal_path[dr] * mu_metal_px).mean())
        v_rest = float(CNR * mu_d * p_dist * p_uniq) * 100.0
        out.append((p_bin, path, np.sqrt(t_metal), t_metal, v_rest))
    return out


def rho(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 30 or np.std(x[m]) == 0:
        return np.nan, int(m.sum())
    return float(spearmanr(x[m], y[m]).statistic), int(m.sum())


def main(a):
    U = dog_channels()
    scale, med = calibrate([a.cho], a.body_L)
    print(f"Calibration: median body path {med:.1f} -> {a.body_L}")
    g_full, g_abl, D_ref, ndef, scout, _ = load_cho(a.cho, a.n, scale)

    mu_poly_px = a.body_L / BODY_PX
    mu_metal_px = a.ratio * mu_poly_px
    print(f"mu_metal = {mu_metal_px:.5f} /px  (ratio {a.ratio})")

    probs = G.ORIENTATION_PRESETS[a.dist]
    rng_seed = np.random.default_rng(12345 + a.seed_offset)
    P, PATH, GRAD, GLIN, REST, DP, VV = ([] for _ in range(7))

    made, attempt_seed, mism, vmax, vbig = 0, a.seed_offset, 0, 0.0, 1.0
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
        n = int(ndef[made])
        ref_D = np.asarray(D_ref[made][:n], float)
        if Dm.shape != ref_D.shape or np.abs(Dm - ref_D).max() > 1e-3:
            mism += 1

        M = d_cho_crn(g_full[made], g_abl[made], n, a.i0, U, a.real,
                      10_000 * made)
        for ang in range(ACTION_SIZE):
            for d, row in enumerate(per_angle(mu, dmask, metal, ang,
                                              mu_metal_px)):
                if d >= n:
                    break
                pb, pt, gr, gl, vr = row
                P.append(pb); PATH.append(pt); GRAD.append(gr)
                GLIN.append(gl); REST.append(vr)
                DP.append(M[d, ang]); VV.append(Dm[d, ang])

                if not np.isnan(vr):
                    vmax = max(vmax, abs(vr * pb - Dm[d, ang]))
                    vbig = max(vbig, abs(Dm[d, ang]))
        made += 1

    if mism:
        raise SystemExit(f"{mism} objects do not match {a.cho}.")
    print(f"Checksum: all {made} phantoms match.")
    rel = vmax / vbig
    print(f"Decomposition check: max |v_rest * P_metal - V| = {vmax:.2e} "
          f"at max |V| = {vbig:.1f}, i.e. relative {rel:.2e}")
    if rel > 1e-5:
        raise SystemExit("The decomposition of V does not match; cannot continue.")
    print("  (consistent with the float32 precision of the stored V)")
    print()

    P, PATH, GRAD = map(np.array, (P, PATH, GRAD))
    GLIN, REST = np.array(GLIN), np.array(REST)
    DP, VV = np.array(DP), np.array(VV)
    shad = PATH > a.min_path

    print(f"pairs total {len(P)}, shadowed (metal path > {a.min_path} px): "
          f"{shad.sum()} ({shad.mean()*100:.1f} %)")
    print(f"metal path: median {np.nanmedian(PATH[shad]):.1f} px, "
          f"90th percentile {np.nanpercentile(PATH[shad], 90):.1f} px, "
          f"max {np.nanmax(PATH[shad]):.1f} px")
    print(f"share of pairs with P_metal = 0 (fully shadowed): "
          f"{(P[shad] == 0).mean()*100:.1f} %\n")

    print("(A) The metal factor alone against the observer, shadowed pairs")
    for name, v in (("P_metal binary", P),
                    ("sqrt(T_metal)", GRAD),
                    ("T_metal (linear)", GLIN)):
        r, nn = rho(v[shad], DP[shad])
        print(f"  {name:<24} rho vs d' = {r:>+.3f}   (n = {nn})")

    print("\n(B) Full V rebuilt WITHOUT division (same pairs for every variant)")
    variants = [("V, binary P_metal", REST * P),
                ("V, sqrt(T_metal)", REST * GRAD),
                ("V, T_metal linear", REST * GLIN),
                ("V without metal factor", REST)]
    for name, v in variants:
        r, nn = rho(v[shad], DP[shad])
        print(f"  {name:<24} rho vs d' = {r:>+.3f}   (n = {nn})")

    print("\n(C) The same on ALL pairs, shadowed or not")
    for name, v in variants:
        r, nn = rho(v, DP)
        print(f"  {name:<24} rho vs d' = {r:>+.3f}   (n = {nn})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dist", required=True,
                   choices=["structured", "realistic", "random"])
    p.add_argument("--seed-offset", type=int, required=True)
    p.add_argument("--cho", required=True)
    p.add_argument("--i0", type=float, default=5000)
    p.add_argument("--real", type=int, default=1000)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--body-L", type=float, default=BODY_L)
    p.add_argument("--ratio", type=float, default=48.0,
                   help="mu_metal / mu_polymer; centre of the range U(40, 57)")
    p.add_argument("--min-path", type=float, default=0.5,
                   help="minimum metal path [px] for a pair to count as shadowed")
    main(p.parse_args())
