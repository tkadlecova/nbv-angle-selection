"""Paired projections for the model-observer validation.

For every object: transmission of the full phantom at all 36 angles
(g_full) and, for every defect, the same phantom with that defect filled
with polymer (g_ablated). Both use the SAME per-angle normaliser (denom), so
the difference between them is the defect signal only. The V matrix and the
scout are stored too; with the same --dist and --seed-offset the objects are
identical to those of generate_vmatrix_dataset.py (checked by --verify-against).

    python make_cho_dataset.py --dist realistic --n 50 --seed-offset 950000 \
        --out cho_realistic.h5 --verify-against dmx_realistic_final.h5
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse

import astra
import h5py
import numpy as np
from tqdm import tqdm

import generate_vmatrix_dataset as G
from phantom_utils import (PhantomGenerator, ACTION_SIZE, IMAGE_SIZE,
                           ANGLES_RANGE)

MAXD = G.MAX_DEF
_PROJ = {}


def sino_raw(image):
    """Raw line integrals at all 36 angles (projector cached per size)."""
    N = image.shape[0]
    if N not in _PROJ:
        pg = astra.create_proj_geom("parallel", 1.0, N, ANGLES_RANGE)
        vg = astra.create_vol_geom(N, N)
        _PROJ[N] = astra.create_projector("strip", pg, vg)
    sid, sino = astra.create_sino(image.astype(np.float32), _PROJ[N])
    astra.data2d.delete(sid)
    return sino.astype(np.float32)


def polymer_mu(mu):
    body = mu[(mu > 0.1) & (mu < 1.5)]
    if body.size == 0:
        raise RuntimeError("no polymer found in the phantom")
    v = float(np.median(body))
    if body.std() > 0.01:
        print(f"  warning: polymer is not constant (std={body.std():.4f}), "
              f"filling with the median {v:.4f}")
    return v


def transmission(raw, denom):
    return np.exp(-raw / denom).astype(np.float32)


def main(a):
    probs = G.ORIENTATION_PRESETS[a.dist]
    rng = np.random.default_rng(12345 + a.seed_offset)

    g_full = np.zeros((a.n, ACTION_SIZE, IMAGE_SIZE), np.float32)
    g_abl = np.zeros((a.n, MAXD, ACTION_SIZE, IMAGE_SIZE), np.float32)
    denom_out = np.zeros((a.n, ACTION_SIZE), np.float32)
    D_out = np.zeros((a.n, MAXD, ACTION_SIZE), np.float32)
    ndef = np.zeros(a.n, np.int64)
    scout = np.zeros(a.n, np.int64)

    # same acceptance loop and scout draw as generate_vmatrix_dataset.generate()
    made, attempt_seed = 0, a.seed_offset
    pbar = tqdm(total=a.n, desc=f"paired projections [{a.dist}]")
    while made < a.n:
        attempt_seed += 1
        gen = PhantomGenerator(seed=attempt_seed, image_size=IMAGE_SIZE,
                               orientation_probs=probs)
        mu_ph, d_mask, _ = gen.generate_object()[:3]
        ids = sorted(int(d) for d in np.unique(d_mask) if d > 0)
        if len(ids) == 0 or len(ids) > MAXD:
            continue

        metal_map = mu_ph > 1.5
        D = G.visibility_matrix(mu_ph, d_mask, metal_map)
        assert len(D) == len(ids), (
            f"V has {len(D)} rows but the mask has {len(ids)} defects; "
            "rows of V would not match the ablated defects")
        s = int(rng.integers(0, ACTION_SIZE))

        L_full = sino_raw(mu_ph)
        denom = L_full.max(axis=1, keepdims=True) + 1e-6
        g_full[made] = transmission(L_full, denom)
        denom_out[made] = denom[:, 0]

        mu_poly = polymer_mu(mu_ph)
        for j, d in enumerate(ids):
            mu_d = mu_ph.copy()
            mu_d[d_mask == d] = mu_poly
            g_abl[made, j] = transmission(sino_raw(mu_d), denom)

        D_out[made, :len(D)] = D
        ndef[made] = len(ids)
        scout[made] = s
        made += 1
        pbar.update(1)
    pbar.close()

    with h5py.File(a.out, "w") as f:
        f.create_dataset("g_full", data=g_full, compression="gzip")
        f.create_dataset("g_ablated", data=g_abl, compression="gzip")
        f.create_dataset("sino_all", data=g_full, compression="gzip")
        f.create_dataset("D", data=D_out, compression="gzip")
        f.create_dataset("denom", data=denom_out)
        f.create_dataset("n_def", data=ndef)
        f.create_dataset("scout_idx", data=scout)
        f.attrs["dist"] = a.dist
        f.attrs["seed_offset"] = a.seed_offset
        f.attrs["max_def"] = MAXD

    sig = np.array([np.abs(g_full[i][None] - g_abl[i, :ndef[i]]).max()
                    for i in range(a.n) if ndef[i] > 0])
    print(f"\nSaved: {a.out}  ({a.n} objects, {ndef.mean():.1f} defects on average)")
    print(f"Signal amplitude: median {np.median(sig):.5f}, min {sig.min():.5f}")
    if sig.min() < 1e-6:
        print("WARNING: zero signal for some object, a defect was not filled. "
              "Check polymer_mu() and the values in d_mask.")

    if a.verify_against:
        verify(a.verify_against, g_full, D_out, ndef, scout)


def verify(path, g_full, D_out, ndef, scout):
    """Check that the objects are the same as in the V-matrix dataset."""
    n = len(g_full)
    with h5py.File(path, "r") as f:
        ref_s = f["sino_all"][:n].astype(np.float32)
        ref_D = f["D"][:n].astype(np.float32)
        ref_nd, ref_sc = f["n_def"][:n], f["scout_idx"][:n]

    dD = np.abs(D_out - ref_D).max()
    dS = np.abs(g_full - ref_s).max()
    same_nd = int((ndef == ref_nd).sum())
    same_sc = int((scout == ref_sc).sum())

    print(f"\nCheck against {path} (first {n} objects):")
    print(f"  max |V - V_ref|          = {dD:.4g}")
    print(f"  max |sino - sino_ref|    = {dS:.4g}   "
          f"(differences of ~1e-3 are fine)")
    print(f"  same number of defects   = {same_nd}/{n}")
    print(f"  same scout angle         = {same_sc}/{n}")
    ok = dD < 1e-2 and dS < 5e-3 and same_nd == n and same_sc == n
    print("  => " + ("OK, same objects and same normalisation."
                     if ok else
                     "MISMATCH, do not use until resolved. Most common cause: "
                     "different transmission normalisation or shifted seeds."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dist", required=True,
                   choices=["structured", "realistic", "random"])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed-offset", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--verify-against", default=None,
                   help="dmx_*.h5 with the same dist and seed offset")
    main(p.parse_args())
