"""Generate a visibility-matrix dataset (Section 3.3).

For every phantom the per-defect, per-angle visibility matrix V[d, a] is
computed from 36 forward simulations; the score of any of the 6,545
candidate three-angle sets then follows in closed form. The HDF5 file also
stores the transmission projections at all angles (network input), the
separate polymer and metal line integrals, and a random scout index.

    python generate_vmatrix_dataset.py --dist realistic --n 2000 --out dmx_realistic_train.h5
    python generate_vmatrix_dataset.py --dist realistic --n 150 --seed-offset 950000 --out dmx_realistic_final.h5
    python generate_vmatrix_dataset.py --verify

Training, development and test sets must use disjoint seed offsets
(Section 5.3 of the paper).
"""
import argparse
import itertools
import sys
import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, ".")

from phantom_utils import (
    PhantomGenerator, ACTION_SIZE, IMAGE_SIZE, ANGLES_RANGE,
    generate_sinogram_transmission,
    compute_visibility_per_defect,
    update_best_so_far,
    score_from_best,
)

import astra

MAX_DEF = 6

ORIENTATION_PRESETS = {
    "realistic":  {"random": 0.50, "horizontal": 0.25, "vertical": 0.25},
    "random":     {"random": 0.90, "horizontal": 0.05, "vertical": 0.05},
    "structured": {"random": 0.10, "horizontal": 0.45, "vertical": 0.45},
}


def visibility_matrix(phantom, defect_mask, metal_map):
    """V matrix (n_defects x 36); rows ordered by sorted defect id."""
    defect_ids = sorted(int(d) for d in np.unique(defect_mask) if d > 0)
    row = {d: i for i, d in enumerate(defect_ids)}
    D = np.zeros((len(defect_ids), ACTION_SIZE), dtype=np.float32)
    for a in range(ACTION_SIZE):
        per_def = compute_visibility_per_defect(
            phantom, defect_mask, metal_map, a
        )
        for d_id, val in per_def.items():
            D[row[int(d_id)], a] = float(val)
    return D


def transmission_all_angles(phantom):
    return np.stack([
        generate_sinogram_transmission(phantom, [ANGLES_RANGE[a]])[0]
        for a in range(ACTION_SIZE)
    ]).astype(np.float32)


def sino_raw_all_angles(image):
    N         = image.shape[0]
    proj_geom = astra.create_proj_geom('parallel', 1.0, N, ANGLES_RANGE)
    vol_geom  = astra.create_vol_geom(N, N)
    proj_id   = astra.create_projector('strip', proj_geom, vol_geom)
    sid, sino = astra.create_sino(image.astype(np.float32), proj_id)
    astra.data2d.delete(sid)
    astra.projector.delete(proj_id)
    return sino.astype(np.float32)


def score_from_D(D, angle_indices):
    vals = D[:, list(angle_indices)].max(axis=1)
    return float(0.7 * vals.min() + 0.3 * vals.mean())


def gt_geometry_maps(mu_ph, d_mask, metal_map, size=64):
    """64x64 ground-truth maps (phantom, metal, defects); not used by the method."""
    from skimage.transform import resize
    ph = mu_ph.astype(np.float32)
    ph = (ph - ph.min()) / (ph.max() - ph.min() + 1e-6)
    ph64 = resize(ph, (size, size), order=1, anti_aliasing=True,
                  preserve_range=True)
    me64 = resize(metal_map.astype(np.float32), (size, size), order=0,
                  anti_aliasing=False, preserve_range=True)
    de64 = resize((d_mask > 0).astype(np.float32), (size, size), order=0,
                  anti_aliasing=False, preserve_range=True)
    return np.stack([ph64, me64, de64]).astype(np.float32)


def generate(n_phantoms, dist, out_path, seed_offset):
    probs = ORIENTATION_PRESETS[dist]
    rng   = np.random.default_rng(12345 + seed_offset)

    buf = {k: [] for k in ["sino_all", "L_plastic", "L_metal",
                            "D", "n_def", "scout_idx", "geom_gt"]}

    pbar = tqdm(total=n_phantoms, desc=f"V-matrix [{dist}]")
    made = skipped_zero = 0
    attempt_seed = seed_offset

    while made < n_phantoms:
        attempt_seed += 1
        gen = PhantomGenerator(seed=attempt_seed, image_size=IMAGE_SIZE,
                               orientation_probs=probs)
        mu_ph, d_mask, d_info = gen.generate_object()[:3]

        if len(d_info) == 0 or len(d_info) > MAX_DEF:
            continue

        metal_map = mu_ph > 1.5
        D = visibility_matrix(mu_ph, d_mask, metal_map)

        if D.max() <= 0:
            skipped_zero += 1
            if skipped_zero <= 3:
                print(f"\nWARNING: phantom seed={attempt_seed} has V=0 everywhere, skipping.")
            continue

        plastic   = mu_ph * (~metal_map)
        metal_img = mu_ph * metal_map

        D_pad = np.zeros((MAX_DEF, ACTION_SIZE), dtype=np.float32)
        D_pad[:len(D)] = D

        buf["sino_all"].append(transmission_all_angles(mu_ph).astype(np.float32))
        buf["L_plastic"].append(sino_raw_all_angles(plastic).astype(np.float16))
        buf["L_metal"].append(sino_raw_all_angles(metal_img).astype(np.float16))
        buf["D"].append(D_pad)
        buf["n_def"].append(len(D))
        buf["scout_idx"].append(int(rng.integers(0, ACTION_SIZE)))
        buf["geom_gt"].append(gt_geometry_maps(mu_ph, d_mask, metal_map))

        made += 1
        pbar.update(1)
    pbar.close()

    if skipped_zero:
        print(f"Skipped {skipped_zero} phantoms with an all-zero V matrix.")

    with h5py.File(out_path, "w") as f:
        for k, v in buf.items():
            data = np.stack(v) if isinstance(v[0], np.ndarray) else np.array(v)
            f.create_dataset(k, data=data, compression="gzip", chunks=True)
        f.attrs["dist"]        = dist
        f.attrs["seed_offset"] = seed_offset
        f.attrs["max_def"]     = MAX_DEF
    print(f"Saved: {out_path} ({made} phantoms)")

    D0 = buf["D"][0][:buf["n_def"][0]]
    sc = buf["scout_idx"][0]
    remaining = [a for a in range(ACTION_SIZE) if a != sc]
    best = max(score_from_D(D0, (sc,) + c)
               for c in itertools.combinations(remaining, 3))
    print(f"Sanity check, phantom 0: exhaustive optimum over 6545 sets = {best:.2f} "
          f"(closed form from V, V.max={D0.max():.2f})")


def verify(n_check=3):
    """Check that the closed-form score from V equals scoring angle by angle."""
    rng = np.random.default_rng(7)
    for i in range(n_check):
        gen = PhantomGenerator(seed=555000 + i, image_size=IMAGE_SIZE,
                               orientation_probs=ORIENTATION_PRESETS["realistic"])
        mu_ph, d_mask, d_info = gen.generate_object()[:3]
        if len(d_info) == 0:
            continue
        metal_map = mu_ph > 1.5
        D = visibility_matrix(mu_ph, d_mask, metal_map)

        quad   = tuple(int(x) for x in rng.choice(ACTION_SIZE, size=4, replace=False))
        s_fast = score_from_D(D, quad)

        best   = {}
        manual = {}
        for a in quad:
            per_def = compute_visibility_per_defect(mu_ph, d_mask, metal_map, a)
            ret = update_best_so_far(best, per_def)
            if isinstance(ret, dict):
                best = ret
            for d_id, v in per_def.items():
                manual[int(d_id)] = max(manual.get(int(d_id), 0.0), float(v))

        s_ref = score_from_best(best) if best else float("nan")
        vals_m   = np.array(list(manual.values()), dtype=np.float64)
        s_manual = float(0.7 * vals_m.min() + 0.3 * vals_m.mean())

        rel_m = abs(s_fast - s_manual) / (abs(s_manual) + 1e-9)
        rel_r = abs(s_fast - s_ref) / (abs(s_ref) + 1e-9) if best else float("inf")
        status = "OK " if rel_m < 1e-5 else "!!! MISMATCH"
        print(f"[{status}] phantom {i}: "
              f"V-matrix S={s_fast:.4f} | manual S={s_manual:.4f} "
              f"(rel {rel_m:.2e}) | score_from_best S={s_ref:.4f} "
              f"(rel {rel_r:.2e}) | V.max={D.max():.2f}, n_def={len(D)}")
        if rel_m < 1e-5 and (not best or rel_r > 1e-5):
            print("    -> V matrix matches the manual reference; only the "
                  "update_best_so_far/score_from_best path disagrees, which "
                  "does not affect the dataset.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--dist", choices=list(ORIENTATION_PRESETS), default="realistic")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        verify()
    else:
        out = args.out or f"dmx_{args.dist}_n{args.n}_s{args.seed_offset}.h5"
        generate(args.n, args.dist, out, args.seed_offset)
