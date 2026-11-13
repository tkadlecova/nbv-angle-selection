"""Phantom generator, forward projection and per-defect visibility V(d, a).

Phantoms model moulded polymer parts with metal inserts (contact pins) and
spherical voids. Projections use ASTRA in 2-D parallel-beam geometry with
36 candidate angles over [0, 180) degrees.
"""
import os
import random
import numpy as np
import torch
import astra
from skimage.draw import disk, ellipse, polygon
from scipy.ndimage import distance_transform_edt


ACTION_SIZE = 36
IMAGE_SIZE  = 512

ANGLES_RANGE = np.linspace(0, np.pi, ACTION_SIZE, endpoint=False)
ANGLES_DEG   = np.linspace(0, 180, ACTION_SIZE, endpoint=False)


class PhantomGenerator:
    def __init__(self, seed=1029, image_size=IMAGE_SIZE,
                 orientation_probs=None):
        self.seed       = seed
        self.image_size = image_size
        if orientation_probs is None:
            orientation_probs = {'random': 0.5, 'horizontal': 0.25, 'vertical': 0.25}
        self.orientation_probs = orientation_probs
        self._seed_all()

    def _seed_all(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def generate_object(self):
        """Return (mu, defect_mask, defects_info, metal_mask, metal_angles)."""
        MU_AIR     = 0.0
        MU_PLASTIC = np.random.uniform(0.45, 0.55)
        # metal/polymer attenuation ratio 40-57: CuZn37 pin in PA66 at 100-150 kVp
        MU_METAL   = MU_PLASTIC * np.random.uniform(40.0, 57.0)
        mu          = np.full((self.image_size, self.image_size), MU_AIR, dtype=np.float32)
        defect_mask = np.zeros((self.image_size, self.image_size), dtype=np.int32)
        center      = self.image_size // 2

        body_mask = np.zeros((self.image_size, self.image_size), dtype=bool)
        if np.random.rand() < 0.5:
            r = np.random.randint(180, 230)
            rr, cc = disk((center, center), r, shape=mu.shape)
        else:
            r1, r2 = np.random.randint(150, 230, size=2)
            rr, cc = ellipse(center, center, r1, r2, shape=mu.shape)
        body_mask[rr, cc] = True
        mu[body_mask] = MU_PLASTIC

        n_metal      = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        metal_angles = []
        for _ in range(n_metal):
            metal_mask = np.zeros_like(mu, dtype=bool)
            w = np.random.randint(10, 30)
            h = np.random.randint(100, 200)
            x, y = np.random.randint(150, self.image_size - 150, size=2)

            orient_keys = list(self.orientation_probs.keys())
            orient_p    = list(self.orientation_probs.values())
            orientation = np.random.choice(orient_keys, p=orient_p)
            if orientation == 'random':
                angle_deg = np.random.uniform(0, 180)
            elif orientation == 'horizontal':
                angle_deg = np.random.uniform(0, 15)
            else:
                angle_deg = np.random.uniform(75, 105)

            metal_angles.append(angle_deg)
            angle = np.radians(angle_deg)
            corners = np.array([[-w//2, -h//2], [w//2, -h//2],
                                 [w//2,  h//2], [-w//2,  h//2]])
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            rotated = corners @ R.T
            pr, pc = polygon(rotated[:, 0] + x, rotated[:, 1] + y, shape=mu.shape)
            metal_mask[pr, pc] = True
            valid = metal_mask & body_mask
            mu[valid] = MU_METAL

        # voids only inside polymer, never touching metal
        n_defects    = np.random.randint(2, 6)
        defects_info = []
        defect_id    = 1
        for _ in range(n_defects):
            r = np.random.randint(5, 18)
            for _ in range(50):
                x, y = np.random.randint(100, self.image_size - 100, size=2)
                rr, cc = disk((x, y), r, shape=mu.shape)
                if np.all(body_mask[rr, cc]) and np.all(mu[rr, cc] == MU_PLASTIC):
                    mu[rr, cc]          = MU_AIR
                    defect_mask[rr, cc] = defect_id
                    defects_info.append({"id": defect_id, "pos": (x, y), "r": r})
                    defect_id += 1
                    break

        return mu, defect_mask, defects_info, (mu > 1.5), metal_angles


def generate_sinogram_transmission(phantom, angles):
    """Transmission exp(-L / max L), normalised per call by the maximum line integral."""
    N         = phantom.shape[0]
    proj_geom = astra.create_proj_geom('parallel', 1.0, N, angles)
    vol_geom  = astra.create_vol_geom(N, N)
    proj_id   = astra.create_projector('strip', proj_geom, vol_geom)
    sino_id, sino = astra.create_sino(phantom, proj_id)
    trans     = np.exp(-sino / (np.max(sino) + 1e-6))
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)
    return trans.astype(np.float32)


def generate_sinogram_raw(mask, angles):
    """Raw line integrals, no normalisation."""
    N         = mask.shape[0]
    proj_geom = astra.create_proj_geom('parallel', 1.0, N, angles)
    vol_geom  = astra.create_vol_geom(N, N)
    proj_id   = astra.create_projector('strip', proj_geom, vol_geom)
    sino_id, sino = astra.create_sino(mask, proj_id)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)
    return sino.astype(np.float32)


NOISE_FLOOR = 0.005
CNR_CLIP    = 50.0


def _defect_visibility_single(projection, defect_sino, metal_binary,
                                  dist_map, all_defect_sinos, defect_id):
    """V(d, a) = CNR * mu_d * P_metal * P_dist * P_unique, scaled by 100."""
    if defect_sino.sum() < 1e-6:
        return 0.0
    defect_region = defect_sino > 0.01
    if not defect_region.any():
        return 0.0
    bg_region = ~defect_region & ~metal_binary
    if bg_region.sum() < 5:
        return 0.0

    signal    = projection[defect_region].mean()
    noise_std = max(float(projection[bg_region].std()), NOISE_FLOOR)
    CNR       = min(abs(1.0 - signal) / noise_std, CNR_CLIP)

    mu_d      = defect_sino[defect_region].mean()
    P_metal   = 1.0 - float(metal_binary[defect_region].mean())
    dist_vals = dist_map[defect_region]
    P_dist    = float(np.clip(dist_vals.mean() / 50.0, 0, 1))

    # halve the score for every other defect covering > 50 % of this footprint
    P_unique = 1.0
    for other_id, other_sino in all_defect_sinos.items():
        if other_id == defect_id:
            continue
        other_region = other_sino > 0.01
        overlap      = (defect_region & other_region).sum()
        if overlap > 0.5 * defect_region.sum():
            P_unique *= 0.5

    return float(CNR * mu_d * P_metal * P_dist * P_unique) * 100.0


def compute_visibility_per_defect(phantom, defect_mask, metal_map, angle_idx):
    """Return {defect_id: V} for one projection angle."""
    angle = ANGLES_RANGE[angle_idx]
    proj  = generate_sinogram_transmission(phantom, [angle])[0]

    metal_proj   = generate_sinogram_transmission(
        metal_map.astype(np.float32), [angle]
    )[0]
    metal_binary = (metal_proj < 1)
    dist_map     = distance_transform_edt(~metal_binary)

    defect_ids = np.unique(defect_mask)
    defect_ids = defect_ids[defect_ids > 0]

    all_sinos = {}
    for d_id in defect_ids:
        binary = (defect_mask == d_id).astype(np.float32)
        all_sinos[int(d_id)] = generate_sinogram_raw(binary, [angle])[0]

    result = {}
    for d_id in defect_ids:
        d_id = int(d_id)
        result[d_id] = _defect_visibility_single(
            proj, all_sinos[d_id], metal_binary, dist_map,
            all_sinos, d_id
        )
    return result


def update_best_so_far(best_so_far, new_per_defect):
    out = dict(best_so_far)
    for d_id, val in new_per_defect.items():
        if d_id not in out or val > out[d_id]:
            out[d_id] = val
    return out


def score_from_best(best_so_far):
    """S = 0.7 * min_d + 0.3 * mean_d of the best V per defect."""
    if not best_so_far:
        return 0.0
    vals = np.array(list(best_so_far.values()))
    return float(0.7 * np.min(vals) + 0.3 * np.mean(vals))
