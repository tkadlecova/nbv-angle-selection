"""Train the Visibility Set Ranker (SVR; called SVR / SetVisibilityRanker in the code).

Listwise KL training over complete candidate angle sets (Section 4).
Each sample takes the scout plus j = 0, 1 or 2 already-acquired angles
(probabilities 0.5 / 0.3 / 0.2), enumerates all completions to three
angles, and scores a list of up to 64 of them exactly from the V matrix.

    python train_svr.py --train dmx_realistic_train.h5 dmx_random_train.h5 \
        dmx_structured_train.h5 --val dmx_dev.h5 --noise

The best checkpoint (lowest validation regret@1) is written to
svr_best.pt. The paper reports an ensemble of three independently
trained models (svr_ens1/2/3.pt).
"""
import argparse
import itertools
import os
import csv
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ACTION_SIZE = 36
PROJ_LEN    = 512
MAX_CTX     = 3
LIST_SIZE   = 64
TAU         = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_sets_from_D(D, scout, ctx_extra, cand_sets):
    """S(A) for every candidate set; the scout and context angles are part of A."""
    base = D[:, [scout] + list(ctx_extra)].max(axis=1, keepdims=True)
    vals = D[:, cand_sets]
    best = np.maximum(vals.max(axis=2), base)
    return 0.7 * best.min(axis=0) + 0.3 * best.mean(axis=0)


def add_poisson_noise(trans, rng, i0_log_range=(3.0, 5.0)):
    i0 = 10.0 ** rng.uniform(*i0_log_range)
    return rng.poisson(np.clip(trans, 0, 1) * i0).astype(np.float32) / i0


ANGLES   = np.linspace(0, np.pi, ACTION_SIZE, endpoint=False)
FBP_SIZE = 64

def fbp_coarse(projs, angle_idx, size=FBP_SIZE):
    """Coarse 64x64 filtered back-projection from the acquired projections."""
    k, L  = projs.shape
    att   = -np.log(np.clip(projs.astype(np.float32), 1e-3, 1.0))
    ramp  = 2.0 * np.abs(np.fft.rfftfreq(L)).astype(np.float32)
    filt  = np.fft.irfft(np.fft.rfft(att, axis=1) * ramp[None], n=L, axis=1)
    xs    = np.linspace(-1, 1, size, dtype=np.float32)
    X, Y  = np.meshgrid(xs, xs, indexing="xy")
    det   = np.arange(L, dtype=np.float32)
    img   = np.zeros((size, size), dtype=np.float32)
    for p, ai in zip(filt, angle_idx):
        th = ANGLES[int(ai)]
        t  = (X * np.cos(th) + Y * np.sin(th) + 1.0) * 0.5 * (L - 1)
        img += np.interp(t.ravel(), det, p).reshape(size, size)
    img /= max(k, 1)
    return ((img - img.mean()) / (img.std() + 1e-6)).astype(np.float32)


class RankerDataset(Dataset):
    J_PROBS = [0.5, 0.3, 0.2]

    def __init__(self, h5_paths, noise=False, samples_per_phantom=8,
                 hard_top=0, geom_input="fbp"):
        sinos, Ds, ndefs, scouts, geoms = [], [], [], [], []
        for p in h5_paths:
            with h5py.File(p, "r") as f:
                sinos.append(f["sino_all"][:])
                Ds.append(f["D"][:])
                ndefs.append(f["n_def"][:])
                scouts.append(f["scout_idx"][:])
                if geom_input == "gt":
                    if "geom_gt" not in f:
                        raise KeyError(
                            f"{p} has no 'geom_gt'; regenerate it with "
                            "generate_vmatrix_dataset.py")
                    geoms.append(f["geom_gt"][:])
        self.sino  = np.concatenate(sinos)
        self.D     = np.concatenate(Ds)
        self.ndef  = np.concatenate(ndefs)
        self.scout = np.concatenate(scouts)
        self.geom  = np.concatenate(geoms) if geoms else None
        self.geom_input = geom_input
        self.noise = noise
        self.spp   = samples_per_phantom
        self.hard_top = hard_top
        print(f"RankerDataset: {len(self.sino)} phantoms "
              f"({len(self.sino)*self.spp} samples/epoch), noise={noise}, "
              f"geom={geom_input}")

    def __len__(self):
        return len(self.sino) * self.spp

    def __getitem__(self, idx):
        pi  = idx % len(self.sino)
        rng = np.random.default_rng()
        D   = self.D[pi][:self.ndef[pi]].astype(np.float32)
        sc  = int(self.scout[pi])

        j         = rng.choice(3, p=self.J_PROBS)
        remaining = [a for a in range(ACTION_SIZE) if a != sc]
        if j > 0 and rng.random() < 0.5:
            # quality-biased context, as in sequential deployment
            singles = np.array(remaining, dtype=np.int64)[:, None]
            s1 = score_sets_from_D(D, sc, [], singles)
            z  = (s1 - s1.max()) / (0.3 * (s1.std() + 1e-6))
            p  = np.exp(z); p /= p.sum()
            ctx_extra = list(rng.choice(remaining, size=j, replace=False, p=p))
        else:
            ctx_extra = list(rng.choice(remaining, size=j, replace=False))
        pool      = [a for a in remaining if a not in ctx_extra]
        k         = 3 - j

        combos = np.array(list(itertools.combinations(pool, k)), dtype=np.int64)
        S_all  = score_sets_from_D(D, sc, ctx_extra, combos)

        # list = hard_top best completions + uniform random rest (hard_top=0: purely random)
        n_hard = min(self.hard_top, len(combos), LIST_SIZE // 2)
        hard   = (np.argsort(S_all)[-n_hard:] if n_hard > 0
                  else np.empty(0, dtype=np.int64))
        rest   = np.setdiff1d(np.arange(len(combos)), hard)
        n_rand = min(LIST_SIZE - n_hard, len(rest))
        rnd    = rng.choice(rest, size=n_rand, replace=False)
        sel    = np.concatenate([hard, rnd])
        rng.shuffle(sel)
        cand   = combos[sel]
        labels = S_all[sel]

        ctx_angles = [sc] + ctx_extra
        projs = self.sino[pi][ctx_angles].astype(np.float32)
        if self.noise and rng.random() < 0.5:
            projs = np.stack([add_poisson_noise(p, rng) for p in projs])

        if self.geom_input == "gt":
            geo = self.geom[pi].astype(np.float32)
        else:
            geo = fbp_coarse(projs, ctx_angles)[None]

        ctx_p = np.zeros((MAX_CTX, PROJ_LEN), dtype=np.float32)
        ctx_a = np.full(MAX_CTX, -1, dtype=np.int64)
        ctx_p[:len(ctx_angles)] = projs
        ctx_a[:len(ctx_angles)] = ctx_angles

        cand_pad = np.full((LIST_SIZE, 3), -1, dtype=np.int64)
        lab_pad  = np.full(LIST_SIZE, -1e9, dtype=np.float32)
        cand_pad[:len(cand), :k] = cand
        lab_pad[:len(cand)]      = labels

        return {
            "ctx_p": ctx_p, "ctx_a": ctx_a, "geo": geo,
            "cand": cand_pad, "labels": lab_pad,
            "n_cand": np.int64(len(cand)),
            "j": np.int64(j),
        }


def angle_feats(idx):
    """Sinusoidal angle encoding; padded entries (-1) map to zeros."""
    theta = idx.clamp(min=0).float() * (np.pi / ACTION_SIZE)
    f = torch.stack([torch.sin(theta), torch.cos(theta),
                     torch.sin(2 * theta), torch.cos(2 * theta)], dim=-1)
    return f * (idx >= 0).unsqueeze(-1).float()


class SetVisibilityRanker(nn.Module):
    """Context encoder (1-D CNN on projections + 2-D CNN on coarse FBP)
    and a permutation-invariant set-scoring head."""

    def __init__(self, emb=64, ctx_dim=256, hidden=256, geo_ch=1, d_apf=32,
                 use_apf=False):
        super().__init__()
        self.geo_ch  = geo_ch
        self.d_apf   = d_apf
        self.use_apf = use_apf

        # rotation grids for the optional pseudo-projection features (ablation only);
        # kept as a buffer so that existing checkpoints load unchanged
        th = torch.tensor(ANGLES, dtype=torch.float32)
        rot = torch.zeros(ACTION_SIZE, 2, 3)
        rot[:, 0, 0] =  torch.cos(th); rot[:, 0, 1] = -torch.sin(th)
        rot[:, 1, 0] =  torch.sin(th); rot[:, 1, 1] =  torch.cos(th)
        grids = F.affine_grid(rot, size=(ACTION_SIZE, 1, FBP_SIZE, FBP_SIZE),
                              align_corners=False)
        self.register_buffer("rot_grids", grids)

        self.apf_enc = nn.Sequential(
            nn.Linear(geo_ch * FBP_SIZE, 64), nn.ReLU(),
            nn.Linear(64, d_apf),
        ) if use_apf else None
        self.proj_enc = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Linear(1024, 128), nn.ReLU(),
        )
        self.geo_enc = nn.Sequential(
            nn.Conv2d(geo_ch, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.ctx_mix = nn.Sequential(
            nn.Linear((128 + 4) * 2 + 128, ctx_dim), nn.LayerNorm(ctx_dim), nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(ctx_dim, ctx_dim), nn.LayerNorm(ctx_dim), nn.ReLU(),
        )
        self.angle_emb = nn.Embedding(ACTION_SIZE, emb)
        d_ang = emb + 4 + (d_apf if use_apf else 0)
        self.head = nn.Sequential(
            nn.Linear(ctx_dim + 2 * d_ang, hidden), nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def pseudo_projections(self, geo):
        B, C, H, W = geo.shape
        g = geo.unsqueeze(1).expand(B, ACTION_SIZE, C, H, W)
        g = g.reshape(B * ACTION_SIZE, C, H, W)
        grids = self.rot_grids.repeat(B, 1, 1, 1)
        rot = F.grid_sample(g, grids, align_corners=False, padding_mode="zeros")
        prof = rot.sum(dim=2)
        prof = prof.reshape(B, ACTION_SIZE, C * W)
        return self.apf_enc(prof)

    def encode_ctx(self, ctx_p, ctx_a, geo):
        """Masked mean/max pooling over acquired projections + FBP branch."""
        B, C, L = ctx_p.shape
        e = self.proj_enc(ctx_p.reshape(B * C, 1, L)).reshape(B, C, -1)
        e = torch.cat([e, angle_feats(ctx_a)], dim=-1)
        mask = (ctx_a >= 0).unsqueeze(-1).float()
        e    = e * mask
        mean = e.sum(1) / mask.sum(1).clamp(min=1)
        mx   = (e + (mask - 1) * 1e9).max(1).values
        f    = self.geo_enc(geo)
        ctx_vec = self.ctx_mix(torch.cat([mean, mx, f], dim=-1))
        apf     = self.pseudo_projections(geo) if self.use_apf else None
        return ctx_vec, apf

    def score_sets(self, ctx_vec, apf, cand):
        """Score candidate sets (B, M, k); angle index -1 is padding."""
        B, M, _ = cand.shape
        valid = (cand >= 0).unsqueeze(-1).float()
        idx   = cand.clamp(min=0)
        emb   = self.angle_emb(idx)
        parts = [emb, angle_feats(cand)]
        if apf is not None:
            parts.append(apf[torch.arange(B, device=cand.device)[:, None, None], idx])
        emb   = torch.cat(parts, dim=-1) * valid
        s_sum = emb.sum(2)
        s_max = (emb + (valid - 1) * 1e9).max(2).values
        set_r = torch.cat([s_sum, s_max], dim=-1)
        ctx_e = ctx_vec.unsqueeze(1).expand(-1, M, -1)
        return self.head(torch.cat([ctx_e, set_r], dim=-1)).squeeze(-1)

    def forward(self, ctx_p, ctx_a, geo, cand):
        ctx_vec, apf = self.encode_ctx(ctx_p, ctx_a, geo)
        return self.score_sets(ctx_vec, apf, cand)


def listwise_loss(pred, labels, n_cand):
    """KL between softmax of z-normalised true scores (temperature TAU) and predictions."""
    mask = torch.arange(pred.shape[1], device=pred.device)[None] < n_cand[:, None]
    lab  = labels.masked_fill(~mask, -1e9)
    mu   = (labels * mask).sum(1, keepdim=True) / n_cand[:, None].clamp(min=1)
    sd   = (((labels - mu) ** 2 * mask).sum(1, keepdim=True)
            / n_cand[:, None].clamp(min=1)).sqrt() + 1e-6
    target = F.softmax(((lab - mu) / sd) / TAU, dim=1)
    logp   = F.log_softmax(pred.masked_fill(~mask, -1e9), dim=1)
    return -(target * logp).sum(1).mean()


def run(args):
    tr = RankerDataset(args.train, noise=args.noise, hard_top=args.hard_top,
                       geom_input=args.geom_input)
    va = RankerDataset(args.val, noise=False, samples_per_phantom=4,
                       hard_top=args.hard_top, geom_input=args.geom_input)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    model = SetVisibilityRanker(geo_ch=3 if args.geom_input == "gt" else 1,
                      use_apf=args.apf).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-5)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    log = os.path.join(".", "svr_log.csv")
    with open(log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_regret"])

    best_regret = float("inf")
    for ep in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for b in tqdm(tl, desc=f"Epoch {ep+1}/{args.epochs}", leave=False):
            ctx_p, ctx_a = b["ctx_p"].to(device), b["ctx_a"].to(device)
            geo          = b["geo"].to(device)
            cand, lab    = b["cand"].to(device), b["labels"].to(device)
            nc           = b["n_cand"].to(device)
            opt.zero_grad()
            loss = listwise_loss(model(ctx_p, ctx_a, geo, cand), lab, nc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tl)

        model.eval()
        va_loss = 0.0
        reg_sum = np.zeros(3); reg_cnt = np.zeros(3)
        with torch.no_grad():
            for b in vl:
                ctx_p, ctx_a = b["ctx_p"].to(device), b["ctx_a"].to(device)
                geo          = b["geo"].to(device)
                cand, lab    = b["cand"].to(device), b["labels"].to(device)
                nc, jj       = b["n_cand"].to(device), b["j"].numpy()
                pred = model(ctx_p, ctx_a, geo, cand)
                va_loss += listwise_loss(pred, lab, nc).item()

                # regret@1: score lost by picking the top-ranked set in the list
                mask = torch.arange(pred.shape[1], device=device)[None] < nc[:, None]
                p = pred.masked_fill(~mask, -1e9); l = lab.masked_fill(~mask, -1e9)
                picked = l.gather(1, p.argmax(1, keepdim=True)).squeeze(1)
                r = (1 - picked / l.max(1).values.clamp(min=1e-6)).clamp(min=0)
                r = r.cpu().numpy()
                for phase in range(3):
                    sel = jj == phase
                    reg_sum[phase] += r[sel].sum(); reg_cnt[phase] += sel.sum()
        va_loss /= len(vl)
        va_reg   = reg_sum.sum() / max(reg_cnt.sum(), 1)
        per_j    = reg_sum / np.maximum(reg_cnt, 1)
        sched.step()

        if va_reg < best_regret:
            best_regret = va_reg
            torch.save({"model_state_dict": model.state_dict(),
                        "val_regret": float(va_reg), "epoch": ep,
                        "noise_trained": args.noise,
                        "geom_input": args.geom_input,
                        "use_apf": args.apf},
                       "svr_best.pt")

        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, f"{tr_loss:.5f}",
                                    f"{va_loss:.5f}", f"{va_reg:.5f}"])
        print(f"Epoch {ep+1:3d} | train={tr_loss:.4f} val={va_loss:.4f} "
              f"| val_regret@1={va_reg:.4f} (best {best_regret:.4f}) "
              f"| per-j: j0={per_j[0]:.3f} j1={per_j[1]:.3f} j2={per_j[2]:.3f}")

    print(f"\nDone. Best val_regret@1 = {best_regret:.4f} "
          f"-> svr_best.pt")
    print("A regret of 0.05 means the selected set reaches on average "
          "95 % of the best set in the list.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val",   nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs",     type=int, default=64)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--wd",     type=float, default=1e-3)
    ap.add_argument("--hard-top", type=int, default=0,
                    help="number of best completions mixed into each list; 0 = uniform random lists, as in the paper")
    ap.add_argument("--geom-input", choices=["fbp", "gt"], default="fbp",
                    help="geometry branch input: fbp (deployable) or gt (oracle diagnostic)")
    ap.add_argument("--apf", action="store_true",
                    help="angle-conditioned pseudo-projection features (ablation)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--noise", action="store_true",
                    help="Poisson noise augmentation of the input projections")
    run(ap.parse_args())
