"""Angle-wise (pointwise) ablation of the ranker.

Same encoder and training procedure as the set ranker, but the head scores
single angles; the selected set is the top three angles (one-shot) or the
best angle re-planned after each acquisition (--sequential).

    python train_pointwise.py --train dmx_*_train.h5 --val dmx_dev.h5 --noise --out pointwise_best.pt
    python train_pointwise.py --eval --data dmx_realistic_final.h5 --model pointwise_ens1.pt pointwise_ens2.pt pointwise_ens3.pt
    python train_pointwise.py --eval --sequential --data ... --model ...
"""
import argparse
import csv
import itertools

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_svr import (
    ACTION_SIZE, PROJ_LEN, MAX_CTX, device,
    RankerDataset, SetVisibilityRanker, angle_feats, listwise_loss,
    score_sets_from_D, fbp_coarse, add_poisson_noise,
)

N_CAND_MAX = ACTION_SIZE - 1


class PointDataset(RankerDataset):
    """Context sampling as in RankerDataset; labels are S of single angles."""

    def __getitem__(self, idx):
        pi = idx % len(self.sino)
        rng = np.random.default_rng()
        D = self.D[pi][: self.ndef[pi]].astype(np.float32)
        sc = int(self.scout[pi])

        j = rng.choice(3, p=self.J_PROBS)
        remaining = [a for a in range(ACTION_SIZE) if a != sc]
        if j > 0 and rng.random() < 0.5:
            singles = np.array(remaining, dtype=np.int64)[:, None]
            s1 = score_sets_from_D(D, sc, [], singles)
            z = (s1 - s1.max()) / (0.3 * (s1.std() + 1e-6))
            p = np.exp(z); p /= p.sum()
            ctx_extra = list(rng.choice(remaining, size=j, replace=False, p=p))
        else:
            ctx_extra = list(rng.choice(remaining, size=j, replace=False))

        pool = np.array([a for a in remaining if a not in ctx_extra],
                        dtype=np.int64)
        labels = score_sets_from_D(D, sc, ctx_extra, pool[:, None])

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
        ctx_p[: len(ctx_angles)] = projs
        ctx_a[: len(ctx_angles)] = ctx_angles

        cand_pad = np.full(N_CAND_MAX, -1, dtype=np.int64)
        lab_pad = np.full(N_CAND_MAX, -1e9, dtype=np.float32)
        cand_pad[: len(pool)] = pool
        lab_pad[: len(pool)] = labels

        return {"ctx_p": ctx_p, "ctx_a": ctx_a, "geo": geo,
                "cand": cand_pad, "labels": lab_pad,
                "n_cand": np.int64(len(pool)), "j": np.int64(j)}


class PointRanker(SetVisibilityRanker):
    """SetVisibilityRanker with a head that scores one angle at a time."""

    def __init__(self, emb=64, ctx_dim=256, hidden=256, geo_ch=1,
                 d_apf=32, use_apf=False):
        super().__init__(emb=emb, ctx_dim=ctx_dim, hidden=hidden,
                         geo_ch=geo_ch, d_apf=d_apf, use_apf=use_apf)
        d_ang = emb + 4 + (d_apf if use_apf else 0)
        self.head = nn.Sequential(
            nn.Linear(ctx_dim + d_ang, hidden), nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def score_angles(self, ctx_vec, apf, cand):
        B, M = cand.shape
        valid = (cand >= 0).unsqueeze(-1).float()
        idx = cand.clamp(min=0)
        parts = [self.angle_emb(idx), angle_feats(cand)]
        if apf is not None:
            parts.append(apf[torch.arange(B, device=cand.device)[:, None], idx])
        ang = torch.cat(parts, dim=-1) * valid
        ctx_e = ctx_vec.unsqueeze(1).expand(-1, M, -1)
        return self.head(torch.cat([ctx_e, ang], dim=-1)).squeeze(-1)

    def forward(self, ctx_p, ctx_a, geo, cand):
        ctx_vec, apf = self.encode_ctx(ctx_p, ctx_a, geo)
        return self.score_angles(ctx_vec, apf, cand)


def train(args):
    tr = PointDataset(args.train, noise=args.noise, geom_input=args.geom_input)
    va = PointDataset(args.val, noise=False, samples_per_phantom=4,
                      geom_input=args.geom_input)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    model = PointRanker(geo_ch=3 if args.geom_input == "gt" else 1,
                        use_apf=args.apf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    log = "pointwise_log.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_top3"])

    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for b in tqdm(tl, desc=f"Epoch {ep+1}/{args.epochs}", leave=False):
            cp, ca = b["ctx_p"].to(device), b["ctx_a"].to(device)
            geo, cand = b["geo"].to(device), b["cand"].to(device)
            lab, nc = b["labels"].to(device), b["n_cand"].to(device)
            opt.zero_grad()
            loss = listwise_loss(model(cp, ca, geo, cand), lab, nc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tl)

        model.eval()
        va_loss, reg_sum, reg_n = 0.0, 0.0, 0
        with torch.no_grad():
            for b in vl:
                cp, ca = b["ctx_p"].to(device), b["ctx_a"].to(device)
                geo, cand = b["geo"].to(device), b["cand"].to(device)
                lab, nc = b["labels"].to(device), b["n_cand"].to(device)
                pred = model(cp, ca, geo, cand)
                va_loss += listwise_loss(pred, lab, nc).item()

                # top-3 regret: mean label of the predicted top 3 vs the true top 3
                mask = (torch.arange(pred.shape[1], device=device)[None]
                        < nc[:, None])
                p = pred.masked_fill(~mask, -1e9)
                l = lab.masked_fill(~mask, -1e9)
                top = l.gather(1, p.topk(3, dim=1).indices).mean(1)
                ref = l.topk(3, dim=1).values.mean(1)
                reg_sum += (1 - top / ref.clamp(min=1e-6)).clamp(min=0).sum().item()
                reg_n += len(top)
        va_loss /= len(vl)
        va_reg = reg_sum / max(reg_n, 1)
        sched.step()

        if va_reg < best:
            best = va_reg
            torch.save({"model_state_dict": model.state_dict(),
                        "val_regret": float(va_reg), "epoch": ep,
                        "noise_trained": args.noise,
                        "geom_input": args.geom_input,
                        "use_apf": args.apf, "pointwise": True},
                       args.out)

        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, f"{tr_loss:.5f}",
                                    f"{va_loss:.5f}", f"{va_reg:.5f}"])
        print(f"Epoch {ep+1:3d} | train={tr_loss:.4f} val={va_loss:.4f} "
              f"| val_top3_regret={va_reg:.4f} (best {best:.4f})")

    print(f"\nDone. Best val regret = {best:.4f} -> {args.out}")


def bf_optimum(D, scout):
    rem = [a for a in range(ACTION_SIZE) if a != scout]
    combos = np.array(list(itertools.combinations(rem, 3)), dtype=np.int64)
    S = score_sets_from_D(D, scout, [], combos)
    return float(S.max())


def load_models(paths):
    models = []
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        m = PointRanker(geo_ch=3 if ck.get("geom_input") == "gt" else 1,
                        use_apf=ck.get("use_apf", False)).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} model(s)")
    return models


@torch.no_grad()
def _ensemble_scores(models, sino, ctx_angles, pool):
    projs = sino[ctx_angles].astype(np.float32)
    geo = fbp_coarse(projs, ctx_angles)[None]
    ctx_p = np.zeros((MAX_CTX, PROJ_LEN), dtype=np.float32)
    ctx_a = np.full(MAX_CTX, -1, dtype=np.int64)
    ctx_p[: len(ctx_angles)] = projs
    ctx_a[: len(ctx_angles)] = ctx_angles
    t = lambda x: torch.as_tensor(x, device=device).unsqueeze(0)
    zs = []
    for m in models:
        s = m(t(ctx_p), t(ctx_a), t(geo).float(), t(pool))[0]
        zs.append((s - s.mean()) / (s.std() + 1e-6))
    return torch.stack(zs).mean(0).cpu().numpy()


def select_oneshot(models, sino, sc, k=3):
    pool = np.array([a for a in range(ACTION_SIZE) if a != sc], dtype=np.int64)
    s = _ensemble_scores(models, sino, [sc], pool)
    return [int(a) for a in pool[np.argsort(s)[::-1][:k]]]


def select_sequential(models, sino, sc, k=3):
    ctx_angles, chosen = [sc], []
    for _ in range(k):
        pool = np.array([a for a in range(ACTION_SIZE)
                         if a != sc and a not in chosen], dtype=np.int64)
        s = _ensemble_scores(models, sino, ctx_angles, pool)
        best = int(pool[int(s.argmax())])
        chosen.append(best)
        ctx_angles.append(best)
    return chosen


@torch.no_grad()
def evaluate(args):
    models = load_models(args.model)
    select = select_sequential if args.sequential else select_oneshot
    print(f"Mode: {'sequential' if args.sequential else 'one-shot'}")
    for path in args.data:
        with h5py.File(path, "r") as f:
            sino, D_all = f["sino_all"][:], f["D"][:]
            ndef, scout = f["n_def"][:], f["scout_idx"][:]

        effs, exact = [], 0
        for i in range(len(D_all)):
            n = int(ndef[i])
            if n == 0:
                continue
            D = D_all[i][:n].astype(np.float64)
            sc = int(scout[i])
            s_bf = bf_optimum(D, sc)
            if s_bf <= 0:
                continue

            top = select(models, sino[i], sc)
            s_sel = float(score_sets_from_D(
                D, sc, [], np.array([sorted(top)], dtype=np.int64))[0])
            effs.append(s_sel / s_bf)
            if abs(s_sel - s_bf) < 1e-9:
                exact += 1

        e = np.array(effs)
        rng = np.random.default_rng(0)
        bs = rng.choice(e, size=(10000, len(e)), replace=True).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{path:<30} n={len(e):>4}  mean={e.mean()*100:5.1f}%  "
              f"median={np.median(e)*100:5.1f}%  "
              f"CI=[{lo*100:.1f}, {hi*100:.1f}]  "
              f"=BF={exact/len(e)*100:.1f}%  <50%={(e<0.5).mean()*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--train", nargs="+")
    ap.add_argument("--val", nargs="+")
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--model", nargs="+", default=["pointwise_best.pt"])
    ap.add_argument("--out", default="pointwise_best.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--geom-input", choices=["fbp", "gt"], default="fbp")
    ap.add_argument("--apf", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--sequential", action="store_true",
                    help="evaluation: select one angle at a time, extend the "
                         "context and re-run the encoder after each step")
    a = ap.parse_args()
    if a.eval:
        assert a.data, "--eval requires --data"
        evaluate(a)
    else:
        assert a.train and a.val, "training requires --train and --val"
        train(a)
