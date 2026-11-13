"""Evaluate the ranker against the exhaustive optimum (Table 1, noise table).

Efficiency = S(selected set) / S(exhaustive optimum over all 6,545 sets).
Reports one-shot and sequential SVR, the three scout-relative fixed
protocols and random selection (mean over 200 draws), with bootstrap CIs and
a one-sided paired Wilcoxon test against the best fixed protocol.

    python eval_ranker.py --data dmx_realistic_final.h5 dmx_random_final.h5 \
        dmx_structured_final.h5 --model svr_ens1.pt svr_ens2.pt svr_ens3.pt

    # Poisson noise on the ranker inputs only (the reference stays noiseless)
    python eval_ranker.py --data ... --model ... --noise --i0 1e3
"""
import argparse
import itertools
from collections import Counter
import numpy as np
import h5py
import torch
from scipy import stats

from train_svr import (SetVisibilityRanker, ACTION_SIZE, PROJ_LEN, MAX_CTX,
                              fbp_coarse)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# offsets in 5-degree steps relative to the scout
FIXED_OFFSETS = {
    "scout+60+120+150": (12, 24, 30),
    "scout+45+90+135":  (9, 18, 27),
    "scout+30+90+150":  (6, 18, 30),
}


def score_sets(D, scout, ctx_extra, cand_sets):
    base = D[:, [scout] + list(ctx_extra)].max(axis=1, keepdims=True)
    vals = D[:, cand_sets]
    best = np.maximum(vals.max(axis=2), base)
    return 0.7 * best.min(axis=0) + 0.3 * best.mean(axis=0)


def bf_optimum(D, scout):
    """Exhaustive optimum over the C(35, 3) = 6,545 sets that exclude the scout."""
    remaining = [a for a in range(ACTION_SIZE) if a != scout]
    combos    = np.array(list(itertools.combinations(remaining, 3)))
    S         = score_sets(D, scout, [], combos)
    return float(S.max()), combos[int(S.argmax())]


def make_ctx(sino, angles, geom_gt=None, noise_rng=None, i0=None):
    projs = sino[list(angles)].astype(np.float32)
    if noise_rng is not None:
        projs = np.stack([
            noise_rng.poisson(np.clip(p, 0, 1) * i0).astype(np.float32) / i0
            for p in projs
        ])
    if geom_gt is not None:
        geo = geom_gt.astype(np.float32)
    else:
        geo = fbp_coarse(projs, list(angles))[None]
    ctx_p = np.zeros((MAX_CTX, PROJ_LEN), dtype=np.float32)
    ctx_a = np.full(MAX_CTX, -1, dtype=np.int64)
    ctx_p[:len(angles)] = projs
    ctx_a[:len(angles)] = list(angles)
    return (torch.tensor(ctx_p[None]).to(device),
            torch.tensor(ctx_a[None]).to(device),
            torch.tensor(geo[None]).to(device))


@torch.no_grad()
def rank_completions(models, ctx_p, ctx_a, geo, pool, k):
    """Score all k-completions; ensemble = mean of z-normalised scores."""
    combos = np.array(list(itertools.combinations(pool, k)))
    pad    = np.full((len(combos), 3), -1, dtype=np.int64)
    pad[:, :k] = combos
    cand_t = torch.tensor(pad[None]).to(device)
    acc = np.zeros(len(combos), dtype=np.float64)
    for m in models:
        ctx_vec, apf = m.encode_ctx(ctx_p, ctx_a, geo)
        pred = m.score_sets(ctx_vec, apf, cand_t)[0].cpu().numpy()
        acc += (pred - pred.mean()) / (pred.std() + 1e-9)
    return combos, acc / len(models)


@torch.no_grad()
def ranker_one_shot(models, sino, scout, geom_gt=None, noise_rng=None, i0=None):
    ctx_p, ctx_a, geo = make_ctx(sino, [scout], geom_gt, noise_rng, i0)
    pool = [a for a in range(ACTION_SIZE) if a != scout]
    combos, pred = rank_completions(models, ctx_p, ctx_a, geo, pool, 3)
    return tuple(combos[int(pred.argmax())])


@torch.no_grad()
def ranker_sequential(models, sino, scout, geom_gt=None, top_consensus=5,
                      noise_rng=None, i0=None):
    """Commit one angle per step: from the current best plan, acquire the angle
    most frequent among the top_consensus completions, then re-plan."""
    acquired = []
    for step in range(3):
        ctx_angles   = [scout] + acquired
        ctx_p, ctx_a, geo = make_ctx(sino, ctx_angles, geom_gt, noise_rng, i0)
        pool    = [a for a in range(ACTION_SIZE) if a not in ctx_angles]
        k       = 3 - step
        combos, pred = rank_completions(models, ctx_p, ctx_a, geo, pool, k)
        best_set = combos[int(pred.argmax())]
        if k == 1:
            acquired.append(int(best_set[0]))
        else:
            top    = combos[np.argsort(pred)[-top_consensus:]]
            counts = Counter(top.ravel())
            acquired.append(int(max(best_set, key=lambda a: counts[a])))
    return tuple(acquired)


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    rng  = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def load_models(paths):
    models, geom_input = [], None
    for p in paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        gi = ckpt.get("geom_input", "fbp")
        if geom_input is None:
            geom_input = gi
        elif gi != geom_input:
            raise ValueError(f"{p}: geom_input={gi}, but previous models "
                             f"use {geom_input}; an ensemble must be homogeneous")
        m = SetVisibilityRanker(geo_ch=3 if gi == "gt" else 1,
                      use_apf=ckpt.get(
                          "use_apf",
                          any("apf_enc" in k for k in ckpt["model_state_dict"]))
                      ).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
        print(f"  {p}: val_regret@1={ckpt.get('val_regret', float('nan')):.4f}, "
              f"noise_trained={ckpt.get('noise_trained', False)}, geom={gi}")
    return models, geom_input


def evaluate(args):
    print(f"Ensemble: {len(args.model)} model(s)")
    models, geom_input = load_models(args.model)
    if geom_input == "gt":
        print("WARNING: oracle-geometry diagnostic, not a deployable method!")

    rows = []
    for path in args.data:
        with h5py.File(path, "r") as f:
            sino  = f["sino_all"][:]
            Dall  = f["D"][:]
            ndef  = f["n_def"][:]
            scout = f["scout_idx"][:]
            dist  = f.attrs.get("dist", "?")
            if geom_input == "gt":
                if "geom_gt" not in f:
                    raise KeyError(f"{path} has no 'geom_gt'; "
                                   "regenerate it with the current generator")
                geoms = f["geom_gt"][:]
            else:
                geoms = None

        n = len(sino)
        eff = {m: np.zeros(n) for m in
               ["one_shot", "sequential", *FIXED_OFFSETS, "random"]}
        rng_noise = np.random.default_rng(args.seed) if args.noise else None
        rng_rand  = np.random.default_rng(123)

        for i in range(n):
            D  = Dall[i][:ndef[i]].astype(np.float32)
            sc = int(scout[i])
            s  = sino[i]
            gg = geoms[i] if geoms is not None else None
            bf, _ = bf_optimum(D, sc)

            def S(triple):
                return score_sets(D, sc, [], np.array([triple]))[0]

            eff["one_shot"][i] = S(ranker_one_shot(
                models, s, sc, gg, rng_noise, args.i0)) / bf
            eff["sequential"][i] = S(ranker_sequential(
                models, s, sc, gg, noise_rng=rng_noise, i0=args.i0)) / bf

            for name, off in FIXED_OFFSETS.items():
                triple = tuple((sc + o) % ACTION_SIZE for o in off)
                eff[name][i] = S(triple) / bf

            rand_scores = [
                S(tuple(rng_rand.choice(
                    [a for a in range(ACTION_SIZE) if a != sc],
                    size=3, replace=False)))
                for _ in range(200)
            ]
            eff["random"][i] = np.mean(rand_scores) / bf

        best_fixed = max(FIXED_OFFSETS, key=lambda k: eff[k].mean())
        w_one = stats.wilcoxon(eff["one_shot"],   eff[best_fixed],
                               alternative="greater")
        w_seq = stats.wilcoxon(eff["sequential"], eff[best_fixed],
                               alternative="greater")

        print(f"\n{'='*66}\nDataset: {path}  |  dist={dist}  |  n={n}"
              + (f"  |  NOISE I0={args.i0:.0e}" if args.noise else ""))
        print(f"{'Method':<24}{'Mean eff':>10}{'Std':>8}{'95% CI':>20}")
        print("-" * 66)
        for m in ["one_shot", "sequential", best_fixed, "random"]:
            lo, hi = bootstrap_ci(eff[m])
            tag = " <- ranker" if m in ("one_shot", "sequential") else ""
            print(f"{m:<24}{eff[m].mean()*100:>9.1f}%{eff[m].std()*100:>7.1f}%"
                  f"   [{lo*100:.1f}%, {hi*100:.1f}%]{tag}")
        print(f"\nWilcoxon (greater) vs {best_fixed}:")
        print(f"  one-shot:   p = {w_one.pvalue:.4g}")
        print(f"  sequential: p = {w_seq.pvalue:.4g}")

        for m, e in eff.items():
            rows.append((path, dist, m, e.mean(), e.std(), n))

    with open(args.out, "w") as f:
        f.write("dataset\tdist\tmethod\tmean_eff\tstd\tn\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\t{r[4]:.4f}\t{r[5]}\n")
    print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="held-out V-matrix h5 files (seed offset disjoint from training)")
    ap.add_argument("--model", nargs="+", default=["svr_best.pt"],
                    help="one or more checkpoints; several = ensemble (z-normalised mean)")
    ap.add_argument("--noise", action="store_true",
                    help="Poisson noise on the ranker inputs (reference stays noiseless)")
    ap.add_argument("--i0", type=float, default=1e4,
                    help="photon flux for the noise experiment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_ranker_results.tsv")
    evaluate(ap.parse_args())
