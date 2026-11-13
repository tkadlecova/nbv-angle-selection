"""Paired statistics per object (Table 1 p-values, one-shot vs sequential).

For each test set: efficiency of the three fixed protocols, SVR one-shot and
sequential, and the angle-wise ablation one-shot and sequential; the paired
difference against the best fixed protocol with a bootstrap CI and a
one-sided paired Wilcoxon test; and one-shot vs sequential for each head
(two-sided paired Wilcoxon).
Also prints the per-object shares of Table 2 (better than the best fixed
protocol, below 50 % of the optimum). Per-object efficiency vectors are
saved to <data>_per_object.npz.

    python paired_stats.py --data dmx_realistic_final.h5 dmx_random_final.h5 dmx_structured_final.h5 \
        --svr svr_ens1.pt svr_ens2.pt svr_ens3.pt \
        --pointwise pointwise_ens1.pt pointwise_ens2.pt pointwise_ens3.pt
"""
import argparse
import itertools
from collections import Counter

import h5py
import numpy as np
import torch
from scipy.stats import wilcoxon

from train_svr import (
    ACTION_SIZE, PROJ_LEN, MAX_CTX, device,
    SetVisibilityRanker, score_sets_from_D, fbp_coarse,
)
from train_pointwise import PointRanker, select_oneshot, select_sequential


FIXED = {"+60/120/150": (12, 24, 30),
         "+45/90/135": (9, 18, 27),
         "+30/90/150": (6, 18, 30)}


def _ctx(sino, angles):
    projs = sino[angles].astype(np.float32)
    geo = fbp_coarse(projs, angles)[None]
    ctx_p = np.zeros((MAX_CTX, PROJ_LEN), dtype=np.float32)
    ctx_a = np.full(MAX_CTX, -1, dtype=np.int64)
    ctx_p[: len(angles)] = projs
    ctx_a[: len(angles)] = angles
    return ctx_p, ctx_a, geo


def load(cls, paths):
    ms = []
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        m = cls(geo_ch=3 if ck.get("geom_input") == "gt" else 1,
                use_apf=ck.get("use_apf", False)).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        ms.append(m)
    return ms


def _score(models, ctx_p, ctx_a, geo, cand):
    t = lambda x: torch.as_tensor(x, device=device).unsqueeze(0)
    zs = []
    for m in models:
        s = m(t(ctx_p), t(ctx_a), t(geo).float(), t(cand))[0]
        zs.append((s - s.mean()) / (s.std() + 1e-6))
    return torch.stack(zs).mean(0).cpu().numpy()


@torch.no_grad()
def svr_oneshot(models, sino, sc):
    rem = [a for a in range(ACTION_SIZE) if a != sc]
    combos = np.array(list(itertools.combinations(rem, 3)), dtype=np.int64)
    ctx_p, ctx_a, geo = _ctx(sino, [sc])
    return [int(x) for x in combos[int(_score(models, ctx_p, ctx_a, geo,
                                              combos).argmax())]]


@torch.no_grad()
def svr_sequential(models, sino, sc, top_consensus=5):
    """Same procedure as eval_ranker.ranker_sequential (6545 -> 561 -> 33 candidates)."""
    acquired = []
    for step in range(3):
        ctx_angles = [sc] + acquired
        ctx_p, ctx_a, geo = _ctx(sino, ctx_angles)
        pool = [a for a in range(ACTION_SIZE) if a not in ctx_angles]
        k = 3 - step
        combos = np.array(list(itertools.combinations(pool, k)), dtype=np.int64)
        pad = np.full((len(combos), 3), -1, dtype=np.int64)
        pad[:, :k] = combos
        pred = _score(models, ctx_p, ctx_a, geo, pad)
        best_set = combos[int(pred.argmax())]
        if k == 1:
            acquired.append(int(best_set[0]))
        else:
            top = combos[np.argsort(pred)[-top_consensus:]]
            counts = Counter(top.ravel())
            acquired.append(int(max(best_set, key=lambda a: counts[a])))
    return acquired


def eff_of(D, sc, angles, s_bf):
    s = float(score_sets_from_D(
        D, sc, [], np.array([sorted(angles)], dtype=np.int64))[0])
    return s / s_bf


def bf(D, sc):
    rem = [a for a in range(ACTION_SIZE) if a != sc]
    combos = np.array(list(itertools.combinations(rem, 3)), dtype=np.int64)
    return float(score_sets_from_D(D, sc, [], combos).max())


def paired_ci(d, n=10000, seed=0):
    """Bootstrap CI of the mean paired difference."""
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def run(args):
    svr = load(SetVisibilityRanker, args.svr) if args.svr else None
    pw = load(PointRanker, args.pointwise) if args.pointwise else None
    print(f"SVR: {len(args.svr) if svr else 0} model(s) | "
          f"pointwise: {len(args.pointwise) if pw else 0} model(s)\n")

    for path in args.data:
        with h5py.File(path, "r") as f:
            sino, D_all = f["sino_all"][:], f["D"][:]
            ndef, scout = f["n_def"][:], f["scout_idx"][:]

        rows = {k: [] for k in list(FIXED)}
        if svr:
            rows["SVR one-shot"] = []
            rows["SVR sequential"] = []
        if pw:
            rows["Pointwise one-shot"] = []
            rows["Pointwise sequential"] = []

        for i in range(len(D_all)):
            n = int(ndef[i])
            if n == 0:
                continue
            D = D_all[i][:n].astype(np.float64)
            sc = int(scout[i])
            s_bf = bf(D, sc)
            if s_bf <= 0:
                continue

            for name, off in FIXED.items():
                ang = [(sc + o) % ACTION_SIZE for o in off]
                rows[name].append(eff_of(D, sc, ang, s_bf))
            if svr:
                rows["SVR one-shot"].append(
                    eff_of(D, sc, svr_oneshot(svr, sino[i], sc), s_bf))
                rows["SVR sequential"].append(
                    eff_of(D, sc, svr_sequential(svr, sino[i], sc), s_bf))
            if pw:
                rows["Pointwise one-shot"].append(
                    eff_of(D, sc, select_oneshot(pw, sino[i], sc), s_bf))
                rows["Pointwise sequential"].append(
                    eff_of(D, sc, select_sequential(pw, sino[i], sc), s_bf))

        rows = {k: np.array(v) for k, v in rows.items()}
        best_name = max(FIXED, key=lambda k: rows[k].mean())
        ref = rows[best_name]

        print(f"{path}   (n = {len(ref)})   best fixed = {best_name} "
              f"({ref.mean()*100:.1f} %)")
        print(f"  {'method':<22} {'mean':>7} {'=BF':>7} "
              f"{'Δ vs fixed':>11} {'95% CI (paired)':>18} {'p':>10}")
        print("  " + "-" * 80)
        for name, e in rows.items():
            d = e - ref
            exact = (e > 1 - 1e-9).mean() * 100
            if name == best_name:
                print(f"  {name + ' (ref)':<22} {e.mean()*100:>6.1f}% "
                      f"{exact:>6.1f}% {'—':>11} {'—':>18} {'—':>10}")
                continue
            lo, hi = paired_ci(d)
            p = (wilcoxon(e, ref, alternative="greater").pvalue
                 if np.any(d != 0) else float("nan"))
            print(f"  {name:<22} {e.mean()*100:>6.1f}% {exact:>6.1f}% "
                  f"{d.mean()*100:>+10.1f} [{lo*100:>+6.1f}, {hi*100:>+6.1f}] "
                  f"{p:>10.2g}")
        print()

        pairs = [("SVR one-shot", "SVR sequential"),
                 ("Pointwise one-shot", "Pointwise sequential")]
        shown = False
        for a, b in pairs:
            if a not in rows or b not in rows:
                continue
            if not shown:
                print(f"  {'one-shot vs sequential':<22} {'':>7} {'':>7} "
                      f"{'Δ':>11} {'95% CI (paired)':>18} {'p':>10}")
                print("  " + "-" * 80)
                shown = True
            d = rows[a] - rows[b]
            lo, hi = paired_ci(d)
            p = (wilcoxon(rows[a], rows[b]).pvalue
                 if np.any(d != 0) else float("nan"))
            print(f"  {a.split()[0] + ': 1-shot vs seq':<22} {'':>7} {'':>7} "
                  f"{d.mean()*100:>+10.1f} [{lo*100:>+6.1f}, {hi*100:>+6.1f}] "
                  f"{p:>10.2g}")
        print()

        # Table 2: per-object success against the best fixed protocol
        print(f"  {'per object':<22} {'> best fixed':>13} {'< 50 % of opt.':>15}")
        print("  " + "-" * 52)
        for name, e in rows.items():
            above = "—" if name == best_name else f"{(e > ref).mean()*100:.0f} %"
            print(f"  {name:<22} {above:>13} {(e < 0.5).mean()*100:>13.0f} %")
        print()

        out = path.replace(".h5", "_per_object.npz")
        np.savez(out, best_fixed=best_name, **rows)
        print(f"  per-object vectors -> {out}\n")

    print("p vs fixed = one-sided paired Wilcoxon (method > best fixed);")
    print("p one-shot vs sequential = two-sided paired Wilcoxon.")
    print("CI = bootstrap of the PAIRED difference, not of the method mean.")
    print("=BF = share of objects where the method hits the exhaustive optimum exactly.")
    print("> best fixed = share of objects where the method is strictly better (Table 2).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--svr", nargs="*", default=[])
    ap.add_argument("--pointwise", nargs="*", default=[])
    run(ap.parse_args())
