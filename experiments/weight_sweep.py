"""Sweep over the min/mean weight w in S(A) = w * min + (1 - w) * mean.

Efficiency is always measured against the exhaustive optimum under the same
w. Reports the greedy oracle, SVR one-shot and the fixed protocols, the
share of objects where greedy is exactly optimal, and the share of tested
(A, B, theta) triples that violate diminishing returns (submodularity).

The ranker is trained at w = 0.7 only; the sweep tests how robust the
conclusions are to this modelling choice, not a ranker retrained per weight.

    python weight_sweep.py --data dmx_realistic_final.h5 dmx_random_final.h5 dmx_structured_final.h5 \
        --svr svr_ens1.pt svr_ens2.pt svr_ens3.pt

SVR selections are cached to <data>_selections.npz, so --svr is needed only once.
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import itertools
import os

import h5py
import numpy as np
from scipy.stats import wilcoxon

from train_svr import ACTION_SIZE, score_sets_from_D

FIXED = {"+60/120/150": (12, 24, 30),
         "+45/90/135": (9, 18, 27),
         "+30/90/150": (6, 18, 30)}


def score_w(D, scout, ctx_extra, cand_sets, w):
    base = D[:, [scout] + list(ctx_extra)].max(axis=1, keepdims=True)
    best = np.maximum(D[:, cand_sets].max(axis=2), base)
    return w * best.min(axis=0) + (1.0 - w) * best.mean(axis=0)


def greedy_w(D, scout, w, k=3):
    chosen = []
    for _ in range(k):
        pool = [a for a in range(ACTION_SIZE)
                if a != scout and a not in chosen]
        S = score_w(D, scout, chosen, np.array(pool, np.int64)[:, None], w)
        chosen.append(int(pool[int(S.argmax())]))
    return sorted(chosen)


def submodularity_check(D, scout, w, trials, rng):
    """Share of random A subset of B, theta not in B with gain(A) < gain(B)."""
    bad = 0
    for _ in range(trials):
        pool = [a for a in range(ACTION_SIZE) if a != scout]
        rng.shuffle(pool)
        A, B, th = pool[:1], pool[:3], pool[4]
        f = lambda ctx, extra: float(score_w(
            D, scout, ctx, np.array([[extra]], np.int64), w)[0])
        g = lambda ctx: float(score_w(
            D, scout, ctx[:-1], np.array([[ctx[-1]]], np.int64), w)[0])
        dA = f(A, th) - g(A)
        dB = f(B, th) - g(B)
        if dA < dB - 1e-9:
            bad += 1
    return bad / trials


def _selftest():
    """score_w(0.7) must equal score_sets_from_D; w = 0 must be submodular."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        n = int(rng.integers(1, 6))
        D = rng.random((n, ACTION_SIZE)) * 100
        sc = int(rng.integers(0, ACTION_SIZE))
        rem = [a for a in range(ACTION_SIZE) if a != sc]
        cb = np.array(list(itertools.combinations(rem, 3))[:300], dtype=np.int64)
        a = score_w(D, sc, [], cb, 0.7)
        b = score_sets_from_D(D, sc, [], cb)
        worst = max(worst, float(np.max(np.abs(a - b) / np.abs(b))))
    assert worst < 1e-12, f"score_w != score_sets_from_D (rel. {worst:.2e})"

    r = np.random.default_rng(1)
    v0 = v7 = 0.0
    for _ in range(8):
        D = r.random((int(r.integers(2, 6)), ACTION_SIZE)) * 100
        sc = int(r.integers(0, ACTION_SIZE))
        v0 += submodularity_check(D, sc, 0.0, 40, r)
        v7 += submodularity_check(D, sc, 0.7, 40, r)
    assert v0 == 0.0, f"w=0 should be submodular, found {v0:.3f} violations"
    assert v7 > 0.0, "w=0.7 should violate submodularity, but does not"
    return worst, v7 / 8


def combos(sc):
    rem = [a for a in range(ACTION_SIZE) if a != sc]
    return np.array(list(itertools.combinations(rem, 3)), dtype=np.int64)


def get_selections(path, svr_paths):
    cache = path.replace(".h5", "_selections.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"  selections loaded from {cache}")
        return {k: z[k] for k in z.files}
    if not svr_paths:
        raise SystemExit(
            f"{cache} does not exist and --svr was not given. The first run "
            "needs the checkpoints to compute the selections.")
    from paired_stats import load, svr_oneshot
    from train_svr import SetVisibilityRanker
    models = load(SetVisibilityRanker, svr_paths)
    with h5py.File(path, "r") as f:
        sino, scout, ndef = f["sino_all"][:], f["scout_idx"][:], f["n_def"][:]
    sel = {"scout": scout.astype(np.int64),
           "SVR one-shot": np.zeros((len(sino), 3), np.int64)}
    for i in range(len(sino)):
        if int(ndef[i]) == 0:
            continue
        sel["SVR one-shot"][i] = sorted(svr_oneshot(models, sino[i], int(scout[i])))
    for k, off in FIXED.items():
        sel[k] = np.array([sorted((int(s) + o) % ACTION_SIZE for o in off)
                           for s in scout], dtype=np.int64)
    np.savez(cache, **sel)
    print(f"  selections computed and saved to {cache}")
    return sel


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def main(a):
    worst, v7 = _selftest()
    print(f"self-test: score_w(w=0.7) == score_sets_from_D "
          f"(max rel. deviation {worst:.1e}, rounding only)")
    print(f"self-test: submodularity, 0 violations at w=0, "
          f"{v7*100:.1f} % at w=0.7, as predicted\n")

    for path in a.data:
        with h5py.File(path, "r") as f:
            D_all, ndef, scout = f["D"][:], f["n_def"][:], f["scout_idx"][:]
        print(f"{path}")
        sel = get_selections(path, a.svr)
        methods = [k for k in sel if k != "scout"]
        rng = np.random.default_rng(0)

        cols = ["greedy oracle"] + methods
        print(f"  {'w (min)':>8}  " + "".join(f"{m:>16}" for m in cols)
              + f"{'=BF greedy':>13}{'non-submod.':>11}")
        print("  " + "-" * (10 + 16 * len(cols) + 24))
        for w in a.weights:
            effs = {m: [] for m in cols}
            g_exact, viol = [], []
            for i in range(len(D_all)):
                n = int(ndef[i])
                if n == 0:
                    continue
                D = D_all[i][:n].astype(np.float64)
                sc = int(scout[i])
                cb = combos(sc)
                S = score_w(D, sc, [], cb, w)
                best = float(S.max())
                if best <= 0:
                    continue
                for m in methods:
                    s = float(score_w(D, sc, [], sel[m][i][None], w)[0])
                    effs[m].append(s / best)
                gset = greedy_w(D, sc, w)
                gs = float(score_w(D, sc, [], np.array([gset], np.int64), w)[0])
                effs["greedy oracle"].append(gs / best)
                g_exact.append(abs(gs - best) < 1e-9)
                viol.append(submodularity_check(D, sc, w, a.submod_trials, rng))
            effs = {m: np.array(v) for m, v in effs.items()}
            bf = max(FIXED, key=lambda k: effs[k].mean())
            row = f"  {w:>8.2f}  "
            for m in cols:
                row += f"{effs[m].mean()*100:>15.1f}%"
            row += f"{np.mean(g_exact)*100:>12.1f}%{np.mean(viol)*100:>10.1f}%"
            print(row)

            e, r = effs["SVR one-shot"], effs[bf]
            d = (e - r) * 100
            lo, hi = boot(d)
            p = wilcoxon(e, r, alternative="greater").pvalue if np.any(d) else np.nan
            print(f"            SVR − {bf:<14} {d.mean():>+6.1f} pp "
                  f"[{lo:>+5.1f}, {hi:>+5.1f}]  p = {p:.3g}")
        print()

    print("Efficiency is relative to the exhaustive optimum UNDER THE SAME WEIGHT,")
    print("so columns are comparable across weights. '=BF greedy' = share of")
    print("objects where greedy is exactly optimal; 'non-submod.' = share of")
    print("tested (A, B, θ) triples violating diminishing returns. At w = 0 it")
    print("must be zero and greedy has the (1 − 1/e) guarantee.\n")
    print("Weights change only at evaluation; SVR is trained at 0.7/0.3.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--svr", nargs="*", default=[])
    p.add_argument("--weights", nargs="+", type=float,
                   default=[1.0, 0.7, 0.5, 0.0])
    p.add_argument("--submod-trials", type=int, default=40,
                   help="number of (A, B, θ) triples per object tested for submodularity")
    main(p.parse_args())
