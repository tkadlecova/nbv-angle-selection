"""Frozen fixed protocol: the best of all 6,545 scout-relative offset triples,
selected on the development set and frozen before the test set.

Reports its test efficiency against the hand-chosen protocols and SVR one-shot
(paired bootstrap CI, one-sided Wilcoxon), and how stable the ranking of
protocols is (dev half vs dev half, dev vs test, top-100 overlap).

    python frozen_fixed.py --dev dmx_dev.h5 --test dmx_realistic_final.h5 dmx_random_final.h5 \
        dmx_structured_final.h5 --svr svr_ens1.pt svr_ens2.pt svr_ens3.pt
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import itertools
import os

import h5py
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from train_svr import ACTION_SIZE, score_sets_from_D

HAND = {"+60/120/150": (12, 24, 30),
        "+45/90/135": (9, 18, 27),
        "+30/90/150": (6, 18, 30)}

# all offset triples 1..35 relative to the scout (5-degree steps)
OFFSETS = np.array(list(itertools.combinations(range(1, ACTION_SIZE), 3)),
                   dtype=np.int64)


def eff_all(D, scout):
    ang = (scout + OFFSETS) % ACTION_SIZE
    S = score_sets_from_D(D, scout, [], ang)
    m = float(S.max())
    return (S / m) if m > 0 else None


def load(path, n=None):
    with h5py.File(path, "r") as f:
        return (f["D"][:n], f["n_def"][:n], f["scout_idx"][:n],
                f["sino_all"][:n] if "sino_all" in f else None)


def all_eff(path, n=None):
    D_all, ndef, scout, _ = load(path, n)
    rows = []
    for i in range(len(D_all)):
        k = int(ndef[i])
        if k == 0:
            continue
        e = eff_all(D_all[i][:k].astype(np.float64), int(scout[i]))
        if e is not None:
            rows.append(e)
    return np.array(rows)


def mean_eff(path, n=None):
    E = all_eff(path, n)
    return E.mean(axis=0), len(E)


def svr_sel(path, svr_paths):
    """SVR one-shot selections, cached in <data>_selections.npz."""
    cache = path.replace(".h5", "_selections.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if "SVR one-shot" in z.files:
            return z["SVR one-shot"]
    if not svr_paths:
        return None
    from paired_stats import load as mload, svr_oneshot
    from train_svr import SetVisibilityRanker
    models = mload(SetVisibilityRanker, svr_paths)
    D_all, ndef, scout, sino = load(path)
    out = np.zeros((len(D_all), 3), np.int64)
    for i in range(len(D_all)):
        if int(ndef[i]) == 0:
            continue
        out[i] = sorted(svr_oneshot(models, sino[i], int(scout[i])))
    np.savez(cache, **{"SVR one-shot": out, "scout": scout})
    return out


def paired(x, y):
    d = x - y
    rng = np.random.default_rng(0)
    bs = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = wilcoxon(x, y, alternative="greater").pvalue if np.any(d) else np.nan
    return d.mean() * 100, lo * 100, hi * 100, p


def main(a):
    dev = a.dev if len(a.dev) == len(a.test) else a.dev * len(a.test)
    assert len(dev) == len(a.test), "--dev must have one file or as many as --test"

    for dpath, tpath in zip(dev, a.test):
        md, ndev = mean_eff(dpath, a.dev_n)
        best = int(md.argmax())
        frozen = tuple(int(x) for x in OFFSETS[best])
        deg = [o * 5 for o in frozen]
        print(f"\n{'='*70}")
        print(f"dev  {dpath}  ({ndev} objects)")
        print(f"  frozen protocol: scout +{deg[0]}/{deg[1]}/{deg[2]}°  "
              f"(dev efficiency {md[best]*100:.1f} %)")
        for name, off in HAND.items():
            j = int(np.where((OFFSETS == np.array(off)).all(axis=1))[0][0])
            print(f"    for comparison {name:<14} dev {md[j]*100:>5.1f} %  "
                  f"(rank {int((md > md[j]).sum()) + 1} of {len(OFFSETS)})")

        D_all, ndef, scout, _ = load(tpath, a.test_n)
        sel = svr_sel(tpath, a.svr)
        rows = {f"frozen +{deg[0]}/{deg[1]}/{deg[2]}": [], **{k: [] for k in HAND}}
        if sel is not None:
            rows["SVR one-shot"] = []
        rank_frozen = []
        for i in range(len(D_all)):
            k = int(ndef[i])
            if k == 0:
                continue
            D = D_all[i][:k].astype(np.float64)
            sc = int(scout[i])
            e = eff_all(D, sc)
            if e is None:
                continue
            idx = {tuple(o): j for j, o in enumerate(OFFSETS.tolist())}
            rows[f"frozen +{deg[0]}/{deg[1]}/{deg[2]}"].append(e[best])
            rank_frozen.append(float((e > e[best]).mean() * 100))
            for name, off in HAND.items():
                rows[name].append(e[idx[off]])
            if sel is not None:
                ang = np.array([sorted(sel[i])], dtype=np.int64)
                S = score_sets_from_D(D, sc, [], ang)
                Sall = score_sets_from_D(D, sc, [], (sc + OFFSETS) % ACTION_SIZE)
                rows["SVR one-shot"].append(float(S[0]) / float(Sall.max()))

        rows = {k: np.array(v) for k, v in rows.items()}
        fz = f"frozen +{deg[0]}/{deg[1]}/{deg[2]}"
        print(f"\ntest {tpath}  ({len(rows[fz])} objects)")
        print(f"  {'protocol':<26} {'efficiency':>11} {'vs frozen':>10} "
              f"{'95% CI':>16} {'p':>9}")
        print("  " + "-" * 76)
        ref = rows[fz]
        for k, v in rows.items():
            if k == fz:
                print(f"  {k + ' (ref)':<26} {v.mean()*100:>10.1f}% "
                      f"{'—':>10} {'—':>16} {'—':>9}")
                continue
            d, lo, hi, p = paired(v, ref)
            print(f"  {k:<26} {v.mean()*100:>10.1f}% {d:>+9.1f} "
                  f"[{lo:>+6.1f}, {hi:>+6.1f}] {p:>9.3g}")
        print(f"  on the test set the frozen protocol is on average above "
              f"{100 - np.mean(rank_frozen):.1f} % of all 6545 protocols")

        # how reproducible is the ranking of the 6545 protocols?
        Ed = all_eff(dpath, a.dev_n)
        Et = all_eff(tpath, a.test_n)
        h = len(Ed) // 2
        r_split = spearmanr(Ed[:h].mean(0), Ed[h:].mean(0)).statistic
        r_cross = spearmanr(Ed.mean(0), Et.mean(0)).statistic
        top = np.argsort(Ed.mean(0))[::-1][:100]
        overlap = len(set(top.tolist())
                      & set(np.argsort(Et.mean(0))[::-1][:100].tolist()))
        print(f"\n  stability of the protocol ranking:")
        print(f"    dev half vs dev half : rho = {r_split:+.3f}")
        print(f"    dev vs test          : rho = {r_cross:+.3f}")
        print(f"    top-100 overlap      : {overlap}/100")
        print(f"    range of mean efficiency across protocols (dev): "
              f"{Ed.mean(0).min()*100:.1f}–{Ed.mean(0).max()*100:.1f} %")
        if r_split < 0.5:
            print("    => ranking is unstable even within the dev set: noise")
            print("       dominates, the population landscape is flat")
        elif r_cross > 0.7:
            print("    => ranking is stable across sets, yet the frozen protocol")
            print("       does not win: the optimum is object-specific")
        else:
            print("    => ranking holds within the dev set but not across sets;")
            print("       both explanations contribute")

    print("\nThe protocol was selected on the development set and frozen before")
    print("the test set, so the comparison is not favoured by test-set selection.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dev", nargs="+", required=True)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--svr", nargs="*", default=[])
    p.add_argument("--dev-n", type=int, default=None)
    p.add_argument("--test-n", type=int, default=None)
    main(p.parse_args())
