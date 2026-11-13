"""Oracle ladder on the true V matrix (no learning involved).

  BF         exhaustive optimum over C(35, 3) = 6,545 sets (= 100 %)
  greedy     sequential greedy selection, each step conditioned on the previous ones
  pointwise  the three best angles by their single-angle score S({a}), unconditioned

The pointwise oracle is an upper bound for any angle-wise model, since it
receives the exact per-angle values for free.

    python greedy_oracle.py --data dmx_structured_final.h5 dmx_realistic_final.h5 dmx_random_final.h5
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import itertools
import numpy as np
import h5py

from train_svr import score_sets_from_D, ACTION_SIZE


def _pool(scout, chosen=()):
    return [a for a in range(ACTION_SIZE) if a != scout and a not in chosen]


def score_of(D, scout, angles):
    cand = np.array([list(angles)], dtype=np.int64)
    return float(score_sets_from_D(D, scout, [], cand)[0])


def bf_optimum(D, scout):
    combos = np.array(list(itertools.combinations(_pool(scout), 3)),
                      dtype=np.int64)
    S = score_sets_from_D(D, scout, [], combos)
    i = int(S.argmax())
    return float(S[i]), tuple(int(x) for x in combos[i])


def greedy_oracle(D, scout, k=3):
    chosen = []
    for _ in range(k):
        pool = _pool(scout, chosen)
        singles = np.array(pool, dtype=np.int64)[:, None]
        S = score_sets_from_D(D, scout, chosen, singles)
        chosen.append(int(pool[int(S.argmax())]))
    return score_of(D, scout, chosen), tuple(chosen)


def pointwise_oracle(D, scout, k=3):
    pool = _pool(scout)
    singles = np.array(pool, dtype=np.int64)[:, None]
    s1 = score_sets_from_D(D, scout, [], singles)
    top = [int(pool[i]) for i in np.argsort(s1)[::-1][:k]]
    return score_of(D, scout, top), tuple(top)


def boot_ci(x, n=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])


METHODS = [("greedy", greedy_oracle), ("pointwise", pointwise_oracle)]


def run(paths):
    for p in paths:
        with h5py.File(p, "r") as f:
            D_all, ndef, scout = f["D"][:], f["n_def"][:], f["scout_idx"][:]

        acc = {name: {"eff": [], "exact": 0} for name, _ in METHODS}
        n_used = 0
        for i in range(len(D_all)):
            n = int(ndef[i])
            if n == 0:
                continue
            D = D_all[i][:n].astype(np.float64)
            sc = int(scout[i])
            s_bf, set_bf = bf_optimum(D, sc)
            if s_bf <= 0:
                continue
            n_used += 1
            for name, fn in METHODS:
                s, st = fn(D, sc)
                acc[name]["eff"].append(s / s_bf)
                if tuple(sorted(st)) == tuple(sorted(set_bf)):
                    acc[name]["exact"] += 1

        print(f"\n{p}   (n = {n_used})")
        print(f"  {'method':<12} {'mean':>8} {'median':>8} {'95% CI':>16} "
              f"{'=BF':>7} {'<50%':>7}")
        print("  " + "-" * 62)
        print(f"  {'BF optimum':<12} {'100.0%':>8} {'100.0%':>8} "
              f"{'—':>16} {'100.0%':>7} {'0.0%':>7}")
        for name, _ in METHODS:
            e = np.array(acc[name]["eff"])
            lo, hi = boot_ci(e)
            print(f"  {name:<12} {e.mean()*100:>7.1f}% {np.median(e)*100:>7.1f}% "
                  f"[{lo*100:>5.1f}, {hi*100:>5.1f}] "
                  f"{acc[name]['exact']/len(e)*100:>6.1f}% "
                  f"{(e < 0.5).mean()*100:>6.1f}%")

    print("\n'=BF'  = share of objects with exactly the optimal set")
    print("'<50%' = share of objects below half of the optimum")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    run(ap.parse_args().data)
