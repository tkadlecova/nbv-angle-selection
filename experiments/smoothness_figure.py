"""Flatness of the objective near its optimum (Figure 6, Table 3).

(a) score retained when every angle of the exhaustive-optimal set is shifted
    by a given amount with a random sign;
(b) share of the 6,545 candidate sets reaching a given percentage of the optimum.

    python smoothness_figure.py
    python smoothness_figure.py --max-shift 45 --n-phantoms 150
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import itertools

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_ranker as E

TEST_FILES = {
    "structured": "dmx_structured_final.h5",
    "realistic":  "dmx_realistic_final.h5",
    "random":     "dmx_random_final.h5",
}
LABEL = {"structured": "Structured", "realistic": "Realistic", "random": "Random"}
COL = {"structured": "#2E7D32", "realistic": "#1F4E79", "random": "#B85C00"}
STEP_DEG = 5

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
})


def retention_curve(D, scout, bf_combo, bf_score, shifts, rng):
    n_ang = D.shape[1]
    out = []
    for sh in shifts:
        if sh == 0:
            out.append(100.0)
            continue
        vals = []
        for _ in range(12):
            signs = rng.choice([-1, 1], size=3)
            cand = [(a + s * sh // STEP_DEG) % n_ang
                    for a, s in zip(bf_combo, signs)]
            if len(set(cand)) < 3 or scout in cand:
                continue
            s = float(E.score_sets(D, scout, [], np.array([cand]))[0])
            vals.append(100.0 * s / bf_score)
        out.append(float(np.mean(vals)) if vals else np.nan)
    return out


def threshold_curve(D, scout, bf_score, thresholds):
    n_ang = D.shape[1]
    rem = [a for a in range(n_ang) if a != scout]
    combos = np.array(list(itertools.combinations(rem, 3)))
    S = E.score_sets(D, scout, [], combos)
    frac = 100.0 * S / bf_score
    return [100.0 * float((frac >= t).mean()) for t in thresholds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shift", type=int, default=40)
    ap.add_argument("--n-phantoms", type=int, default=None)
    ap.add_argument("--out", default="fig3_smoothness_simple")
    args = ap.parse_args()

    shifts = list(range(0, args.max_shift + 1, STEP_DEG))
    thresholds = np.arange(80, 100.5, 1.0)
    rng = np.random.default_rng(0)

    ret, thr, stats = {}, {}, {}

    for dist, path in TEST_FILES.items():
        with h5py.File(path, "r") as f:
            n = f["D"].shape[0] if args.n_phantoms is None else min(
                args.n_phantoms, f["D"].shape[0])
            D_all = f["D"][:n].astype(np.float64)
            n_def = f["n_def"][:n]
            scouts = f["scout_idx"][:n]

        R, T = [], []
        for i in range(n):
            D = D_all[i][: int(n_def[i])]
            scout = int(scouts[i])
            bf_score, bf_combo = E.bf_optimum(D, scout)
            if bf_score <= 0:
                continue
            R.append(retention_curve(D, scout, list(bf_combo), bf_score,
                                     shifts, rng))
            T.append(threshold_curve(D, scout, bf_score, thresholds))
        ret[dist] = np.nanmean(np.array(R), axis=0)
        thr[dist] = np.mean(np.array(T), axis=0)

        i15 = shifts.index(15) if 15 in shifts else None
        stats[dist] = {
            "ret15": ret[dist][i15] if i15 is not None else np.nan,
            "frac95": thr[dist][int(np.argmin(np.abs(thresholds - 95)))],
        }
        print(f"{dist}: retained at ±15° = {stats[dist]['ret15']:.1f} % | "
              f"sets above 95 % of optimum = {stats[dist]['frac95']:.1f} %")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    for dist in TEST_FILES:
        ax.plot(shifts, ret[dist], "-o", color=COL[dist], lw=1.5, ms=3.5,
                label=LABEL[dist])

    ax.set_xlabel("Angular error from the optimal set [°]")
    ax.set_ylabel("Score retained [% of optimum]")
    ax.set_xlim(0, args.max_shift)
    ax.set_ylim(50, 102)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    for dist in TEST_FILES:
        ax.plot(thresholds, thr[dist], "-", color=COL[dist], lw=1.5,
                label=LABEL[dist])

    ax.set_xlabel("Threshold [% of optimum]")
    ax.set_ylabel("Candidate sets above threshold [%]")
    ax.set_xlim(80, 100)
    ax.set_ylim(0, None)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {args.out}.png / .pdf")

    print("\nSummary:")
    for dist in TEST_FILES:
        s = stats[dist]
        print(f"  {LABEL[dist]}: {s['ret15']:.0f} % score retained at ±15°, "
              f"{s['frac95']:.1f} % of sets within 5 % of the optimum")


if __name__ == "__main__":
    main()
