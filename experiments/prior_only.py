"""Prior-only ablation: choose the three angles WITHOUT looking at the image.

A lookup table maps a bin of the scout angle to a triple of offsets relative
to the scout, fitted by averaging S(A) over a set of objects (no training).
With 1 bin this is the frozen fixed protocol; with 36 bins it is a free table
scout angle -> triple. Because the fit improves monotonically with the number
of bins, the resolution is selected on a separate development set:

  --fit   set used to fit the tables (the training set, i.e. the same data SVR saw)
  --dev   set used to select the number of bins
  --test  set used to report the result

The table is indexed by the scout angle in the object's own frame, which is
not observable in practice (the pose of an incoming part is unknown), so it is
a partial oracle rather than a deployable protocol. If SVR beats it, SVR
extracts object-specific information from the scout image.

    python prior_only.py --fit dmx_structured_train.h5 --dev dmx_structured_test.h5 \
        --test dmx_structured_final.h5 --paired dmx_structured_final_per_object.npz \
        --key "SVR one-shot"
"""
import argparse
import itertools

import h5py
import numpy as np
from scipy.stats import wilcoxon

ACTION_SIZE = 36
STEP_DEG = 180.0 / ACTION_SIZE
W_MIN = 0.7
BINS = [1, 2, 3, 4, 6, 9, 12, 18, 36]          # divisors of ACTION_SIZE


def load(path, n=None):
    with h5py.File(path, "r") as f:
        D = f["D"][:n].astype(np.float64)
        nd = f["n_def"][:n].astype(int)
        sc = np.asarray(f["scout_idx"][:n]).astype(int).ravel()
    keep = nd > 0
    if not keep.all():
        print(f"  {path}: skipped {int((~keep).sum())} objects without defects")
    return D[keep], nd[keep], sc[keep]


def _cand(s):
    other = [a for a in range(ACTION_SIZE) if a != s]
    return np.array(list(itertools.combinations(other, 3)), dtype=np.int64)


CAND = {s: _cand(s) for s in range(ACTION_SIZE)}
NCAND = CAND[0].shape[0]


def _rel_perm(s):
    """Permutation sorting the candidates of scout s by their relative offsets,
    so that column j means the same offset triple for every scout."""
    off = np.sort((CAND[s] - s) % ACTION_SIZE, axis=1)
    key = off[:, 0] * ACTION_SIZE**2 + off[:, 1] * ACTION_SIZE + off[:, 2]
    return np.argsort(key)


REL = {s: _rel_perm(s) for s in range(ACTION_SIZE)}
REL_OFF = np.sort((CAND[0] - 0) % ACTION_SIZE, axis=1)[REL[0]]


def score_rel(Dk, scout):
    """S(A) for all 6545 candidates, in the common relative order."""
    base = Dk[:, [scout]].max(axis=1, keepdims=True)
    vals = Dk[:, CAND[scout]]
    best = np.maximum(vals.max(axis=2), base)
    s = W_MIN * best.min(axis=0) + (1.0 - W_MIN) * best.mean(axis=0)
    return s[REL[scout]]


def accumulate(D, nd, sc, norm):
    """Sum of (normalised) S per scout angle; any binning is a sum of rows."""
    acc = np.zeros((ACTION_SIZE, NCAND))
    cnt = np.zeros(ACTION_SIZE, dtype=int)
    for i in range(len(D)):
        r = score_rel(D[i][:nd[i]], sc[i])
        acc[sc[i]] += r / r.max() if norm else r
        cnt[sc[i]] += 1
    return acc, cnt


def fit_bins(acc, cnt, k, min_n):
    """Best column per bin; bins with fewer than min_n objects fall back to 1 bin."""
    w = ACTION_SIZE // k
    glob = int(np.argmax(acc.sum(axis=0)))
    out, ns = [], []
    for b in range(k):
        sl = slice(b * w, (b + 1) * w)
        n = int(cnt[sl].sum())
        ns.append(n)
        out.append(int(np.argmax(acc[sl].sum(axis=0))) if n >= min_n else None)
    return out, ns, glob


def effs_all(D, nd, sc, specs):
    out = {k: np.empty(len(D)) for k, _, _ in specs}
    for i in range(len(D)):
        r = score_rel(D[i][:nd[i]], sc[i])
        m = r.max()
        for k, tab, glob in specs:
            c = tab[sc[i] // (ACTION_SIZE // k)]
            out[k][i] = r[glob if c is None else c] / m
    return out


def paired(a, b, label):
    d = a - b
    if np.allclose(d, 0):
        print(f"  {label}: identical")
        return
    one = wilcoxon(a, b, alternative="greater").pvalue
    two = wilcoxon(a, b).pvalue
    bs = [np.mean(d[np.random.randint(0, len(d), len(d))]) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {label}: {d.mean()*100:+.2f} pp  95% CI "
          f"[{lo*100:+.2f}, {hi*100:+.2f}]  p(one) {one:.2g}  p(two) {two:.2g}")


def main(a):
    np.random.seed(0)
    step = STEP_DEG
    norm = not a.raw_mean

    print(f"fit  {a.fit}")
    Df, ndf, scf = load(a.fit, a.n)
    print(f"dev  {a.dev}")
    Dd, ndd, scd = load(a.dev, a.n)
    print(f"test {a.test}")
    Dt, ndt, sct = load(a.test, a.n)
    print(f"  fit {len(Df)}, dev {len(Dd)}, test {len(Dt)} objects; "
          f"{NCAND} candidates, step {step:g} deg")

    _, c = np.unique(scf, return_counts=True)
    print(f"  fit set: {len(c)} distinct scouts, per angle min {c.min()}, "
          f"median {int(np.median(c))}, max {c.max()}")

    print("\nfit criterion: " + ("mean efficiency S/S_opt"
                                 if norm else "mean raw S"))
    acc, cnt = accumulate(Df, ndf, scf, norm)

    specs, meta = [], {}
    for k in BINS:
        tab, ns, glob = fit_bins(acc, cnt, k, a.min_n)
        specs.append((k, tab, glob))
        meta[k] = (tab, glob, int(np.median(ns)),
                   sum(c is not None for c in tab))
    EF = effs_all(Df, ndf, scf, specs)
    ED = effs_all(Dd, ndd, scd, specs)
    ET = effs_all(Dt, ndt, sct, specs)

    rows = []
    print(f"\n{'bins':>5} {'obj/bin':>8} {'fit':>8} {'dev':>8} {'test':>8}"
          f"   {'fitted':>9}")
    for k, tab, glob in specs:
        _, _, nmed, nfit = meta[k]
        rows.append((k, tab, glob, EF[k], ED[k], ET[k], nfit))
        print(f"{k:>5} {nmed:>8} {EF[k].mean()*100:>7.1f}% "
              f"{ED[k].mean()*100:>7.1f}% {ET[k].mean()*100:>7.1f}%   "
              f"{nfit:>4}/{k:<4}")

    # the number of bins is selected on dev only
    best = max(rows, key=lambda r: r[4].mean())
    k, tab, glob, _, ed, et, nfit = best
    print(f"\nselected on dev: {k} bins (dev {ed.mean()*100:.1f} %)")
    print(f"TEST, prior-only: {et.mean()*100:.1f} %   median "
          f"{np.median(et)*100:.1f} %")
    base = rows[0]
    if k != 1:
        paired(et, base[5], f"{k} bins - 1 bin  (does the scout angle help?)")
    off1 = [int(x) for x in REL_OFF[base[1][0] if base[1][0] is not None
                                   else base[2]]]
    print(f"offsets at 1 bin: indices {tuple(off1)} = "
          f"+{'/'.join(str(round(x*step)) for x in off1)} deg from the scout")
    if k != 1:
        # repeated absolute angles across bins = the table exploits alignment
        # of internal geometry with the object frame
        span = ACTION_SIZE * step
        print(f"offsets at {k} bins, with ABSOLUTE angles "
              f"(bin centre + offset, mod {span:g} deg):")
        w = ACTION_SIZE // k
        absall = []
        for b, c in enumerate(tab):
            lo, hi = round(b*w*step), round(((b+1)*w - 1)*step)
            if c is None:
                print(f"  scout {lo:>3}-{hi:>3} deg: as 1 bin")
                continue
            off = [int(x) for x in REL_OFF[c]]
            ctr = (b * w + (w - 1) / 2.0) * step
            ab = sorted((ctr + x * step) % span for x in off)
            absall += ab
            print(f"  scout {lo:>3}-{hi:>3} deg: "
                  f"+{'/'.join(str(round(x*step)) for x in off):<14} "
                  f"-> {'/'.join(f'{x:.0f}' for x in ab)}")
        if absall:
            h, _ = np.histogram(absall, bins=12, range=(0, span))
            print(f"  absolute angles per {span/12:.0f} deg: "
                  + " ".join(f"{int(x)}" for x in h)
                  + f"   ({len(absall)} angles from {sum(c is not None for c in tab)} bins)")

    if not a.paired:
        return
    z = np.load(a.paired, allow_pickle=True)

    def numeric(key):
        w = np.asarray(z[key])
        if w.dtype.kind not in "fiub" or w.size != len(et):
            return None
        return w.astype(float).ravel()

    # 1 bin vs the hand-chosen protocols: 'identical' means the fitted protocol
    # is one of them (which also confirms that scoring matches eval_ranker)
    fixed = [key for key in z.files
             if key.startswith("+") and numeric(key) is not None]
    if fixed:
        print("\n1 bin vs the fixed protocols in the npz (paired)")
        for key in fixed:
            paired(base[5], numeric(key), f"1 bin - {key}")

    v = numeric(a.key)
    if v is None:
        print(f"\n{a.paired}: key {a.key!r} missing or of different length "
              f"than the test set ({len(et)}); available {z.files}")
        return
    print(f"\nSVR ({a.key}) vs image-blind prior, paired on test")
    print(f"  SVR: {v.mean()*100:.1f} %   median {np.median(v)*100:.1f} %")
    paired(v, et, f"SVR - prior-only ({k} bins)")
    if k != 1:
        paired(v, base[5], "SVR - frozen fixed (1 bin)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit", required=True, help="h5 used to fit the tables (training set)")
    p.add_argument("--dev", required=True, help="h5 used to select the number of bins")
    p.add_argument("--test", required=True, help="h5 used to report the result")
    p.add_argument("--paired", help="per-object npz from paired_stats.py")
    p.add_argument("--key", default="SVR one-shot", help="SVR key in the npz")
    p.add_argument("--min-n", type=int, default=20,
                   help="minimum objects per bin; smaller bins fall back to 1 bin")
    p.add_argument("--raw-mean", action="store_true")
    p.add_argument("--n", type=int, default=None)
    main(p.parse_args())
