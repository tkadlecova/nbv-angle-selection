"""Inference latency of one-shot and sequential mode (Section 6.1).

Uses the same functions as eval_ranker.py. Each call is timed between
torch.cuda.synchronize() barriers after a warm-up, on real test objects;
reports the median and the 10-90 % interdecile range.

    python measure_latency.py --data dmx_realistic_final.h5 --n 150
"""
import _paths  # noqa: F401  (makes core/, observer/, ... importable)
import argparse
import time

import numpy as np
import h5py
import torch

import eval_ranker as E


def _ms(fn, *args):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dmx_realistic_final.h5")
    ap.add_argument("--model", nargs="+",
                    default=["svr_ens1.pt", "svr_ens2.pt",
                             "svr_ens3.pt"])
    ap.add_argument("--n", type=int, default=150,
                    help="number of test objects to time")
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    models, geom_input = E.load_models(args.model)
    print(f"Ensemble: {len(models)} model(s), geom_input={geom_input}\n")

    with h5py.File(args.data, "r") as f:
        sino_all = f["sino_all"][: args.n].astype(np.float32)
        scouts = f["scout_idx"][: args.n]
        geoms = f["geom_gt"][: args.n] if (geom_input == "gt" and "geom_gt" in f) else None

    n = len(sino_all)
    print(f"Timing on {n} objects from {args.data}")
    print(f"({args.warmup} warm-up calls, then one measurement per object)\n")

    s0, sc0 = sino_all[0], int(scouts[0])
    g0 = geoms[0] if geoms is not None else None
    for _ in range(args.warmup):
        E.ranker_one_shot(models, s0, sc0, g0)
        E.ranker_sequential(models, s0, sc0, g0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_one, t_seq = [], []
    for i in range(n):
        s, sc = sino_all[i], int(scouts[i])
        gg = geoms[i] if geoms is not None else None
        t_one.append(_ms(E.ranker_one_shot, models, s, sc, gg))
        t_seq.append(_ms(E.ranker_sequential, models, s, sc, gg))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n}")

    t_one, t_seq = np.array(t_one), np.array(t_seq)

    print("\n" + "=" * 62)
    print(f"{'Mode':<14}{'median':>10}{'p10':>10}{'p90':>10}{'mean':>10}")
    print("-" * 62)
    for name, t in [("one-shot", t_one), ("sequential", t_seq)]:
        print(f"{name:<14}{np.median(t):>9.2f}{np.percentile(t,10):>10.2f}"
              f"{np.percentile(t,90):>10.2f}{t.mean():>10.2f}   ms")
    print("=" * 62)
    ratio = np.median(t_seq) / np.median(t_one)
    print(f"\nRatio sequential / one-shot (median): {ratio:.2f}x")
    print(f"Difference in median: +{np.median(t_seq) - np.median(t_one):.2f} ms")

    np.savetxt("latency_modes.csv", np.c_[t_one, t_seq], delimiter=",",
               header="one_shot_ms,sequential_ms", comments="", fmt="%.4f")
    print("\nSaved: latency_modes.csv")


if __name__ == "__main__":
    main()
