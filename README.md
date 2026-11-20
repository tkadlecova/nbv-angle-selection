# Projection-angle selection from a single scout view

Code for the paper *Projection-Angle Selection from a Single Scout View: A
Set-Ranking Approach for Radiographic Inspection* (M. Tkadlecová, P. Blažek,
J. Batenburg, T. Zikmund, J. Kaiser).

A neural ranker selects three follow-up projection angles from one scout
radiograph by scoring all 6,545 candidate angle sets. Everything is simulated
in 2-D parallel-beam geometry with ASTRA.

**Naming.** The method is the *Set Visibility Ranker (SVR)*; the per-defect,
per-angle criterion is the *visibility V*. In the code the network class is
`SetVisibilityRanker`, the training script `core/train_svr.py`, and the
visibility matrix of each object is stored in the HDF5 files under the key `D`. Naming note: the method is called VSR (Visibility Set Ranker) in the paper; in the code it appears as SVR (train_svr.py, SetVisibilityRanker, svr_ens*.pt).

## Layout

| Folder | Contents |
|---|---|
| `core/` | phantom generator and visibility V, dataset generation, the ranker and its training, evaluation and paired statistics |
| `experiments/` | ablations and baselines: oracle ladder, weight sweep, frozen fixed protocol, prior-only ablation, flatness of the objective, latency |
| `observer/` | validation of V against a channelized Hotelling observer |

Scripts in `experiments/` and `observer/` import from the other
folders through the small `_paths.py` next to them, so every script can be
run directly. Data, checkpoints and outputs are read from and written to the
current working directory.

## Installation

```
pip install -r requirements.txt
```

ASTRA Toolbox is easiest to install with conda (`conda install -c astra-toolbox astra-toolbox`).
All projectors are CPU (`strip`); a GPU only speeds up training.

## Pipeline

Run from the folder that holds (or will hold) the data; `<repo>` is the path to this repository.

**1. Data.** One HDF5 file per regime and split. Test sets use the seed
offsets 950000 (realistic), 960000 (random) and 970000 (structured); the
development set uses 900000.

```
python <repo>/core/generate_vmatrix_dataset.py --dist realistic  --n 2000 --seed-offset <train offset> --out dmx_realistic_train.h5
python <repo>/core/generate_vmatrix_dataset.py --dist realistic  --n 150  --seed-offset 950000 --out dmx_realistic_final.h5
python <repo>/core/generate_vmatrix_dataset.py --dist random     --n 150  --seed-offset 960000 --out dmx_random_final.h5
python <repo>/core/generate_vmatrix_dataset.py --dist structured --n 150  --seed-offset 970000 --out dmx_structured_final.h5
python <repo>/core/generate_vmatrix_dataset.py --verify      # closed-form score == angle-by-angle score
```

**2. Training.** Three independently trained models form the ensemble.

```
python <repo>/core/train_svr.py --train dmx_realistic_train.h5 dmx_random_train.h5 dmx_structured_train.h5 \
    --val dmx_dev.h5 --noise          # -> svr_best.pt, rename to svr_ens{1,2,3}.pt
python <repo>/core/train_pointwise.py  --train ... --val dmx_dev.h5 --out pointwise_ens1.pt
```

**3. Evaluation and ablations** — see the tables below. Each script
prints its usage in its header.

## Where each result comes from

| Result | Script |
|---|---|
| Efficiency vs the exhaustive optimum, one-shot / sequential / fixed / random (Table 1) | `core/eval_ranker.py` |
| Paired differences, bootstrap CIs, Wilcoxon tests; angle-wise ablation; one-shot vs sequential; per-object vectors | `core/paired_stats.py` (uses `core/train_pointwise.py`) |
| Noise robustness of the ranker inputs | `core/eval_ranker.py --noise --i0 ...` |
| Oracle ladder: exhaustive, greedy, pointwise | `experiments/greedy_oracle.py` |
| Weight sweep of the min/mean weighting, submodularity violations | `experiments/weight_sweep.py` |
| Fixed protocol frozen on the development set, ranking stability | `experiments/frozen_fixed.py` |
| Prior-only ablation (scout angle without the image) | `experiments/prior_only.py` |
| Inference latency, one-shot vs sequential | `experiments/measure_latency.py` |
| Flatness of the objective near its optimum | `experiments/smoothness_figure.py` |

### Model-observer validation (`observer/`)

| Step | Script |
|---|---|
| Paired projections (defect present / filled with polymer) | `make_cho_dataset.py` |
| CHO, physical rescaling, choice of I0 (`--pilot`) | `cho_validation.py` |
| d' with common random numbers, flatness under the observer | `cho_compare.py` |
| Exhaustive selection, SVR and fixed protocols under the observer, one I0 per run | `pu_oracle_test.py` |
| Split-half headroom above the fixed protocol | `cho_headroom.py` |
| Headroom for budgets of 1, 2 and 3 angles | `cho_budget_sweep.py` |
| Correlation of each factor of V with d' | `v_factor_analysis.py` |
| Binary vs graded metal penalty | `metal_penalty_diagnostic.py` |
| Per-defect vs pooled correlation of V and d' | `per_defect_rho.py` |

```
python <repo>/observer/make_cho_dataset.py --dist realistic --n 50 --seed-offset 950000 \
    --out cho_realistic.h5 --verify-against dmx_realistic_final.h5
python <repo>/observer/pu_oracle_test.py --dist realistic --seed-offset 950000 --cho cho_realistic.h5 \
    --i0 5000 --svr svr_ens1.pt svr_ens2.pt svr_ens3.pt
```

The observer scripts regenerate each phantom from its seed and stop with an
error if the regenerated V matrix does not match the stored one.

## Data format

`dmx_*.h5`: `sino_all` (n, 36, 512) transmission at all angles, `D` (n, 6, 36)
visibility matrix V padded to 6 defects, `n_def`, `scout_idx`, `L_plastic` and
`L_metal` (separate line integrals, for a later polychromatic model), and
`geom_gt` (64x64 ground-truth maps, not used by the method).

`cho_*.h5` adds `g_full`, `g_ablated` and the per-angle normaliser `denom`.
