# ScrollGT

[![CI](https://github.com/jonmarrs/scrollgt/actions/workflows/ci.yml/badge.svg)](https://github.com/jonmarrs/scrollgt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Registered human ground-truth ink evaluation for the open Vesuvius Challenge SOTA scroll data.**

The Vesuvius Challenge open-data bucket ships surface volumes and *model predictions* —
but no human ground truth aligned to the new re-flattened geometry. That makes an
uncomfortable question hard to answer: **does your ink model actually read, or does it
reproduce another model?**

ScrollGT closes that gap. It registers the 2023 Grand-Prize-era human ink annotations
onto the SOTA re-flattened geometry (exact `original.obj` UV bridge, ~8-voxel median
residual, gated alignment validation) and ships them as scoreable targets with a
one-command harness.

## Why trust this eval?

Because it has teeth — demonstrated on its own authors. Scored against these targets:

- the **released canon prediction** itself reads a held-out segment at ROC-AUC **0.56** (near chance);
- our **distilled students** (which score 0.79+ on a train-exposed segment) drop to **~0.55 held-out** — distillation reproduces the teacher *including its failures*;
- **fine-tuning on the registered GT made it worse** (0.558 → 0.531, collapsing to the trivial all-positive predictor).

Every one of those negatives is published in [`baselines/BASELINES.md`](baselines/BASELINES.md).
This benchmark was built by catching our own over-reads; it will catch yours too.

## Quickstart

```bash
git clone https://github.com/jonmarrs/scrollgt && cd scrollgt
pip install -e .            # (PyPI package coming; for now install from source)
# predict a probability map over the target region (see data/<target>/meta.json
# for the exact SOTA S3 zarr, pyramid level, and y0/x0/size), then:
scrollgt score my_prediction.png data/scroll1_20231210121321 --json-out card.json
```

Output: a markdown scorecard row + JSON with threshold-swept **F1** (primary),
**AP-prevalence-lift** (the imbalance-robust real-signal gate: a constant prediction
scores ~1.0 no matter how it games F1), and **ROC-AUC** (secondary diagnostic).

Prize-compliance pre-check (window cap + train/predict overlap):

```bash
scrollgt check --window-px 64 --scan-um 8.0 --regions-json regions.json
```

## Targets (v0.1)

| target | role | registration validation |
|---|---|---|
| `data/scroll1_20230702185753` | train-exposed for the published baselines (disclosed) | enrichment-gated (5.05), residual 7.92vx |
| `data/scroll1_20230702185753_y7000_x4000` | second region of the train-exposed segment | direct 4-candidate orientation probe (3.13 vs ≤1.50), residual 8.07vx |
| `data/scroll1_20231210121321` | **held-out flagship** — no public model we know of trained here | teacher-free (residual 7.85vx, text-line periodicity 0.871) |

A fourth gate-passing region was **withheld** because its orientation is currently
unverifiable (chance-quality teacher there defeats the enrichment check) — see
`baselines/BASELINES.md`. Targets only ship when validation is real.

## Leaderboard (held-out flagship `scroll1_20231210121321`)

The number that matters — scored against human ground truth on a segment no listed model
trained on. Everything published so far sits at chance; **an honest ROC-AUC > 0.60 here
would be news.** Full tables + the train-region contrast in
[`baselines/BASELINES.md`](baselines/BASELINES.md); submit a row via
[`CONTRIBUTING.md`](CONTRIBUTING.md).

| model | exposure | ROC-AUC | AP-lift | val_f1 |
|---|---|---|---|---|
| canon teacher (released prediction) | — | 0.563 | 1.150 | 0.295 |
| arm A (1-scroll student) | selection-set only | 0.563 | 1.203 | 0.311 |
| arm B (2-scroll student) | **clean held-out** | 0.553 | 1.161 | 0.311 |
| arm C (3-scroll student) | **clean held-out** | 0.558 | 1.165 | 0.310 |
| arm C + GT fine-tune | **clean held-out** | 0.531 | 1.090 | 0.309 |
| trivial all-positive | — | 0.500 | 1.000 | 0.309 |

Note how close the `val_f1` column is to the trivial predictor (0.309) — at this ink
prevalence F1 is near-degenerate, which is exactly why ScrollGT's headline is
AP-prevalence-lift, not F1.

Each target directory contains `gt_ink.png` (registered binary label), `meta.json`
(exact predict-region spec + full registration provenance and caveats), and an
`overlay_vs_canon.png` sanity visual. The ~8-voxel registration residual smears stroke
edges at 64px scale: **scores are lower bounds on true agreement** — stated once here
and in every meta.json, so nobody over-reads a low score either.

## Honest-metrics contract

`scrollgt.metrics.segmentation_metrics` is the exact contract used for all published
baselines (kept in sync with
[vesuvius-autoresearch](https://github.com/jonmarrs/vesuvius-autoresearch)
`detector/metrics.py`):

- **`val_f1`** — threshold-swept F1, the headline number;
- **`ap_prevalence_lift`** — average precision ÷ ink prevalence; the anti-gaming gate
  (all-positive predictors get F1 = 2p/(1+p) for free, but lift ≈ 1.0);
- **`roc_auc`** — secondary diagnostic only;
- mask-restricted, pooled over the full region; degenerate regions return NaN, never a
  fake score.

## Roadmap

- **v0.2 (August):** registered-GT targets on **Scrolls 2–3** — the live First
  Letters/Title prize targets.
- Leaderboard: submit a scorecard via PR/issue (see `baselines/BASELINES.md`).

## Provenance & method

Registration method, gates, and the full audit trail (including one target whose
teacher-dependent gate correctly *false-negatived* and was validated teacher-free) live
in the meta.json files and in the source repo's reports
(`registered_gt_validation.md`, `registered_gt_heldout_validation.md`). Ground truth
origin: 2023 Grand-Prize-era human annotations (villa `ink-detection` train scrolls);
surface volumes: `s3://vesuvius-challenge-open-data/` (anonymous).

## License

MIT. Ground-truth annotations derive from the Vesuvius Challenge open data release;
see the challenge's data terms.
