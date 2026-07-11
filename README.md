# ScrollGT

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
pip install scrollgt        # or: pip install -e . from a clone
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
| `data/scroll1_20231210121321` | **held-out flagship** — no public model we know of trained here | teacher-free (residual 7.85vx, text-line periodicity 0.871) |

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
