# ScrollGT

[![CI](https://github.com/jonmarrs/scrollgt/actions/workflows/ci.yml/badge.svg)](https://github.com/jonmarrs/scrollgt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Human ground-truth evaluation for the open Vesuvius Challenge scroll data — registered ink
targets, column-level reading targets, and fiber connectivity targets, each with anti-gaming
floors and our own negative results published.**

> ## ⚠ 2026-08-07 — the three `scroll1_*` pixel targets are misregistered; their leaderboard is withdrawn
>
> Agreement between the registered GT and the canon prediction does **not** peak at zero
> shift. It peaks ~190 level-0 voxels away on the train-exposed target and **~1766 voxels**
> away on the held-out flagship. Correcting a pure translation takes the canon teacher on
> the held-out target from roc_auc 0.582 to **0.718** — clearing the "> 0.60 would be news"
> bar stated below with two free parameters.
>
> **The "everything published reads at chance held-out" headline is therefore not
> established**, and neither is the GT-fine-tune negative that depends on it. Do not score
> against the `scroll1_*` targets until this is root-caused; treat published pixel rows as
> withdrawn rather than as a bar to beat.
>
> The ~8-voxel residual quoted throughout measured correspondence scatter, not absolute
> placement — a registration can have tight residuals and still be bodily displaced. We
> shipped the former as evidence for the latter, and the shipped `overlay_vs_canon.png`
> could not have caught it: it paints GT over a prediction that is itself near chance there.
>
> **Unaffected:** the PHerc 1667 column targets and all six fiber targets — different
> ground truth, no registration bridge.
>
> Detail + reproduction:
> [registration_offset_2026-08-07.md](https://github.com/jonmarrs/vesuvius-autoresearch/blob/main/reports/detector/registration_offset_2026-08-07.md).
> Found because `erdpx` closed villa PR
> [#1280](https://github.com/ScrollPrize/villa/pull/1280) saying the alignment example
> didn't show alignment working. It didn't.

The Vesuvius Challenge open-data bucket ships surface volumes and *model predictions* —
but no human ground truth aligned to the new re-flattened geometry. That makes an
uncomfortable question hard to answer: **does your ink model actually read, or does it
reproduce another model?**

ScrollGT closes that gap. It registers the 2023 Grand-Prize-era human ink annotations
onto the SOTA re-flattened geometry (exact `original.obj` UV bridge, ~8-voxel median
residual, gated alignment validation) and ships them as scoreable targets with a
one-command harness.

## Why trust this eval?

> **⚠ Withdrawn 2026-08-07** — all three bullets below are scored against the
> misregistered pixel targets (see the banner above). The eval did have teeth; it bit its
> authors for the wrong reason. What it caught was its own displaced labels, not weak models.

Because it has teeth — demonstrated on its own authors. Scored against these targets:

- the **released canon prediction** itself reads a held-out segment at ROC-AUC **0.56** (near chance);
- our **distilled students** (which score 0.79+ on a train-exposed segment) drop to **~0.55 held-out** — distillation reproduces the teacher *including its failures*;
- **fine-tuning on the registered GT made it worse** (0.558 → 0.531, collapsing to the trivial all-positive predictor).

Every one of those negatives is published in [`baselines/BASELINES.md`](baselines/BASELINES.md).
This benchmark was built by catching our own over-reads; it will catch yours too.

## Quickstart

```bash
git clone https://github.com/jonmarrs/scrollgt && cd scrollgt
pip install -e .            # installs the `scrollgt` CLI (source install; not on PyPI)
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

## Leaderboard (held-out flagship `scroll1_20231210121321`) — WITHDRAWN 2026-08-07

> These rows measure agreement with a ground truth displaced ~1766 level-0 voxels. They are
> kept visible for the correction record, not as a bar to beat. See the banner at the top.

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

## Column-level targets (v0.2 preview): PHerc 1667 merged geometry

`data/pherc1667_merged_columns` is the first **non-training-scroll** target — the merged
full-reading geometry of PHerc 1667 (read in full June 2026), with the published reading's
**22 columns registered onto the canonical grid** (all three preprint figure strips
independently recover the same transform; tiling closure 3 px over 30,097). There is **no
pixel GT** here: the ground truth is eight papyrologists' column-level consensus
(Coll. 1–4 traces, 5–22 text), CC BY-NC 4.0. Scoring measures *consistency with the
reading*, never letter accuracy:

```bash
# predict at grid resolution (full grid or a sub-extent + --origin), then:
scrollgt score-columns my_pred.npy data/pherc1667_merged_columns --json-out card.json
```

Metrics: `col_gutter_auc` (region-level — does signal concentrate in text columns vs
inter-column gutters?), `col_gutter_pixel_auc`, `line_period_peak_mean` (text-line
periodicity inside columns). Anti-gaming floor, measured: constant and papyrus-mask
predictions score exactly 0.5 (gutters are papyrus too); random noise shows the
region-AUC granularity (~0.58 at n=18 vs 17); the disclosed geometry-oracle ceiling is
1.0. Surface volumes for this segment don't exist in the bucket — render them with the
[gate-validated renderer](https://github.com/jonmarrs/vesuvius-autoresearch/blob/main/docs/SURFACE_RENDERER.md)
(clean-triple NCC 0.78 on this very scroll).

## Fiber connectivity targets (v0.3): can your tracer hold one fiber's identity?

Papyrus fibers physically define the U and V axes of a sheet, so tracing them helps both
flattening and surface segmentation. villa's 2026 open-problems post asks for exactly this, and
states the preference plainly: *"a tracer that confidently follows fewer fibers correctly is more
useful than one that follows more fibers with a higher error rate."*

**The headline finding: coverage and precision cannot rank a fiber tracer.** Four completely
different instance labellings of the same cube — connected components, one instance for
everything, one instance per voxel, and 50 random labels — all score **identical** coverage
(0.9177) and precision (0.2194), because those metrics are properties of the fiber *mask*, not of
the *labelling*. Only expected run length and the merge count separate them, and even raw ERL
alone is gameable: labelling the whole cube once scores 199.18 against an oracle's 258.27 while
its merge-penalized ERL is exactly **0.00**.

So `score-fibers` never prints one ERL without the other, and never prints either without the
tolerance. That claim is pinned by `tests/test_fiber_gaming.py` — break it and CI fails.

```bash
# labels.npy: cube-shaped int array, 0 = background, one distinct id per predicted fiber
scrollgt score-fibers labels.npy data/fibers_s1_00497_01497_03997_256 --json-out card.json
```

**No GPU, no model download, no network.** Each target ships the hand-traced ground truth and the
reference fiber mask (~250 KB/cube), so the published floors reproduce from the repo alone — a
test enforces exactly that. Pass `--recompute-floors` to verify them yourself from the shipped
mask (~50 s per cube) instead of reading the published values.

Six cubes: five from Scroll 1, plus `s5_03997_01497_03997_256` designated the **cross-scroll**
reporting split. The ground truth is a public villa dataset and cannot be hidden, so that is a
labelled convention for reporting transfer — not a claim of held-out secrecy.

Ground truth: villa's `fiber-skeletons` dataset (`dl.ash2txt.org/datasets/fiber-skeletons/`),
every fiber in each cube hand-traced in WEBKNOSSOS at 7.91 µm. Only the `nml/` files carry fiber
identity; the shipped `labelsTr/*.tif` are semantic and cannot support connectivity metrics.
Reference mask: `scrollprize/fiber_hz_vt` (Apache-2.0) at P ≥ 0.5, identical for every entrant so
scorecard differences come from the labelling rather than the segmentation.

**Our own tracer loses to connected components on both metrics, on all six cubes** — published in
[`baselines/BASELINES.md`](baselines/BASELINES.md) rather than hidden. That is the bar to clear.

## Roadmap

- **v0.2:** the PHerc 1667 column-level target above **shipped 2026-07-18** (the open
  bucket ships only model predictions for 1667 — never GT-eligible here — so the target
  registers the published scholar-validated reading instead). Remaining v0.2 work:
  model baseline rows for `score-columns`, and per-line structure as the transcription
  artifacts propagate to the bucket. (Scrolls 2–3 are not extendable today: a 2026-07-17
  bucket survey found Scroll 2 ships no segments and Scroll 3 no labels — both scrolls
  are unread, which is exactly why the First Letters prizes are open.)
- Converting the three **withheld** v0.1 regions into targets as independent orientation
  validation becomes available.
- Leaderboard: submit a scorecard via PR/issue (see `baselines/BASELINES.md`).

## Provenance & method

Registration method, gates, and the full audit trail (including one target whose
teacher-dependent gate correctly *false-negatived* and was validated teacher-free) live
in the meta.json files and in the source repo's reports
(`registered_gt_validation.md`, `registered_gt_heldout_validation.md`). Ground truth
origin: 2023 Grand-Prize-era human annotations (villa `ink-detection` train scrolls);
surface volumes: `s3://vesuvius-challenge-open-data/` (anonymous).

## License

**Code: MIT** (see `LICENSE`).

**Data: mixed, per target.** Check each target's `meta.json` before reuse.

- The three pixel-level Scroll-1 targets (`scroll1_*`) register 2023 Grand-Prize-era human ink
  annotations from the Vesuvius Challenge open data release; see the challenge's data terms.
- The six fiber targets (`fibers_*`) carry hand-traced skeletons from villa's `fiber-skeletons`
  dataset (`dl.ash2txt.org/datasets/fiber-skeletons/`), also a Vesuvius Challenge release — see
  the challenge's data terms. Each target's `mask.npz` is derived from `scrollprize/fiber_hz_vt`
  (Apache-2.0). Provenance for both is recorded in the target's `meta.json`.
- `data/pherc1667_merged_columns/` is **CC BY-NC 4.0**: its column coordinates and
  transcription facts derive from Angelotti et al., *Complete virtual unwrapping and reading of
  a rolled Herculaneum papyrus* (<https://scrollprize.org/pdf/main.pdf>). Attribution required,
  **non-commercial use only**. This directory does not inherit the MIT grant.
