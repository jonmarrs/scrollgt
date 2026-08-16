# ScrollGT

[![CI](https://github.com/jonmarrs/scrollgt/actions/workflows/ci.yml/badge.svg)](https://github.com/jonmarrs/scrollgt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Human ground-truth evaluation for the open Vesuvius Challenge scroll data — registered ink
targets, column-level reading targets, and fiber connectivity targets, each with anti-gaming
floors and our own negative results published.**

> ## ⚠ 2026-08-07 — the held-out target was misregistered. It is fixed, and the headline result REVERSES.
>
> **What was wrong.** The held-out label was built with a hardcoded `LEVEL0_SHAPE` belonging
> to a *different segment*, so its region crop was scaled wrongly — emitting a label that was
> displaced and stretched. Agreement with the canon prediction peaked ~1766 level-0 voxels
> away from zero shift instead of at it.
>
> **What that means for what this benchmark claimed.** The old headline — *"everything
> published reads at chance on the held-out segment"* — was an artifact of our own broken
> registration. It is **retracted**. On the corrected label, on the same segment, with the
> same models:
>
> | model (clean held-out) | roc_auc, old (invalid) | roc_auc, corrected |
> |---|---|---|
> | canon teacher | 0.563 | **0.753** |
> | arm B (2-scroll student) | 0.553 | **0.731** |
> | arm C (3-scroll student) | 0.558 | **0.746** |
> | legacy detector (all-positive floor) | 0.501 | 0.518 |
>
> AP-prevalence-lift moves from ~1.15 (chance) to **2.15–2.44**. These models were reading
> held-out ink the whole time; the benchmark was measuring its own misalignment.
>
> **The gate caught this and we overrode it.** The 2026-07 run failed the teacher-enrichment
> gate (1.68), and we attributed that to a weak teacher and built a teacher-free gate to get
> past it. On the fixed pipeline the same convention scores enrichment **6.01**. The gate was
> right; we explained away a true positive.
>
> **Also retracted:** the GT-fine-tune negative, which was fine-tuning on displaced labels.
>
> **Resolution limit (a spec, not an open bug).** A ~32 level-2 px / **0.31 mm** placement
> uncertainty remains and is irreducible for this method: `original.obj` carries the
> 2023 label mapping in the *old* 7.91 µm scan frame, and the 2023 and 2026 segmentations of
> this sheet are materially different surfaces (unpaired 3D similarity between the two
> meshes leaves p50 64 / p90 249 old-scan voxels). Two candidate fixes were tested and
> falsified. **Features closer together than ~0.31 mm cannot be scored reliably here, and
> all absolute scores are mild lower bounds.** A placement gate enforces this at 48 px —
> 9× below the 435 px bug above. Per-target, and the **global figure is optimistic** —
> placement varies across the region, so per-768px-tile scatter is quoted too:
>
> | target | global | per-tile sd (dy/dx) | worst tile | verdict |
> |---|---|---|---|---|
> | held-out `20231210121321` | 32.0 px / 0.31 mm | 8.2 / 9.5 | ~50 px | usable |
> | train-exposed `20230702185753` | 46.6 px / 0.45 mm | **26.8 / 33.0** | **~100 px / 0.96 mm** | **indicative only** |
>
> The field is **non-rigid** — a fitted plane leaves scatter equal to the raw scatter, so it
> is neither a constant offset nor a scale error, and there is no convention bug left to
> find. The threshold was not raised to accommodate the train-exposed target.
>
> **Unaffected:** the PHerc 1667 column targets and every fiber target — different ground
> truth, no registration bridge. (Fiber targets numbered six at the time of this note;
> the family has since grown to eleven — see the Fiber connectivity section below.)
>
> Detail + reproduction:
> [registration_offset_2026-08-07.md](https://github.com/jonmarrs/vesuvius-autoresearch/blob/main/reports/detector/registration_offset_2026-08-07.md)
> · check any registration with `scripts/probe_registration_offset.py`.
> Found because `erdpx` closed villa PR
> [#1280](https://github.com/ScrollPrize/villa/pull/1280) saying the alignment example
> didn't show alignment working. It didn't, and this is why.

The Vesuvius Challenge open-data bucket ships surface volumes and *model predictions* —
but no human ground truth aligned to the new re-flattened geometry. That makes an
uncomfortable question hard to answer: **does your ink model actually read, or does it
reproduce another model?**

ScrollGT closes that gap. It registers the 2023 Grand-Prize-era human ink annotations
onto the SOTA re-flattened geometry (exact `original.obj` UV bridge, ~8-voxel median
residual, gated alignment validation) and ships them as scoreable targets with a
one-command harness.

## Why trust this eval?

Not because it produced a dramatic negative result — it did, and the negative result was
**our own bug**. Trust it because that is documented rather than buried:

- the 2026-07 release claimed every published model reads the held-out segment at chance;
- the actual cause was a hardcoded constant in our registration code, found only after an
  external reviewer said our alignment example didn't show alignment working;
- the retraction, the root cause, the corrected numbers, and the resolution limit we
  *cannot* engineer away are all in this README, `baselines/BASELINES.md`, and the report.

The eval had teeth. It bit its authors — for the wrong reason first, and now for the right
one. What it actually establishes today:

- the **released canon prediction** reads the held-out segment at ROC-AUC **0.753**;
- our **distilled students**, never trained on it, read it at **0.731–0.746** (AP-lift
  2.3–2.4) — genuine held-out generalization, not the chance result we published;
- the **all-positive floor** sits at 0.518, so those numbers are above a real baseline;
- **a tight registration residual is not a placement check** — the ~8-voxel residual we
  cited as evidence of correct alignment coexisted with a ~1766-voxel displacement. Every
  target is now gated on agreement peaking at zero shift, not on residual alone;
- **the targets resolve to ~0.31 mm**, stated as a spec rather than discovered later.

The full record — the withdrawn rows, the corrected rows, and what is still broken — is in
[`baselines/BASELINES.md`](baselines/BASELINES.md).

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

**One scoreable pixel target, plus a documented contrast.** As of 2026-08-14 both
`20230702185753` regions are marked **non-scoring**: local placement error there reaches
~1.9x the 512 um prize analysis window (64 px at 8 um), so within a single window a model
can be scored against ground truth from a different part of the sheet. `20231210121321`
stays scoreable, with worst-case local error at 0.94 windows. Their published rows remain in
[`baselines/BASELINES.md`](baselines/BASELINES.md) as the train-region contrast that shows
why held-out evaluation matters -- a record, not a bar to beat.

`scrollgt score` refuses a non-scoring target (`--allow-non-scoring` reproduces the
historical rows) and separately refuses one that fails its placement check
(`--allow-failing-placement`). The two are independent, because "should you evaluate against
this" and "is the label in the right place" are different questions. Every scorecard reports
placement, not just the residual.

| target | role | scoreable? | placement (gate 48 px) | registration validation |
|---|---|---|---|---|
| `data/scroll1_20230702185753` | train-region contrast (record only) | **no** | 46.6 px / 0.45 mm — passes by 1.4 px; worst tile ~0.96 mm = 1.9 windows | enrichment-gated (5.05), residual 7.92vx |
| `data/scroll1_20230702185753_y7000_x4000` | second region, same segment (record only) | **no** | **53.3 px / 0.51 mm — FAILS the gate** | direct 4-candidate orientation probe (3.13 vs ≤1.50), residual 8.07vx |
| `data/scroll1_20231210121321` | **held-out flagship** — no public model we know of trained here | **yes** | **32.0 px / 0.31 mm — passes; worst tile 0.94 windows** | re-registered 2026-08-07; enrichment 6.01 (decisive), residual 7.95vx, periodicity 0.867 |

**The problem is segment-wide, not region-wide.** Both regions of `20230702185753` are
poorly placed (46.6 px and 53.3 px, local error to ~1 mm) while `20231210121321` is 3–4×
tighter. This is cross-scan disagreement between the 2023 and 2026 segmentations of that
sheet, not a correctable offset — so `20231210121321` is the only pixel target we would
stand behind.

**And the pool is exhausted, not merely unprocessed.** Six Scroll-1 segments carry a 2023
hand ink label. Three of them (`20230820203112`, `20230826170124`, `20230903193206`) are
absent from the open data entirely — neither `ink-detection/` nor `surface-volumes/`
resolves — so there is no geometry to register a label onto. Of the three that remain,
`20230702185753` is poorly placed in both of its regions and `20231005123336` sits at 57.5
level-2 px against the 48 px gate (the withheld region discussed below), which leaves
`20231210121321`. Measured 2026-08-15; the open data changes, so this is a
[re-runnable probe][probe] rather than a claim.

**What this costs you as a user.** A single-target pixel family cannot separate model
quality from segment idiosyncrasy: a score here is a score on one sheet. Read the pixel
leaderboard accordingly, and prefer the fiber family when you need more than one point of
comparison — it is the only family here with more than one target; the column family has
exactly one and carries the same caveat, disclosed under it below. Expanding it needs new
upstream data — either a re-flattening of
one of the three absent labelled segments, or a hand label on one of the eight 2023-era
segments that are re-flattened but unlabelled. Neither is sufficient on its own: a fresh
segment still has to place well enough to score, and clearing the gate is necessary rather
than sufficient. Of the three labelled segments present in the open data, two clear the
48 px placement gate in at least one region — the table above scores regions, and
`20230702185753` clears it in one region while failing in the other — and only one of those
two is usable as a target.

[probe]: https://github.com/jonmarrs/vesuvius-autoresearch/blob/main/reports/detector/gt_training_data_exhaustion_2026-08-15.md

A fourth region (`20231005123336_y4000_x2500`) is **withheld**, but not for the reason this
README used to give. We said its orientation was unverifiable because the canon teacher was
chance-quality there; that collapse was our own second hardcoded level-0 shape, and
re-registered with the fix, teacher-enrichment is 4.88 and the orientation is decisively
determined. It stays withheld on a properly measured criterion instead: placement 57.5
level-2 px, over the 48 px gate — see `baselines/BASELINES.md`, which also records why that
figure is quoted to ±1 px rather than more finely. Targets only ship when
validation is real, and so do the reasons we give for holding them back.

## Leaderboard (held-out flagship `scroll1_20231210121321`)

**Corrected 2026-08-07** — these replace the withdrawn 2026-07 rows, which were scored
against a misregistered label (see the banner at the top). Scored against human ground
truth on a segment no listed model trained on. Full tables + the train-region contrast in
[`baselines/BASELINES.md`](baselines/BASELINES.md); submit a row via
[`CONTRIBUTING.md`](CONTRIBUTING.md).

| model | exposure | ROC-AUC | AP-lift | val_f1 | *(withdrawn 2026-07 ROC)* |
|---|---|---|---|---|---|
| canon teacher (released prediction) | — | 0.753 | 2.154 | 0.572 | *0.563* |
| arm A (1-scroll student) | selection-set only | 0.772 | 2.672 | 0.501 | *0.563* |
| arm B (2-scroll student) | **clean held-out** | 0.731 | 2.338 | 0.440 | *0.553* |
| arm C (3-scroll student) | **clean held-out** | 0.746 | 2.440 | 0.466 | *0.558* |
| legacy detector (all-positive) | — | 0.518 | 1.009 | 0.311 | *0.501* |

The old table claimed every row sat at chance and that "an honest ROC-AUC > 0.60 would be
news." The models were already there; our registration was hiding it. The all-positive
floor at 0.518 / lift 1.009 is the comparison that makes the rest meaningful.

`arm C + GT fine-tune` is **not listed**: it was fine-tuned on the displaced label, so its
published 0.531 measured nothing. It needs retraining before it can be scored.

Scores are **mild lower bounds**: the target resolves to ~0.31 mm (32 level-2 px placement
uncertainty), which is a floor of the method, not an open defect.

Each target directory contains `gt_ink.png` (registered binary label) and `meta.json`
(exact predict-region spec + full registration provenance and caveats).

**Alignment evidence.** `data/scroll1_20231210121321/alignment_evidence.png` shows, at
letterform scale: the canon prediction alone · the registered GT drawn as an *outline* over
it · and a per-pixel agreement map (green = both, red = GT only, blue = prediction only).

This replaces the old `overlay_vs_canon.png`, which painted the GT opaquely *on top of* the
prediction — hiding the agreement it was supposed to demonstrate, and showing nothing at all
on a segment where the prediction is weak. That visual is why a misregistration survived to
release. Regenerate with
[`scripts/make_alignment_evidence.py`](https://github.com/jonmarrs/vesuvius-autoresearch/blob/main/scripts/make_alignment_evidence.py).

Read the agreement map for *systematic* colour fringing: red consistently on one edge of a
stroke and blue on the opposite edge means a residual shift, which is exactly what this
target still shows (~130 voxels). Symmetric fringing is just stroke-edge scatter.

**Do not use the residual to judge placement.** The ~8-voxel median residual measures
correspondence *scatter*; it was ~8 voxels while the label sat ~1766 voxels out of place.
Use `scripts/probe_registration_offset.py`, which checks that agreement peaks at zero shift.
Scores here remain **lower bounds on true agreement**.

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

**What this costs you as a user.** The column family has exactly one target,
`pherc1667_merged_columns`. Like the single target the pixel family is down to above, a
single target cannot separate model quality from target idiosyncrasy — this scroll's
preservation, column count, and gutter geometry are baked into every score, and there is no
second target to check whether a high or low score generalizes. Expanding it needs another
scroll with a published column-level reading whose geometry can be registered onto the open
data the way PHerc 1667's was; no second candidate is in view today. This is a limitation of
the benchmark, not a complaint about the upstream reading, which is exactly what makes this
target scoreable at all.

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
reference fiber mask — ~210–350 KB for a 256³ cube and 2.1–2.9 MB for a 512³ one, 12 MB for all
of `data/` — so the published floors reproduce from the repo alone, and a test enforces exactly
that. Pass `--recompute-floors` to verify them yourself from the shipped mask instead of reading
the published values: ~50 s for a 256³ cube, several minutes for a 512³ one (its
connected-components floor alone is ~70 s, at ~8 GB peak RSS).

Eleven cubes, in two size classes: **eight at 256³** and **three at 512³**. The five `s1_*`
cubes are the `primary` split; the six `s5_*` cubes — three at each size — are the
**cross-scroll** reporting split, up from the single `s5_03997_01497_03997_256` cube this
family shipped with. The ground truth is a public villa dataset and cannot be hidden, so that
split is a labelled convention for reporting transfer — not a claim of held-out secrecy. ERL is
a length statistic and is never averaged across the two size classes; `aggregate_fiber_scores`
raises rather than doing so. That has a cost worth stating up front: the 512³ class is 3
cross-scroll and **0 primary**, so it carries no same-scroll counterpart to transfer from, and
the usable same-vs-cross comparison is n=3 against n=5 inside the 256³ class alone. Full
per-class tables, including oracle ceilings for every cube, are in
[`baselines/BASELINES.md`](baselines/BASELINES.md).

Ground truth: villa's `fiber-skeletons` dataset (`dl.ash2txt.org/datasets/fiber-skeletons/`),
every fiber in each cube hand-traced in WEBKNOSSOS at 7.91 µm. Only the `nml/` files carry fiber
identity; the shipped `labelsTr/*.tif` are semantic and cannot support connectivity metrics.
Reference mask: `scrollprize/fiber_hz_vt` (Apache-2.0) at P ≥ 0.5, identical for every entrant so
scorecard differences come from the labelling rather than the segmentation.

**Our own tracer loses to connected components on both metrics, on all six cubes it has been
scored against** — published in [`baselines/BASELINES.md`](baselines/BASELINES.md) rather than
hidden. That is the bar to clear. The five cubes added since ship with ground truth, mask, and
oracle/floor scores; the tracer has not yet been re-run against them.

## Roadmap

- **v0.2:** the PHerc 1667 column-level target above **shipped 2026-07-18** (the open
  bucket ships only model predictions for 1667 — never GT-eligible here — so the target
  registers the published scholar-validated reading instead). Remaining v0.2 work:
  model baseline rows for `score-columns`, and per-line structure as the transcription
  artifacts propagate to the bucket. (Scrolls 2–3 are not extendable today: a 2026-07-17
  bucket survey found Scroll 2 ships no segments and Scroll 3 no labels — both scrolls
  are unread, which is exactly why the First Letters prizes are open.)
- **The v0.1 pixel family will not grow, and this item is closed rather than pending.** An
  earlier version of this roadmap promised to convert three withheld regions into targets
  "as independent orientation validation becomes available". That was wrong twice over.
  There is one withheld region (`20231005123336_y4000_x2500`), plus its sibling
  `20231005123336_y7000_x4000` dropped earlier at prep for near-zero registered ink — and
  orientation is not what holds either of them back. The withheld region is placed at 57.5
  level-2 px against a 48 px gate, a figure determined to about ±1 px. We have not isolated
  a cause for it — no per-tile field was measured there — but it is consistent with the cross-scan disagreement
  measured on `20230702185753`, where re-registration demonstrably does not close the gap.
  What we can say without a cause is that nothing in our processing queue is known to clear
  it. The labelled-segment pool behind the family is exhausted — see the disclosure under
  **Targets** above.
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
- The eleven fiber targets (`fibers_*`) carry hand-traced skeletons from villa's `fiber-skeletons`
  dataset (`dl.ash2txt.org/datasets/fiber-skeletons/`), also a Vesuvius Challenge release — see
  the challenge's data terms. Each target's `mask.npz` is derived from `scrollprize/fiber_hz_vt`
  (Apache-2.0). Provenance for both is recorded in the target's `meta.json`.
- `data/pherc1667_merged_columns/` is **CC BY-NC 4.0**: its column coordinates and
  transcription facts derive from Angelotti et al., *Complete virtual unwrapping and reading of
  a rolled Herculaneum papyrus* (<https://scrollprize.org/pdf/main.pdf>). Attribution required,
  **non-commercial use only**. This directory does not inherit the MIT grant.
