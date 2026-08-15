# ScrollGT baselines (v0.1)

> ## ⚠ CORRECTED 2026-08-07 — the held-out table below is REPLACED; the old one was invalid
>
> The held-out label was generated with a hardcoded `LEVEL0_SHAPE` belonging to a different
> segment, so its region crop was scaled wrongly and the label came out displaced and
> stretched (~1766 level-0 voxels off). The resulting "everything reads at chance held-out"
> headline was an artifact of our own registration, and is **retracted**.
>
> The held-out target has been re-registered and every row re-scored. Teacher-enrichment on
> the same convention went 1.68 → **6.01**, and the corrected rows appear below. Note that
> the enrichment gate *failed* on the bad label and we overrode it with a teacher-free gate,
> attributing the failure to a weak teacher. The gate was right.
>
> **`arm C + GT fine-tune` has been removed**, not re-scored: it was fine-tuned *on* the
> displaced label, so its published 0.531 measured nothing. It must be retrained.
>
> **Placement is now gated and measured per target** (agreement must peak within 48 level-2
> px of zero shift): held-out **32.0 px / 0.31 mm**, train-exposed **46.6 px / 0.45 mm**.
> Both pass; the train-exposed target clears by only 1.4 px, so read its rows with more
> caution. Its numbers are unchanged — it was never mismeasured, only unverified, and it was
> not affected by the `LEVEL0_SHAPE` bug. These offsets are the method's **resolution
> limit**, not pending bugs, and the threshold was not raised to accommodate either target.
> Column and fiber targets are unaffected — different ground truth, no registration bridge.
>
> Evidence + reproduction:
> [registration_offset_2026-08-07.md](https://github.com/jonmarrs/vesuvius-autoresearch/blob/main/reports/detector/registration_offset_2026-08-07.md).
> Surfaced by `erdpx` closing villa PR
> [#1280](https://github.com/ScrollPrize/villa/pull/1280).

All rows scored with `scrollgt score` semantics (`scrollgt.metrics.segmentation_metrics`,
all-valid mask) against the registered ground truth. Sources: the vesuvius-autoresearch
reports `registered_gt_validation.json`, `registered_gt_heldout_validation.json`,
`gt_finetune_heldout.json` (commit-tracked; see that repo for training details).

**Read the two tables together — that is the benchmark's point.** Train-exposed scores
run higher than held-out ones for every model, so high scores on `scroll1_20230702185753`
alone still mean train-region fit rather than reading. What the corrected numbers no longer
support is the stronger claim we previously made — that held-out performance *collapses to
chance*. It does not; it degrades.

## Target `scroll1_20230702185753` (TRAIN-EXPOSED; **NON-SCORING as of 2026-08-14**)

> **Record, not a bar to beat.** `scrollgt score` refuses this target. Local placement error
> on this segment reaches ~1.9x the 512 um prize analysis window, so a model can be scored
> against ground truth from a different part of the sheet. The rows below are retained
> because the train-vs-held-out contrast is the point they demonstrate; they are not a
> leaderboard. `--allow-non-scoring` reproduces them.

**⚠ Placement 46.6 level-2 px (0.45 mm) — passes the 48 px gate by only 1.4 px, and the
global figure is optimistic: per-768px-tile scatter is sd 26.8 (dy) / 33.0 (dx) with the
worst tile ~100 px (~0.96 mm).** The field is non-rigid, so no correction fixes it. **Treat
these rows as indicative only.** Numbers are unchanged from the 2026-07 release; this
target was not affected by the `LEVEL0_SHAPE` bug — it was never mismeasured, only
unverified.

| model | val_f1 | f1_at_0.5 | average_precision | ap_prevalence_lift | roc_auc |
|---|---|---|---|---|---|
| canon teacher (binarized release) | 0.4372 | 0.4372 | 0.2573 | 2.0048 | 0.7031 |
| legacy detector (not trained here) | 0.2275 | 0.1845 | 0.1201 | 0.9359 | 0.4858 |
| arm A (1-scroll student) | 0.4568 | 0.4568 | 0.4096 | 3.1914 | 0.7941 |
| arm B (2-scroll student) | 0.4401 | 0.4401 | 0.3898 | 3.0374 | 0.7807 |
| arm C (3-scroll student) | 0.4675 | 0.4675 | 0.4222 | 3.2898 | 0.7995 |

Caveat (binary vs continuous): the canon teacher is a BINARY map; ROC-AUC/AP structurally
understate it relative to continuous probability maps. The fair teacher-vs-student
comparison is `f1_at_0.5`.

## Target `scroll1_20231210121321` (HELD-OUT — the flagship)

**Re-registered and re-scored 2026-08-07.** Registration: enrichment 6.01 (decisive vs
1.77/1.84/1.61), residual 7.95vx, periodicity 0.867.

| model | val_f1 | f1_at_0.5 | average_precision | ap_prevalence_lift | roc_auc |
|---|---|---|---|---|---|
| canon teacher (binarized release) | 0.5718 | 0.5718 | 0.3970 | 2.1542 | 0.7526 |
| legacy detector | 0.3112 | 0.2794 | 0.1859 | 1.0086 | 0.5176 |
| arm A (1-scroll student) | 0.5014 | 0.5013 | 0.4924 | 2.6716 | 0.7716 |
| arm B (2-scroll student) | 0.4404 | 0.4375 | 0.4309 | 2.3382 | 0.7305 |
| arm C (3-scroll student) | 0.4656 | 0.4549 | 0.4496 | 2.4397 | 0.7462 |

Withdrawn 2026-07 values, for the record: canon teacher 0.5632, arm A 0.5626,
arm B 0.5531, arm C 0.5576, legacy 0.5006 roc_auc — all reported as chance.

Arms B and C never saw this segment, and read it at roc_auc 0.73–0.75 with AP-lift
2.3–2.4 against an all-positive floor of 0.518 / 1.009. That is genuine held-out
generalization; the previous release said it was chance.

Metric note: at this region's ink prevalence (~0.18) the trivial all-positive predictor
already scores F1 ≈ 0.31, so `val_f1` is degenerate here; the robust reads are
`ap_prevalence_lift` and `roc_auc`, where the all-positive floor sits at 1.009 / 0.518.

## Target `scroll1_20230702185753_y7000_x4000` (v0.1.1)

Second region of the train-exposed segment; orientation DOUBLE-validated (enrichment
probe 3.13 + independent surface-NCC 0.28 — see meta.json). **Exposure per row is
stated explicitly — read this table as a demonstration of what exposure does.**

| model | exposure on THIS region | val_f1 | f1_at_0.5 | average_precision | ap_prevalence_lift | roc_auc |
|---|---|---|---|---|---|---|
| canon teacher (binarized release) | — (it *is* the reference model) | 0.4627 | 0.4627 | 0.2860 | 2.2425 | 0.7259 |
| legacy detector | **clean** | 0.2192 | 0.1632 | 0.1150 | 0.9391 | 0.4802 |
| arm A (1-scroll student) | teacher-supervised here | 0.4873 | 0.4800 | 0.4327 | 3.5322 | 0.8367 |
| arm B (2-scroll student) | teacher-supervised here | 0.3930 | 0.3930 | 0.3425 | 2.7955 | 0.7680 |
| arm C (3-scroll student) | teacher-supervised here | 0.4216 | 0.4204 | 0.3767 | 3.0752 | 0.7817 |
| arm C + GT fine-tune | **GT-supervised here (its training region)** | 0.7343 | 0.7019 | 0.7919 | 6.4643 | **0.9538** |

**The exhibit — ⚠ withdrawn 2026-08-07.** This previously read: `arm C + GT fine-tune`
scores ROC 0.9538 on its own training region and 0.5308 on the held-out target, a
0.42-ROC fit-vs-reading gap. The held-out half of that comparison was measured against the
misregistered label, and the model itself was fine-tuned on a displaced label, so the gap
is not interpretable. The model must be retrained on the corrected GT before the exhibit
can be restated. The 0.9538 is retained only as evidence that this region's labels are
learnable signal — note that a model can fit displaced labels perfectly well, so it says
nothing about placement.

## A target we did NOT ship (and why)

A fourth registered region (`20231005123336_y4000_x2500`) is **withheld** — still the right
call, but **the reason we published was wrong**.

We said the canon teacher was chance-quality on that segment (enrichment ≈ 1 for all four
orientation candidates, 0.79–1.02) so the label's orientation could not be verified. That
enrichment collapse was **our own bug**: `gt_register.py` carried a second hardcoded
level-0 shape, and this segment's true shape is 34880×97280 against the assumed
50600×36400 — a 167% x-scale error that scattered the label. Re-registered with the fix,
**teacher-enrichment is 4.88**, decisively orientating the label. The teacher was never
chance-quality here; our registration was broken.

It stays withheld on a properly measured criterion instead: **placement 55.1 level-2 px
(0.53 mm), over the 48 px gate.** Its sibling region `20231005123336_y7000_x4000` is
dropped earlier still, at prep — periodicity 0.556 and a registered ink fraction of
0.0005, i.e. essentially no ink lands in it.

The correction matters beyond this one target: three separate times now, a gate or
diagnostic fired correctly and we attributed the failure to the data rather than to our
code. See the upstream report for the pattern.

**This region cannot be replaced.** Withholding it leaves the pixel family at one scoreable
target, and as of 2026-08-15 there is no fourth candidate. Six Scroll-1 segments carry a
2023 hand ink label. Three — `20230820203112`, `20230826170124`, `20230903193206` — are
absent from the open data entirely, so there is no geometry to register a label onto. The
three that remain are accounted for one by one: `20230702185753` is poorly placed in both
of its regions, `20231005123336` is the segment withheld here, at 55.1 px over the gate,
and `20231210121321` is the held-out flagship already in service. The family is capped by
data availability, not by processing effort. See the
[availability survey](https://github.com/jonmarrs/vesuvius-autoresearch/blob/main/reports/detector/gt_training_data_exhaustion_2026-08-15.md).

## Column-level target `pherc1667_merged_columns` (v0.2) — anti-gaming floor

The first non-training-scroll target scores at COLUMN granularity (no pixel GT exists —
see the target's meta.json). Floor and ceiling, measured on the full grid
(22 columns scored, 17 gutters, 4 excluded for cross-strip-flagged neighbors):

| prediction | col_gutter_auc | col_gutter_pixel_auc | line_period_peak_mean |
|---|---|---|---|
| constant 0.5 | 0.5000 | 0.5000 | 0.0000 |
| uniform noise | 0.5784 | 0.5000 | 0.0622 |
| papyrus-mask copy | **0.5000** | 0.5000 | 0.0000 |
| geometry oracle (disclosed cheat: paints the target's own column boxes) | 1.0000 | 1.0000 | 0.0000 |

*(2026-07-19: `line_pitch_range` calibrated to [85, 160] from the measured figure-ink
line pitch — well-preserved columns cluster at 108–132 grid px, median 120; per-column
measured pitch now ships in columns.json. Floor rows re-measured under the calibrated
range; only noise periodicity moved, 0.0681 → 0.0622.)*

Read the floor rows before celebrating a score: predicting "papyrus everywhere" earns
exactly 0.5 (the gutters are papyrus too — that is the design), and the region-level AUC
has ~±0.08 statistical granularity at n = 18 text columns vs 17 gutters (the noise row).
The oracle row is the geometric ceiling and is trivially reachable by reading the public
columns.json — which is why column scores measure *consistency with the published
reading*, are necessary-not-sufficient evidence, and must be accompanied by the
prediction itself for visual review.

**Model rows — full band, all 22 columns** (rendered y=[100,1950) × 30097, stitched from
9 overlap-trimmed chunks; n = 18 text columns vs 17 gutters; maps published for review in
the provenance repo):

| model | 1667 exposure | col_gutter_auc | col_gutter_pixel_auc | line_period_peak_mean | traces_mean_ratio |
|---|---|---|---|---|---|
| arm C (3-scroll student) | **none** (held-out scroll) | 0.575 | 0.562 | 0.266 | 0.560 |
| legacy detector | **none** | 0.592 | 0.589 | 0.348 | 0.400 |
| *(uniform noise, same n)* | — | *0.585* | *0.500* | *0.062* | — |

**Both models are statistically at the noise floor** (0.575 / 0.592 vs the noise
realization 0.585), and both maps are texture without letterforms. *(2026-07-19: the two
cross-strip column bboxes were refined by local junction-window registration — the
inter-strip gaps close to <35 px and both widths land on the neighbor trend; uncertainty
is now measured at ±90 grid px rather than bounded at ±250. Rows re-scored against the
refined target; every shift ≤0.007, conclusion unchanged.)* The earlier
partial-extent rows (cols 17–19 only, n=3v2: arm C 0.667, legacy 0.000) are
**superseded** — their spread across zero-to-high was exactly the small-n quantization
artifact the caveat warned about, now demonstrated by measurement. The pixel AUCs
(0.56–0.59) sit slightly above chance, consistent with preservation-correlated texture
rather than reading (the traces_mean_ratio < 1 shows both models predict less on the
fragmentary traces columns — damage response, not text detection). Periodicity remains a
supporting diagnostic only: the prediction map is a required part of any submission.

## Fiber connectivity targets (v0.3)

Six 256³ cubes from villa's `fiber-skeletons` dataset, tolerance 2.0 voxels, every row scored
against the identical `fiber_hz_vt` mask shipped with each target. Source:
`reports/fiber_benchmark_all_cubes.json` in vesuvius-autoresearch.

**Connected components is a strong baseline, and our own tracer does not beat it** — losing on
raw ERL and on merge-penalized ERL, on every cube.

| cube | tracer ERL | cc ERL | tracer ERLpen | cc ERLpen | tracer coverage |
|---|---|---|---|---|---|
| s1_00497_01497_03997 | 26.6 | 197.1 | 23.2 | 37.1 | 0.623 |
| s1_00497_02497_02997 | 45.8 | 207.5 | 33.6 | 64.3 | 0.704 |
| s1_00997_02497_02997 | 36.3 | 195.8 | 29.8 | 56.5 | 0.605 |
| s1_08997_02997_02497 | 34.1 | 186.5 | 30.8 | 106.1 | 0.671 |
| s1_10997_02997_02997 | 37.4 | 194.1 | 34.2 | 57.7 | 0.616 |
| s5_03997_01497_03997 (cross-scroll) | 31.5 | 182.2 | 25.4 | 51.1 | 0.623 |

Fragmentation is the cause. The tracer finds the fibers — coverage 0.605–0.704 of ground-truth
length is claimed by *something* — but it cannot hold one identity along them, so its runs are
short and ERL is low. An earlier reading that the tracer was marginally ahead on the penalized
metric came from a 128³ sub-volume and **does not survive at full-cube scale**; it should not be
cited.

### What the floors establish

All four floors score **identical coverage and precision**, because those metrics are properties
of the shared fiber mask rather than of the instance labelling. Only ERL and the merge count
separate them. On `s1_00497_01497_03997`:

| labelling | ERL | ERLpen | coverage | precision | splits | merges | n inst |
|---|---|---|---|---|---|---|---|
| *oracle (disclosed)* | *258.27* | *239.46* | *1.0000* | *1.0000* | *14* | *7* | *87* |
| floor: one instance for everything | 199.18 | **0.00** | 0.9177 | 0.2194 | 243 | 86 | 1 |
| floor: connected components | 197.11 | 37.13 | 0.9177 | 0.2194 | 265 | 66 | 299 |
| floor: one instance per voxel | 0.94 | 0.94 | 0.9177 | 0.2194 | 23406 | 7 | 1005366 |
| floor: 50 random instances | 0.98 | **0.00** | 0.9177 | 0.2194 | 22937 | 4125 | 50 |

A benchmark reporting coverage and precision alone cannot distinguish a correct tracer from
`numpy.random`. Raw ERL alone is gameable too: labelling everything once scores 199.18 against an
oracle's 258.27 — within 23% — while its merge-penalized ERL is exactly 0.00. **Both ERL and the
merge count are required**, which is why `scrollgt score-fibers` never prints one without the
other. The claim is pinned by `tests/test_fiber_gaming.py`, so a change that breaks it fails CI.

## Submit a row

Score your model's probability map on the held-out target and open a PR/issue with the
scorecard JSON (`scrollgt score pred.png data/scroll1_20231210121321 --json-out card.json`).
State plainly whether your model saw segment 20231210121321 (or its 2023 labels) in
training. **Beating ROC 0.60 held-out, honestly, would be news.**

For fibers, score an instance labelling against any cube
(`scrollgt score-fibers labels.npy data/fibers_<cube> --json-out card.json`) and report both ERL
variants with the tolerance. **Beating connected components on both metrics would be news.**
