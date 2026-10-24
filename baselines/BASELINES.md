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

It stays withheld on a properly measured criterion instead: **placement 57.5 level-2 px
(0.55 mm), over the 48 px gate.** Its sibling region `20231005123336_y7000_x4000` is
dropped earlier still, at prep — periodicity 0.556 and a registered ink fraction of
0.0005, i.e. essentially no ink lands in it.

That placement figure was corrected on 2026-08-15 from the 55.1 px first published here.
The original was a hand-run measurement that no committed code reproduces; the reproducible
figure, at the default search parameters, is 57.5 px. The agreement peak is broad — it sits
anywhere between roughly 56.5 and 57.5 px depending on the search window — so the offset is
determined to about ±1 px rather than to three significant figures, and should be quoted
that way. A broad peak is itself what a poorly placed label looks like; a correctly placed
one peaks sharply at zero. Every value in that range clears the 48 px gate by more than
eight px, so the decision to withhold is unchanged — only the number is.

The correction matters beyond this one target: three separate times now, a gate or
diagnostic fired correctly and we attributed the failure to the data rather than to our
code. See the upstream report for the pattern.

**This region cannot be replaced.** Withholding it leaves the pixel family at one scoreable
target, and as of 2026-08-15 there is no fourth candidate. Six Scroll-1 segments carry a
2023 hand ink label. Three — `20230820203112`, `20230826170124`, `20230903193206` — are
absent from the open data entirely, so there is no geometry to register a label onto. The
three that remain are accounted for one by one: `20230702185753` is poorly placed in both
of its regions, `20231005123336` is the segment withheld here, at 57.5 px over the gate,
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

Eleven cubes from villa's `fiber-skeletons` dataset, tolerance 2.0 voxels, every row scored
against the identical `fiber_hz_vt` mask shipped with each target: **eight at 256³** and
**three at 512³**. Splits: the five `s1_*` cubes are `primary`; the six `s5_*` cubes are
`cross_scroll` — three at each size class. Cross-scroll started this benchmark as a single
cube (`s5_03997_01497_03997`) and is now six. No cube in this expansion failed to export;
all eleven shipped with ground truth, a reference mask, and oracle/floor scores. Source:
`reports/fiber_benchmark_all_cubes.json` in vesuvius-autoresearch.

Every target's `meta.json` records `convention_check.landing_enrichment_vs_chance`: how many
times more often a ground-truth skeleton node lands on the scoring mask than chance density
would predict, which is what establishes that the skeleton and the mask share a coordinate
frame. Across the original six that enrichment sits at 13.6×–16.4×; across all eleven
targets it spans **9.9×–21.5×** (low: `s5_06994_00994_04994`; high:
`s5_07997_02997_05497`) — wider than the original six but every target still many times
chance, so the expansion did not introduce one with a silently broken registration.

**ERL does not compare across cube sizes and is never averaged across them.** ERL is
expected run length in voxels (`Σr²/L_gt`): a 512³ cube admits fibers up to twice as long per
axis as a 256³ cube, so its ERL runs roughly double for purely geometric reasons, not
because the tracer or the fibers are different. `aggregate_fiber_scores` in
`src/scrollgt/fibers/target.py` raises on a mixed size-class input rather than silently
producing a misleading mean. The tables below are split the same way, each with its own
oracle row so the ceiling sits beside the scores.

**What that costs the cross-scroll axis, stated plainly.** Six cross-scroll cubes are not six
comparable cross-scroll points. Per class the split is 5 primary + 3 cross_scroll at 256³ and
**0 primary + 3 cross_scroll at 512³** — and since scores are never compared across classes,
the three 512³ cross-scroll cubes have no same-scroll counterpart *in their own class* to
transfer from. The usable same-vs-cross comparison is therefore n=3 cross-scroll against n=5
primary, inside the 256³ class only; the 512³ cubes are absolute scores against their own
oracles, not a transfer measurement. Closing that gap needs 512³ Scroll-1 cubes, which the
villa dataset does not currently offer.

Every cube has tracer, connected-components (cc) and oracle rows, all scored with fiber scoring
version 3: the path walk of scrollgt#1, with ERL divided by the ground-truth length (0.7.0). The tracer is our own, at its published defaults: relink on,
no smoothing, no skipping, no seed suppression. It was re-measured on 2026-10-09 (vesuvius-autoresearch
finding 80, pre-registered). Its labellings, scored with version 1, reproduce the 0.3.2 tracer rows
exactly on the six cubes that had one, so the tracer itself has not changed; the scorer moved its
numbers. The source is recorded in each `meta.json` under `floors_external_source`.

**0.5.0:** three cubes (`s1_00497_02497_02997`, `s1_08997_02997_02497`, `s5_14997_01497_01497`) now carry
villa's current annotation (Dataset004), with 22–53% more fibers than the Dataset001 version shipped
before (corrected 2026-10-12 from "28–53%": `s5_14997_01497_01497` goes from 147 to 179, +22%). Their
oracle, floor and tracer rows are rescored against it (vesuvius-autoresearch finding 82, pre-registered). No ranking reversed. The reference mask, `fiber_hz_vt`, was trained on all 11 cubes in
their earlier versions, so every row built on it, connected components and our tracer included, is
scored on an in-sample segmentation.

**0.7.0, scoring version 3:** ERL and ERLpen now divide by the ground-truth length, not the covered length.
Under version 2 fiber a labelling missed never entered the score, so deleting short fragments raised it
(vesuvius-autoresearch finding 85). Version 3 = version 2 × coverage exactly. Every row below was re-scored:
floors from the shipped mask, tracer rows from the saved labellings. Each row's version-2 values reproduce the
0.6.0 row exactly and stay in `meta.json` as `erl_covered` and `erl_merge_penalized_covered`. Since the tracer
covers less fiber than cc, the change lowers its rows most (×0.26–0.68, against ×0.53–0.92 for cc).

**Connected components is far ahead on raw ERL, on every cube** (5.0–8.9×). **On merge-penalized
ERL the tracer is ahead on 4 of 11 cubes** (bold below): `s5_14997_01497_01497` at 256³, now a near-tie
(19.94 against 19.88), and all three 512³ cubes, where cc merges so many fibers that its penalized ERL falls
to 8.6–14.0. Until 0.3.2 this page said the tracer lost on both metrics on every cube scored. That was measured
with the order-dependent version-1 scorer, on six cubes, none of them 512³, and it does not hold under versions
2 or 3.

### 256³ cubes (n=8)

| cube | split | oracle ERL | tracer ERL | cc ERL | tracer ERLpen | cc ERLpen | tracer coverage |
|---|---|---|---|---|---|---|---|
| s1_00497_01497_03997 | primary | 247.84 | 22.05 | 183.97 | 19.22 | 34.04 | 0.6231 |
| s1_00497_02497_02997 | primary | 228.92 | 32.15 | 160.98 | 21.76 | 37.19 | 0.6136 |
| s1_00997_02497_02997 | primary | 243.56 | 29.55 | 169.31 | 23.44 | 50.02 | 0.6054 |
| s1_08997_02997_02497 | primary | 246.70 | 25.17 | 132.73 | 20.29 | 47.92 | 0.5598 |
| s1_10997_02997_02997 | primary | 244.10 | 29.69 | 171.71 | 27.24 | 49.80 | 0.6164 |
| s5_03997_01497_03997 | cross_scroll | 236.56 | 25.21 | 157.94 | 20.75 | 46.95 | 0.6233 |
| s5_07997_02997_05497 | cross_scroll | 250.55 | 32.42 | 187.17 | 29.48 | 79.76 | 0.6755 |
| s5_14997_01497_01497 | cross_scroll | 242.18 | 26.67 | 140.99 | **19.94** | 19.88 | 0.5932 |

### 512³ cubes (n=3)

| cube | split | oracle ERL | tracer ERL | cc ERL | tracer ERLpen | cc ERLpen | tracer coverage |
|---|---|---|---|---|---|---|---|
| s5_06494_01994_03994 | cross_scroll | 499.86 | 35.87 | 286.00 | **28.63** | 13.85 | 0.6716 |
| s5_06994_00994_04994 | cross_scroll | 507.25 | 24.56 | 199.38 | **19.66** | 8.57 | 0.5759 |
| s5_07994_01994_05494 | cross_scroll | 491.18 | 23.78 | 212.83 | **19.63** | 14.00 | 0.5742 |

Note the 512³ oracle ERLs (491.18–507.25) against the 256³ oracle ERLs (228.92–250.55): roughly
double, which is the geometric effect above, not a difference in fiber quality between
cubes.

Fragmentation limits the tracer. It finds the fibers (coverage 0.560–0.676 of ground-truth length is
claimed by *something*) but cannot hold one identity along them, so its runs are short and its raw
ERL is low. It merges less than cc, which is why it can lead on the penalized metric where cc merges
heavily. An earlier reading of the tracer ahead on the penalized metric came from a 128³ sub-volume
and should still not be cited. The full-cube result above is the one to use.

### Held-out cubes (0.6.0)

Ten cubes from villa's `fiber-skeletons` Dataset004 that the reference mask never trained on: absent from
Dataset002 and Dataset003, the releases matching `fiber_hz_vt`'s training fingerprint. Eight are from Scroll 3 at
512³, a scroll the mask never saw; two are new Scroll-5 cubes at 256³. Split `held_out`. Every row's version-2
values equal the row vesuvius-autoresearch finding 83 published (pre-registered, `reports/fiber_heldout_split.md`),
field for field; `tools/add_fiber_target.py` refuses to write a target otherwise. Frame check: nodes land on the mask at 6.4–13.2×
its density (in sample 9.9–21.5×) and on villa's semantic label at 1.000.

`cc coverage` is the mask's recall of the hand-traced fibers. It holds on Scroll 3 (median 0.826, in-sample 512³
0.822) and falls on the two Scroll-5 cubes (0.64 and 0.53, in-sample Scroll-5 256³ 0.82–0.91). Those two targets are built
correctly: no rigid shift within ±4 voxels gains more than 0.002, and lowering the threshold to 0.1 lifts coverage only
to 0.76 and 0.69. A fifth to a quarter of their fiber has no voxel at P ≥ 0.05 within 2 voxels (vesuvius-autoresearch
finding 84, pre-registered). The cause is unknown.

**Held-out cubes are never averaged with in-sample ones.** `aggregate_fiber_scores` refuses the mix.

#### Held out, 256³ (n=2)

| cube | split | oracle ERL | tracer ERL | cc ERL | tracer ERLpen | cc ERLpen | tracer coverage | cc coverage |
|---|---|---|---|---|---|---|---|---|
| s5_13497_01497_02497 | held_out | 205.49 | 11.08 | 53.29 | **10.17** | 4.73 | 0.3783 | 0.6427 |
| s5_16997_03997_04497 | held_out | 201.33 | 6.70 | 41.37 | **4.71** | 3.76 | 0.2549 | 0.5291 |

#### Held out, 512³ (n=8)

| cube | split | oracle ERL | tracer ERL | cc ERL | tracer ERLpen | cc ERLpen | tracer coverage | cc coverage |
|---|---|---|---|---|---|---|---|---|
| s3_00994_00494_01994 | held_out | 496.71 | 26.24 | 187.41 | **22.09** | 15.26 | 0.5328 | 0.8096 |
| s3_00994_01994_00494 | held_out | 524.89 | 30.89 | 200.48 | **24.76** | 23.81 | 0.5515 | 0.8097 |
| s3_00994_02494_01494 | held_out | 561.99 | 22.01 | 175.80 | **17.18** | 6.24 | 0.4998 | 0.7938 |
| s3_01994_01494_00994 | held_out | 445.11 | 24.12 | 195.64 | **21.27** | 15.48 | 0.5510 | 0.8424 |
| s3_01994_01994_01494 | held_out | 388.83 | 24.18 | 192.39 | **19.52** | 12.99 | 0.5823 | 0.8626 |
| s3_02994_02494_01994 | held_out | 502.18 | 37.66 | 266.75 | **29.97** | 14.83 | 0.6390 | 0.8677 |
| s3_03494_01494_00494 | held_out | 533.99 | 28.38 | 158.40 | 20.70 | 25.93 | 0.5020 | 0.7694 |
| s3_04494_00994_01494 | held_out | 479.98 | 24.15 | 212.27 | **17.82** | 8.58 | 0.5756 | 0.8573 |

On merge-penalized ERL our tracer leads connected components on nine of the ten held-out cubes; on
`s3_03494_01494_00494` connected components leads, 25.93 against 20.70. Up to 0.6.0 this page said all ten: under
version 2 the tracer led there 41.24 to 33.70, because its missed fiber did not count. On raw ERL connected
components leads on all ten (4.8–8.8×). On the two Scroll-5 cubes the tracer's lead comes from connected
components' collapse (its merge-penalized ERL falls to 3.8 and 4.7) rather than from the tracer, whose own
penalized ERL also falls below its in-sample range.

### villa's neural fiber tracer (0.8.0)

A third entrant, and the first that ScrollGT's authors did not write: villa's neural fiber tracer
([villa #2045](https://github.com/ScrollPrize/villa/pull/2045), `vesuvius.neural_tracing.tracers.fibers`) with its
default checkpoint, `scrollprize/fiber_single_small`, in volume mode at every default. We ran it on all 21 cubes and
scored it (vesuvius-autoresearch finding 91, pre-registered, `reports/neural_tracer_entrant.md`). Its rows are
`villa_neural_tracer`, and each `meta.json` records their provenance under `floors_external_source`: the villa commit,
the checkpoint's revision and sha256, and the four flags that read a 7.91 µm cube.

How its rows differ from the others:
* **Its labelling is its own.** It exports traced fibers as polylines, rasterised here as thin centrelines. Its
  presence input is the probability of `fiber_hz_vt`, the reference mask's model, but its voxels are not a
  relabelling of the shipped mask, so its coverage and precision are its own, not the floors'.
* **It trained on different data from the mask.** Read from the checkpoint: 563 Scroll 1 fibers and four automated
  fiber volumes (PHerc0175A, PHerc1447, PHerc0125, Paris 4 central), at 8.64–9.6 µm, and no Scroll 3 or Scroll 5.
  So the eight Scroll-3 cubes are a scroll that none of the three entrants trained on, and the in-sample cubes
  include its own training scroll, Scroll 1, in another scan.
* **Its runs are deterministic:** two repeated cubes reproduce exactly.

**On Scroll 3 it ranks first on merge-penalized ERL.** It is above our tracer on all 8 cubes (1.11–1.33×, median
25.30 against 20.98) and above connected components on all 8 (median 15.04). It covers less fiber: median coverage
0.480, against our tracer's 0.551 and connected components' 0.826. Connected components still leads on raw ERL on
every cube (4.8–7.7×), so it does not clear the bar of beating connected components on both metrics.

**Its Scroll-3 lead depends on running at the cubes' own 7.91 µm (0.8.1).** Resampled to its 9.6 µm training pitch, it
scores lower on all 21 cubes (merge-penalized ERL a median 0.63 of the published rows) and leads our tracer on none of
the 8 Scroll-3 cubes (vesuvius-autoresearch finding 92, pre-registered, `reports/neural_tracer_pitch.md`). The rows
here are the native run, the higher of the two on every cube, so they stay the better measure of the tool as run here.
Why it falls is not established: its settings are in grid voxels, so resampling also lengthens each in physical units
and shrinks each cube's trace grid, and interpolation smooths the CT. Resampling is not a 9.6 µm scan.

**On the other 13 cubes it trails our tracer on merge-penalized ERL,** including the five from Scroll 1, its own
training scroll: median 7.22 against 21.26 on the 8 in-sample 256³ cubes, and 9.33 against 19.66 on the 3 in-sample
512³ ones. On the two held-out Scroll-5 cubes its coverage is 0.134 and 0.096, and it leads connected components on
one of them (5.71 against 4.73). Pitch is not the main cause of the in-sample deficit: at 9.6 µm it still trails our
tracer on all 11 in-sample cubes (finding 92). Why the scrolls differ is otherwise untested: the scan (its Scroll 1 data
is the 2.4 µm rescan), the cube size, or annotation style. The in-sample cubes also favour our tracer and connected
components, which are built from a mask that trained on them.

**Its in-sample deficit is a tolerance-2 result on the 512³ cubes (0.8.2, finding 94): at tolerance 4 it leads our
tracer on all three in-sample 512³ cubes (median 49.50 against 43.69), and still trails on all eight in-sample 256³
cubes and both held-out Scroll-5 cubes. Its Scroll-3 lead holds at tolerance 4 and widens (a median 2.22 times our
tracer, against 1.26).** Finding 94 (vesuvius-autoresearch, `reports/tolerance4_rescore.md`) was pre-registered. The
tables here stay at tolerance 2, and those three cubes' `floors_external_source` give their tolerance-4 scores.

**On five in-sample cubes the hand-traced centrelines sit about a voxel off the CT's fibers (0.8.2).** villa's neural
tracer and #1934's generator agree on a shift of 1.01–1.53 voxels there. The CT alone puts the fibers' centres
0.51–0.93 voxels off in the same direction (vesuvius-autoresearch PR #157, exploratory). Moving the neural tracer by
the generator's shift raises its share within 2 voxels by 0.06–0.09 on these cubes: part of its in-sample gap, not all
of it. These five are where the offset is largest, by a cut chosen after seeing the values. The CT alone puts the
fibers' centres a smaller 0.18–0.48 voxels off on 7 of the 8 Scroll-3 cubes, so an offset of a fraction of a voxel may
be general. No row changes. Each cube's values are in its `meta.json`, under `ground_truth.placement_note`.

**On the 256³ cubes this benchmark's protocol keeps it from seeding near the faces (0.8.3, finding 96): it drops any
seed whose 77-voxel CT window crosses the edge of the CT it is given, and it is given the cube alone.** On all 21
cubes its closest traced seed lay 38 voxels from a face (vesuvius-autoresearch finding 96, exploratory,
`reports/neural_tracer_boundary.md`). A seed's window fits in 0.348 of a 256³ cube and 0.618 of a 512³ one.
* **Seeds dropped:** 0.54–0.70 of its seed dots on the ten 256³ cubes, against 0.26–0.36 on the eleven 512³ ones.
* **Output:** 0.28–0.78 of the ground-truth length on the 256³ cubes, against 0.83–1.43.
* **Size, not scroll:** both separate by cube size across scrolls, while our tracer's coverage does not (0.255–0.675 on
  256³ cubes, 0.500–0.672 on 512³ ones). Ten of the 13 cubes on which it trails our tracer are 256³.

**Given the scan around each 256³ cube, it keeps its seeds and about doubles its coverage** (vesuvius-autoresearch
finding 97, pre-registered, `reports/neural_tracer_ct_context.md`). The run gave it 128 voxels of real CT around each
cube, with the box still the cube and everything else as here.
* **Seeds dropped:** 0.002–0.010 on the eight in-sample cubes, and 0.266 and 0.353 on the two held-out ones.
* **Coverage:** 1.76–2.78 times its row here, on all ten.
* **Against our tracer:** at tolerance 4 it leads on 8 of 10. At tolerance 2 it leads on both held-out cubes. It still
  trails on the eight in-sample ones, but at a median 0.60–0.65 of our tracer rather than 0.27–0.36.

So on the 256³ cubes its rows here understate it as it would run on a scroll. Our tracer's rows and the reference mask
also come from the cube alone. No row changes. Each 256³ cube's `floors_external_source` gives its own values, with
and without the scan around it.

One checkpoint, in volume mode, at its defaults: VC3D's seed mode is not tested. Bold: its merge-penalized ERL
leads both our tracer and connected components.

#### villa's neural tracer, in sample, 256³ (n=8)

| cube | split | neural ERL | cc ERL | neural ERLpen | tracer ERLpen | cc ERLpen | neural coverage | tracer coverage |
|---|---|---|---|---|---|---|---|---|
| s1_00497_01497_03997 | primary | 5.73 | 183.97 | 5.20 | 19.22 | 34.04 | 0.2119 | 0.6231 |
| s1_00497_02497_02997 | primary | 11.86 | 160.98 | 7.77 | 21.76 | 37.19 | 0.2682 | 0.6136 |
| s1_00997_02497_02997 | primary | 9.75 | 169.31 | 8.62 | 23.44 | 50.02 | 0.2418 | 0.6054 |
| s1_08997_02997_02497 | primary | 6.73 | 132.73 | 5.88 | 20.29 | 47.92 | 0.2422 | 0.5598 |
| s1_10997_02997_02997 | primary | 11.68 | 171.71 | 11.64 | 27.24 | 49.80 | 0.2628 | 0.6164 |
| s5_03997_01497_03997 | cross_scroll | 4.80 | 157.94 | 4.80 | 20.75 | 46.95 | 0.1564 | 0.6233 |
| s5_07997_02997_05497 | cross_scroll | 7.89 | 187.17 | 7.82 | 29.48 | 79.76 | 0.2354 | 0.6755 |
| s5_14997_01497_01497 | cross_scroll | 7.81 | 140.99 | 6.67 | 19.94 | 19.88 | 0.2384 | 0.5932 |

#### villa's neural tracer, in sample, 512³ (n=3)

| cube | split | neural ERL | cc ERL | neural ERLpen | tracer ERLpen | cc ERLpen | neural coverage | tracer coverage |
|---|---|---|---|---|---|---|---|---|
| s5_06494_01994_03994 | cross_scroll | 14.45 | 286.00 | 12.66 | 28.63 | 13.85 | 0.3807 | 0.6716 |
| s5_06994_00994_04994 | cross_scroll | 10.50 | 199.38 | 8.46 | 19.66 | 8.57 | 0.3332 | 0.5759 |
| s5_07994_01994_05494 | cross_scroll | 10.93 | 212.83 | 9.33 | 19.63 | 14.00 | 0.3422 | 0.5742 |

#### villa's neural tracer, held out, 256³ (n=2)

| cube | split | neural ERL | cc ERL | neural ERLpen | tracer ERLpen | cc ERLpen | neural coverage | tracer coverage |
|---|---|---|---|---|---|---|---|---|
| s5_13497_01497_02497 | held_out | 5.91 | 53.29 | 5.71 | 10.17 | 4.73 | 0.1340 | 0.3783 |
| s5_16997_03997_04497 | held_out | 4.03 | 41.37 | 2.80 | 4.71 | 3.76 | 0.0964 | 0.2549 |

#### villa's neural tracer, held out, 512³ (n=8)

| cube | split | neural ERL | cc ERL | neural ERLpen | tracer ERLpen | cc ERLpen | neural coverage | tracer coverage |
|---|---|---|---|---|---|---|---|---|
| s3_00994_00494_01994 | held_out | 34.62 | 187.41 | **29.48** | 22.09 | 15.26 | 0.5414 | 0.5328 |
| s3_00994_01994_00494 | held_out | 38.69 | 200.48 | **32.70** | 24.76 | 23.81 | 0.5124 | 0.5515 |
| s3_00994_02494_01494 | held_out | 27.76 | 175.80 | **22.87** | 17.18 | 6.24 | 0.4419 | 0.4998 |
| s3_01994_01494_00994 | held_out | 25.94 | 195.64 | **23.69** | 21.27 | 15.48 | 0.4415 | 0.5510 |
| s3_01994_01994_01494 | held_out | 28.63 | 192.39 | **24.65** | 19.52 | 12.99 | 0.4616 | 0.5823 |
| s3_02994_02494_01994 | held_out | 39.01 | 266.75 | **33.64** | 29.97 | 14.83 | 0.5180 | 0.6390 |
| s3_03494_01494_00494 | held_out | 32.97 | 158.40 | **25.96** | 20.70 | 25.93 | 0.4608 | 0.5020 |
| s3_04494_00994_01494 | held_out | 27.41 | 212.27 | **21.35** | 17.82 | 8.58 | 0.4981 | 0.5756 |

### What the floors establish

All four floors score **identical coverage and precision**, because those metrics are properties
of the shared fiber mask rather than of the instance labelling. Only ERL and the merge count
separate them. On `s1_00497_01497_03997`:

| labelling | ERL | ERLpen | coverage | precision | splits | merges | n inst |
|---|---|---|---|---|---|---|---|
| *oracle (disclosed)* | *247.84* | *229.29* | *1.0000* | *1.0000* | *14* | *7* | *87* |
| floor: one instance for everything | 184.02 | **0.00** | 0.9177 | 0.2194 | 130 | 86 | 1 |
| floor: connected components | 183.97 | 34.04 | 0.9177 | 0.2194 | 131 | 66 | 299 |
| floor: one instance per voxel | 0.98 | 0.98 | 0.9177 | 0.2194 | 20984 | 7 | 1005366 |
| floor: 50 random instances | 1.01 | **0.00** | 0.9177 | 0.2194 | 20568 | 4125 | 50 |

A benchmark reporting coverage and precision alone cannot distinguish a correct tracer from
`numpy.random`. Raw ERL alone is gameable too: labelling everything once scores 184.02 against an
oracle's 247.84 — 74% of it — while its merge-penalized ERL is exactly 0.00. **Both ERL and the
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
