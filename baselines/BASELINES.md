# ScrollGT baselines (v0.1)

All rows scored with `scrollgt score` semantics (`scrollgt.metrics.segmentation_metrics`,
all-valid mask) against the registered ground truth. Sources: the vesuvius-autoresearch
reports `registered_gt_validation.json`, `registered_gt_heldout_validation.json`,
`gt_finetune_heldout.json` (commit-tracked; see that repo for training details).

**Read the two tables together — that is the benchmark's point.** On the train-exposed
target every distilled model looks strong; on the held-out target everything collapses to
near chance, including the released canon prediction itself. High scores on
`scroll1_20230702185753` alone mean train-region fit, not reading.

## Target `scroll1_20230702185753` (TRAIN-EXPOSED for the distilled rows; disclosed)

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

| model | val_f1 | f1_at_0.5 | average_precision | ap_prevalence_lift | roc_auc |
|---|---|---|---|---|---|
| canon teacher (binarized release) | 0.2950 | 0.2950 | 0.2102 | 1.1501 | 0.5632 |
| legacy detector | 0.3090 | 0.2668 | 0.1837 | 1.0052 | 0.5006 |
| arm A (1-scroll student) | 0.3107 | 0.2578 | 0.2198 | 1.2027 | 0.5626 |
| arm B (2-scroll student) | 0.3107 | 0.2431 | 0.2121 | 1.1609 | 0.5531 |
| arm C (3-scroll student) | 0.3098 | 0.2304 | 0.2130 | 1.1654 | 0.5576 |
| arm C + GT fine-tune (negative result) | 0.3090 | 0.1507 | 0.1991 | 1.0898 | 0.5308 |

Metric note: at this region's ink prevalence (~0.18) the trivial all-positive predictor
already scores F1 ≈ 0.31, so `val_f1` is degenerate here; the robust reads are
`ap_prevalence_lift` and `roc_auc` — all rows sit ≈ chance. The `arm C + GT fine-tune`
row is a documented negative: fine-tuning on registered GT *reduced* discrimination and
collapsed toward the trivial predictor.

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

**The exhibit:** `arm C + GT fine-tune` scores **ROC 0.9538 on its own training region**
and **0.5308 on the held-out target** (same model, table above). That 0.42-ROC gap is
what train-region fit looks like when an eval has a held-out surface — and why scores on
exposed regions must never be read as reading ability. (Secondary observations: the clean
legacy row sits at chance, and the 0.95 also confirms this region's registered labels are
learnable signal, not noise.)

## A target we did NOT ship (and why)

A fourth registered region (`20231005123336_y4000_x2500`) passed the residual and
periodicity gates but was **withheld**: the canon teacher is chance-quality on that
segment (enrichment ≈ 1 for ALL four orientation candidates: 0.79–1.02), so its 2D
orientation cannot currently be verified by any teacher-based or teacher-free check we
have (periodicity is flip-invariant). A benchmark target whose label orientation is
unverifiable is not a target. It will ship if/when an independent orientation check
exists. This is what the gates are for.

## Column-level target `pherc1667_merged_columns` (v0.2) — anti-gaming floor

The first non-training-scroll target scores at COLUMN granularity (no pixel GT exists —
see the target's meta.json). Floor and ceiling, measured on the full grid
(22 columns scored, 17 gutters, 4 excluded for cross-strip-flagged neighbors):

| prediction | col_gutter_auc | col_gutter_pixel_auc | line_period_peak_mean |
|---|---|---|---|
| constant 0.5 | 0.5000 | 0.5000 | 0.0000 |
| uniform noise | 0.5784 | 0.5000 | 0.0681 |
| papyrus-mask copy | **0.5000** | 0.5000 | 0.0000 |
| geometry oracle (disclosed cheat: paints the target's own column boxes) | 1.0000 | 1.0000 | 0.0000 |

Read the floor rows before celebrating a score: predicting "papyrus everywhere" earns
exactly 0.5 (the gutters are papyrus too — that is the design), and the region-level AUC
has ~±0.08 statistical granularity at n = 18 text columns vs 17 gutters (the noise row).
The oracle row is the geometric ceiling and is trivially reachable by reading the public
columns.json — which is why column scores measure *consistency with the published
reading*, are necessary-not-sufficient evidence, and must be accompanied by the
prediction itself for visual review.

**Model rows** (rendered cols 17–19 region, grid origin y=100 x=20800, 1710×3990 — a
partial extent, so n = 3 text columns vs 2 gutters and the region AUC is quantized to
sixths; both prediction maps are published for review in the provenance repo):

| model | 1667 exposure | col_gutter_auc | col_gutter_pixel_auc | line_period_peak_mean |
|---|---|---|---|---|
| arm C (3-scroll student) | **none** (held-out scroll) | 0.667 | 0.521 | 0.132 |
| legacy detector | **none** | 0.000 | 0.389 | 0.433 |

Both maps are texture without letterforms (visual review), consistent with every prior
cross-scroll result. Two honest readings of these rows: (1) arm C's 0.667 at n=3v2 is
one rank-step from chance — not evidence of reading; (2) legacy's periodicity 0.433 is a
**measured confound, not text**: its map shows broad horizontal banding (damage/fiber
following and inference tiling) whose pitch falls inside the line-pitch range. This is
exactly why `line_period_peak_mean` is a supporting diagnostic and the prediction map is
a required part of any submission — a periodicity score alone can be an artifact.

## Submit a row

Score your model's probability map on the held-out target and open a PR/issue with the
scorecard JSON (`scrollgt score pred.png data/scroll1_20231210121321 --json-out card.json`).
State plainly whether your model saw segment 20231210121321 (or its 2023 labels) in
training. **Beating ROC 0.60 held-out, honestly, would be news.**
