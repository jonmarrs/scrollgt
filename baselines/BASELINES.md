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

## Submit a row

Score your model's probability map on the held-out target and open a PR/issue with the
scorecard JSON (`scrollgt score pred.png data/scroll1_20231210121321 --json-out card.json`).
State plainly whether your model saw segment 20231210121321 (or its 2023 labels) in
training. **Beating ROC 0.60 held-out, honestly, would be news.**
