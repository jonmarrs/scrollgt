# Contributing to ScrollGT

The most useful contribution is **an external score on a held-out target** — a row from a
model that ScrollGT's authors did not train. That is the whole point of the benchmark.

## Submit a leaderboard row

1. Produce a probability map over **exactly** the target region. Open the target's
   `meta.json` for the SOTA S3 zarr path, pyramid level, and `y0/x0/size` (e.g.
   `data/scroll1_20231210121321/meta.json`). Save your prediction as an 8-bit PNG
   (interpreted as `prob = pixel/255`) or a `.npy` float array in `[0, 1]`, matching the
   ground-truth tile's height and width.

2. Score it:
   ```bash
   scrollgt score my_prediction.png data/scroll1_20231210121321 --json-out card.json
   ```

3. Open an issue or PR titled `leaderboard: <model name> on <target>` containing:
   - the `card.json` scorecard,
   - one sentence on the model,
   - **an explicit statement of exposure**: did the model see this segment (or its 2023
     labels) in training? Held-out rows are the ones that matter.

We add verified rows to `baselines/BASELINES.md`. **Beating ROC-AUC 0.60 on the held-out
target, honestly, would be news.**

## Submit a row on the column target (`pherc1667_merged_columns`)

The PHerc-1667 target has **no pixel ground truth** — scoring measures *consistency with
the published reading* at column granularity, never letter accuracy. The flow differs:

1. The bucket ships no surface volume for the merged segment, so render one first with the
   [full-3D-validated renderer](https://github.com/jonmarrs/vesuvius-autoresearch/blob/main/docs/SURFACE_RENDERER.md).
   `data/pherc1667_merged_columns/README.md` has copy-paste `--region` commands per column
   and the full-band protocol.
2. Predict a probability map at grid resolution over your rendered extent, then:
   ```bash
   scrollgt score-columns my_pred.npy data/pherc1667_merged_columns --origin <Y> <X>
   ```
   where `--origin` is your extent's grid top-left. Multi-column extents are required for a
   meaningful `col_gutter_auc` (a single column has no gutters); the full band (all 22
   columns, n=18v17) is the definitive protocol.
3. **Include your prediction map with the row.** Because the column layout is public, a
   high `col_gutter_auc` proves layout-consistency, not reading — the map is a required
   part of the evidence, and `line_period_peak_mean` is a supporting diagnostic only (a
   periodicity score alone can be an inference-banding artifact, as the legacy-detector
   baseline shows). State your PHerc-1667 exposure explicitly.

## Submit a row on a fiber connectivity target (`fibers_*`)

These targets score an **instance labelling**, not a probability map: which voxels belong to
which fiber. Everything needed ships in the repo, so no GPU, model download, or network access
is involved.

1. Produce `labels.npy` — a cube-shaped integer array (256³), `0` = background, one distinct id
   per predicted fiber instance. The reference fiber mask is in the target's `mask.npz` if you
   want to label that rather than segment your own; every published row is scored against that
   same mask, so differences come from the labelling.
2. Score it:
   ```bash
   scrollgt score-fibers labels.npy data/fibers_<cube> --json-out card.json
   ```
   Add `--recompute-floors` to verify the published floors from the shipped mask yourself
   (~50 s per cube) rather than reading them from `meta.json`.
3. **Report both ERL variants together, with the tolerance.** Raw ERL alone is gameable — a
   single instance covering the whole cube scores within 23% of the oracle while its
   merge-penalized ERL is exactly 0.00 — so a row quoting one number without the other will be
   sent back. Splits and merges are reported separately and must never be summed.
4. Score all six cubes, and report `s5_03997_01497_03997_256` separately as the cross-scroll
   split. Beating connected components on **both** metrics is the bar; our own tracer does not
   clear it (see `baselines/BASELINES.md`).

## Add a target

New registered-GT targets are welcome but must clear **two independent bars**, and both are
binding — a target that clears one and fails the other does not ship.

1. **Orientation.** An **independent, teacher-free** validation of the label's 2D
   orientation (teacher-enrichment is not enough where the released prediction is weak).
   See the orientation-validation methods in the source repo
   ([vesuvius-autoresearch](https://github.com/jonmarrs/vesuvius-autoresearch),
   `reports/detector/orientation_probe_2026-07-11.md`). A gate-passing
   residual/periodicity is necessary but not sufficient; a flat orientation profile means
   the target is withheld.
2. **Placement.** Global placement error at or under the **48 px** gate (level-2 px,
   ~0.46 mm) is **necessary but not sufficient** — `20230702185753_y4000_x2500` clears the
   gate at 46.6 px and is still non-scoring, because local error within a single 64 px
   window reaches ~1.9 windows. So report placement per-target alongside per-768px-tile
   scatter: the field is non-rigid and the global figure is optimistic. This is the
   criterion that currently binds: `20231005123336_y4000_x2500` is withheld at 55.1 px
   despite a decisively validated orientation, and both `20230702185753` regions are
   non-scoring on local error. On `20230702185753` the cause is measured — cross-scan
   disagreement between the 2023 and 2026 segmentations of that sheet, which re-running the
   registration does not fix. Expect to have to rule that out for a new target rather than
   assume a correctable offset.

See the withheld-target discussion in `baselines/BASELINES.md` for a worked example of each.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

CI runs the suite on Python 3.10–3.12 plus a CLI smoke on every push (see
`.github/workflows/ci.yml`). Keep `src/scrollgt/metrics.py` byte-identical to the source
repo's `detector/metrics.py` — the metric contract is the product, and the two must not
drift.
