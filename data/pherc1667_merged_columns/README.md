# How to predict on this target

There are no surface volumes for the merged segment in the bucket — render them
yourself with the [full-3D-validated renderer](https://github.com/jonmarrs/vesuvius-autoresearch/blob/main/docs/SURFACE_RENDERER.md)
(clean-triple NCC 0.78 on this scroll; all 26 depth layers match the released w011
stack at NCC 0.84–0.89 with exact depth scaling; default `--sign 1` is the released
convention). Predict a probability map at grid resolution over your rendered extent,
then score with `--origin Y X` set to the extent's grid top-left.

**Full band (all 22 columns; ~55 Mpx, plan hours of S3 time):**

```bash
uv run python -m repro.sota_data.merged_fullband_render   # 9 resumable chunks
# ... run your model per chunk, stitch (see merged_fullband_score.py), then:
scrollgt score-columns pred.npy data/pherc1667_merged_columns --origin 100 0
```

**Single column (with 50 px margins; ~15–30 min each):** `--region Y0 X0 H W` in grid px:

```bash
uv run python -m repro.sota_data.render_cli \
  --tifxyz vesuvius-challenge-open-data/PHerc1667/segments/20260612121456-w011_20260108140509268_merged_v4_flatboi_straightened_v4/mesh/20260612121456-on-20251217075048-2.399um.tifxyz \
  --volume vesuvius-challenge-open-data/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr \
  --out local_data/rendered --frag-id colN --region <Y0> <X0> <H> <W> --level 2
```

| col | status | `--region Y0 X0 H W` | `--origin Y X` |
|---|---|---|---|
| 1 | traces | `439 0 1253 938` | `--origin 439 0` |
| 2 | traces | `429 1017 1487 1181` | `--origin 429 1017` |
| 3 | traces | `410 2300 1576 1181` | `--origin 410 2300` |
| 4 | traces | `467 3583 1471 1162` | `--origin 467 3583` |
| 5 | text | `373 4847 1596 1162` | `--origin 373 4847` |
| 6 | text | `415 6111 1487 1167` | `--origin 415 6111` |
| 7 | text | `326 7380 1665 1163` | `--origin 326 7380` |
| 8 | text | `288 8584 1572 1162` | `--origin 288 8584` |
| 9 ⚑ | text | `514 9839 969 1195` | `--origin 514 9839` |
| 10 | text | `303 11169 1580 1204` | `--origin 303 11169` |
| 11 | text | `307 12532 1510 1204` | `--origin 307 12532` |
| 12 | text | `241 13900 1661 1199` | `--origin 241 13900` |
| 13 | text | `218 15145 1627 1200` | `--origin 218 15145` |
| 14 | text | `185 16508 1543 1204` | `--origin 185 16508` |
| 15 | text | `86 17876 1613 1200` | `--origin 86 17876` |
| 16 ⚑ | text | `63 19305 1650 1232` | `--origin 63 19305` |
| 17 | text | `58 20752 1604 1289` | `--origin 58 20752` |
| 18 | text | `67 22148 1694 1294` | `--origin 67 22148` |
| 19 | text | `86 23544 1397 1294` | `--origin 86 23544` |
| 20 | text | `95 25001 1586 1350` | `--origin 95 25001` |
| 21 | text | `58 26453 1787 1350` | `--origin 58 26453` |
| 22 | text | `72 27910 1764 1346` | `--origin 72 27910` |

⚑ = cross-strip bbox (edges ±90 grid px, measured). A single-column extent scores the
column's periodicity but no gutters (n too small for `col_gutter_auc`) — meaningful
rows need multi-column extents; the full band is the definitive protocol. Include your
prediction map with any submitted row.

## ⚠ 2026-08-27 — per-strip scale discrepancy, unresolved

Column width steps at the figure-strip boundaries: 51.3 ± 0.4 mm (cols 2-8), 52.9 ± 0.1 mm
(10-15), 58.6 ± 1.4 mm (17-22). Strip 3 is 1.14x strip 1, permutation p < 0.0001. The step aligns
with where the source figure was cropped, not with anything the scribe did, so per-column
boundaries in cols 17-22 carry roughly 14% width uncertainty.

The tiling closure quoted in `meta.json` cannot rule this out: it constrains total length, so
per-strip errors in opposite directions cancel.

The global transform is unaffected. The valid mask inside the 22 boxes totals 867 cm2 against the
~860 cm2 of preserved writing surface in the published reading, a 1.01x match on a constraint that
is quadratic in scale, and the 22-column count matches exactly.

Region-level scoring over the whole target remains usable. Per-column comparisons in cols 17-22 do
not. Analysis: github.com/jonmarrs/vesuvius-autoresearch, `reports/column_width_by_strip.txt`.

