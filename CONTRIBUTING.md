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

## Add a target

New registered-GT targets are welcome but must clear the integrity bar: an **independent,
teacher-free** validation of the label's 2D orientation (teacher-enrichment is not enough
where the released prediction is weak). See the withheld-target discussion in
`baselines/BASELINES.md` and the orientation-validation methods in the source repo
([vesuvius-autoresearch](https://github.com/jonmarrs/vesuvius-autoresearch),
`reports/detector/orientation_probe_2026-07-11.md`). A gate-passing residual/periodicity is
necessary but not sufficient; a flat orientation profile means the target is withheld.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

CI runs the suite on Python 3.10–3.12 plus a CLI smoke on every push (see
`.github/workflows/ci.yml`). Keep `src/scrollgt/metrics.py` byte-identical to the source
repo's `detector/metrics.py` — the metric contract is the product, and the two must not
drift.
