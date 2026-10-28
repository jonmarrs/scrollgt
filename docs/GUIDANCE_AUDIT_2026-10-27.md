# Benchmark guidance checked against shipped data

Checked October 27 Pacific / October 28 UTC, using the v0.9.1 source at
`0f3fa90890fe40bc48a1f0274154697d23c29f9a`, cached Python 3.10.21,
NumPy 2.2.6, SciPy 1.15.3 and scikit-learn 1.7.2. The
[raw record](evidence/guidance_audit_2026-10-27.json) preserves source and input
SHA-256 hashes, pixel counts, exact floor values and file sizes. The source
hashes match that commit. This is a CPU documentation check, with no model
inference, new quality gain or change to the metric definition or target data.

The flagship `scroll1_20231210121321` has 3,091,940 positive pixels out of
16,777,216 selected pixels, prevalence 0.1842939853668213. Both the constant
1.0 and constant 0.5 predictors produce:

| Metric | Value |
| --- | --- |
| ROC AUC | 0.5 |
| Average precision | 0.1842939853668213 |
| AP / prevalence | 1.0 |
| Threshold-swept F1 | 0.3112301297548824 |

The F1 equals `2p/(1+p)` within 1e-12. The legacy detector's published
ROC AUC 0.5176 and AP lift 1.0086 are model measurements, not constant floors.
Its historical row and every other model row remain unchanged. The existing
clean held-out students already exceed ROC AUC 0.60, so contributor instructions
now refer readers to the current model rows and computed floors.

The 63 committed fiber target files (`meta.json`, `mask.npz`, `skeleton.npz`)
total **38,789,957 bytes**. Across the ten 256³ targets, each directory holds
208,047–394,798 bytes; across the eleven 512³ targets, 2,184,875–3,725,580 bytes.
These are stored file sizes, not scorer memory. The README's memory guidance
now cites the separately published v0.9.1 connected-components measurements;
this audit does not repeat that corpus run or time all four floors.

Reproduce the constant floors and byte total from an existing configured clone:

```sh
python - <<'PY'
import json
from pathlib import Path
from scrollgt.score import load_target, ink_floors

gt, mask, _ = load_target('data/scroll1_20231210121321')
floors = ink_floors(gt, mask)
print(json.dumps({k: floors[k] for k in ('floor_all_positive', 'floor_constant')}, indent=2))
files = [p for target in Path('data').glob('fibers_*')
         for name in ('meta.json', 'mask.npz', 'skeleton.npz')
         for p in [target / name]]
print('fiber target bytes:', sum(p.stat().st_size for p in files))
PY
```

Use the published predictions and provenance when reproducing model rows.
The Docker image contains scoring code, targets and built-in floors; it does
not contain every entrant's model or prediction map. Fiber scorecards normally
read version-checked published floors; `--recompute-floors` requests fresh
computation. Ink and column scorecards compute their floors at scoring time.

## Fresh upstream annotation check

All **21 NML files** named by the targets' `ground_truth.nml` records were
freshly fetched from Dataset004. Every download matches its recorded SHA-256,
and every packed skeleton matches all nine arrays in the shipped NPZ.
This was checked both with `tools/update_fiber_ground_truth.py --verify-only`
and with an independent ElementTree decode of the XML: coordinates, edges,
per-fiber node and edge offsets, IDs, names, scale, origin and shape. Across the corpus,
that covers 467,440 nodes and 458,949 edges, including out-of-cube context
retained in the source annotations. There were no missing files or mismatches.

The [per-target record](evidence/nml_provenance_audit_2026-10-27.json) retains
the exact URLs, download sizes and hashes, CLI commands, source/input hashes
and comparison outcomes. Its [raw archive](evidence/nml_provenance_audit_2026-10-27_raw.zip)
contains the original audit harness and all 21 CLI logs. The check downloaded
102,933,318 bytes of annotations, with no CT, semantic volumes or model data.
The checked source and target bytes remained unchanged throughout verification.

To repeat the entire check, extract the archive outside the checkout and run
from an existing configured ScrollGT checkout, giving a new output directory
outside it:

```sh
python /path/to/extracted/audit_scrollgt_nml.py /path/to/new-nml-audit
```

The harness uses four download/verification workers and a 32 MiB per-file
bound. Each downloaded file must match the target's recorded hash before it
is parsed; each existing verification CLI has a 90-second timeout. It writes
an EXACT or FAILED report and retains failures. All 21 targets are required.
This confirms source identity and packaging fidelity. It does not independently
validate the annotators' fiber choices, their physical placement in the CT,
model training exposure or any model-quality claim. No targets were rewritten
and no prediction scores were recomputed in this annotation check.
