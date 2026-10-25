# Reproducing fiber scores within workstation memory

The fiber scorer uses a Euclidean distance transform (EDT) for nearest-label
tolerance growth and for precision. SciPy's whole-volume temporary arrays can
exceed a small machine's memory before a 512³ score is produced.

The scorer now processes 64 z planes at a time with a `ceil(tolerance)` halo
on either side. Every foreground voxel within the requested tolerance is still
available to SciPy, including its nearest-label tie handling. Halo results are
discarded. Windows with no foreground are skipped so an all-background EDT
cannot invent an outside seed. Both tolerance growth and precision use this
path. Scoring version remains **3**; no metric, model, target or published row
is changed. Infinite/NaN radii retain the original whole-volume path.

## Real-data check

Baseline: tag **v0.9.0**, commit
`7ddff50b2bde41fe1bf1b05621c2eb299816bfbc`. The same CLI, target files and
runtime were used before and after the change, with one run per case.
Each process had a **7 GiB address-space limit** (`RLIMIT_AS`); this is not a
7 GiB physical-RAM measurement or an OOM-killer result.

| Target | Baseline | Slab scorer |
| --- | --- | --- |
| Scroll 5 `s5_13497_01497_02497_256` | completes, 9.04 s, 1.06 GiB peak RSS | completes, 7.04 s, 0.57 GiB peak RSS |
| Scroll 3 `s3_00994_00494_01994_512` | `MemoryError` after 33.55 s; 5.50 GiB peak RSS before failure | completes, 54.93 s, 2.57 GiB peak RSS |

Both successful patched scorecards exactly equal their target's published
connected-components row, across all 15 serialized fields. The small cube's
grown label volume is also voxel-identical to the original full transform
(SHA-256 `87ef8cbdca64f5dabfd571eef13a780cc9612074cdbe40cbd52613f1c317f1f7`),
and every unrounded score field, including the run-length list, is identical.
The failing large-volume baseline cannot supply a complete scorecard under
this limit; its comparison is to the released row, not a completed baseline
run in this environment.

| Patched row | Coverage | ERL | Merge-penalized ERL | Precision |
| --- | ---: | ---: | ---: | ---: |
| Scroll 5, 256³ | 0.6427 | 53.29 | 4.73 | 0.1420 |
| Scroll 3, 512³ | 0.8096 | 187.41 | 15.26 | 0.2058 |

These two held-out targets differ in scroll **and cube size**; this walkthrough
does not compare their quality or average their ERLs. Held out refers to the
reference mask's training split, not every possible entrant.

[Raw measurements, scorecards and hashes](evidence/fiber_memory_2026-10-25.json)
include the input and scorer hashes. Runtime: Linux x86-64, Python 3.10.21,
NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, one OpenMP/OpenBLAS/MKL thread.
Wall time uses `time.perf_counter`; peak RSS is Linux
`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`, from a fresh measurement
process for each scoring command. These are single-run operational checks,
not a general throughput benchmark. No packages were installed; fresh sparse
checkouts used an existing cached Python environment. GPU tracing, model
inference and other operating systems were not tested here.

## Run the example

From a checkout with its Python dependencies already available:

```sh
PYTHONPATH=src python -m scrollgt.cli score-fibers --floor connected_components \
  data/fibers_s5_13497_01497_02497_256 --json-out scroll5.json
PYTHONPATH=src python -m scrollgt.cli score-fibers --floor connected_components \
  data/fibers_s3_00994_00494_01994_512 --json-out scroll3.json
```

The chosen prediction is recomputed from each shipped mask. The other baseline
rows on the card come from published metadata; these commands do not recompute
all four floors. A fresh installation and download time are outside the timing
above.

To enforce the same budget and collect a per-process measurement on Linux,
run one target per invocation of this wrapper with `PYTHONPATH=src` and the
three thread variables set to `1`:

```python
import resource, subprocess, sys, time

def budget():
    limit = 7 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

started = time.perf_counter()
run = subprocess.run([sys.executable, "-m", "scrollgt.cli", "score-fibers",
                      "--floor", "connected_components", sys.argv[1]],
                     preexec_fn=budget, timeout=300)
print(run.returncode, time.perf_counter() - started,
      resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
```

Regression checks include random label ties, slab boundaries, fractional and
large radii, empty halos, large unsigned labels, unchanged input arrays, bounded
EDT calls in both scoring paths, and equality of all raw score fields.

```sh
PYTHONPATH=src python -m pytest -q tests/test_fiber_distance_blocks.py \
  tests/test_fiber_eval_trace.py
```
