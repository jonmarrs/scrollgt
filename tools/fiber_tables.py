"""Maintainer tool: print the per-size-class fiber tables for baselines/BASELINES.md from the shipped meta.json files.

Generated, never hand-copied, so a published number cannot drift from the data that tests reproduce.

Usage: python tools/fiber_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(size: int, held_out: bool = False) -> str:
    """One size class, in-sample targets or held-out ones. Held-out tables add cc coverage, the mask's recall there."""
    cov = " cc coverage |" if held_out else ""
    out = ["| cube | split | oracle ERL | tracer ERL | cc ERL | tracer ERLpen | cc ERLpen | tracer coverage |" + cov,
           "|---|---|---|---|---|---|---|---|" + ("---|" if held_out else "")]  # fmt: skip
    for p in sorted((ROOT / "data").glob(f"fibers_*_{size}")):
        m = json.loads((p / "meta.json").read_text())
        if (m["split"] == "held_out") != held_out:
            continue
        f = m["floors"]
        o, c, t = f["oracle"], f["floor_connected_components"], f["tracer_strict_relink"]
        pen = f"**{t['erl_merge_penalized']:.2f}**" if t["erl_merge_penalized"] > c["erl_merge_penalized"] else f"{t['erl_merge_penalized']:.2f}"
        cube = p.name.removeprefix("fibers_").removesuffix(f"_{size}")
        out.append(f"| {cube} | {m['split']} | {o['erl']:.2f} | {t['erl']:.2f} | {c['erl']:.2f} | {pen} | "
                   f"{c['erl_merge_penalized']:.2f} | {t['coverage']:.4f} |" + (f" {c['coverage']:.4f} |" if held_out else ""))  # fmt: skip
    return "\n".join(out)


NEURAL = "villa_neural_tracer"
SCAN_CONTEXT = "villa_neural_tracer_scan_context"  # 0.9.0: the same tracer with the scan around each cube


def entrant_table(size: int, held_out: bool = False, row: str = NEURAL) -> str:
    """An external entrant beside our tracer and cc, one size class and split. Bold: its ERLpen leads both."""
    out = ["| cube | split | neural ERL | cc ERL | neural ERLpen | tracer ERLpen | cc ERLpen | neural coverage | tracer coverage |",
           "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for p in sorted((ROOT / "data").glob(f"fibers_*_{size}")):
        m = json.loads((p / "meta.json").read_text())
        if (m["split"] == "held_out") != held_out:
            continue
        f = m["floors"]
        n, t, c = f[row], f["tracer_strict_relink"], f["floor_connected_components"]
        lead = n["erl_merge_penalized"] > max(t["erl_merge_penalized"], c["erl_merge_penalized"])
        pen = f"**{n['erl_merge_penalized']:.2f}**" if lead else f"{n['erl_merge_penalized']:.2f}"
        cube = p.name.removeprefix("fibers_").removesuffix(f"_{size}")
        out.append(f"| {cube} | {m['split']} | {n['erl']:.2f} | {c['erl']:.2f} | {pen} | {t['erl_merge_penalized']:.2f} | "
                   f"{c['erl_merge_penalized']:.2f} | {n['coverage']:.4f} | {t['coverage']:.4f} |")  # fmt: skip
    return "\n".join(out)


FLOORS = {"oracle": "*oracle (disclosed)*", "floor_single_instance": "floor: one instance for everything",
          "floor_connected_components": "floor: connected components",
          "floor_voxel_instances": "floor: one instance per voxel", "floor_random_instances": "floor: 50 random instances"}  # fmt: skip


def floors_table(cube: str = "s1_00497_01497_03997_256") -> str:
    """Every floor on one cube: identical coverage and precision, separated only by ERL and merges."""
    f = json.loads((ROOT / "data" / f"fibers_{cube}" / "meta.json").read_text())["floors"]
    out = ["| labelling | ERL | ERLpen | coverage | precision | splits | merges | n inst |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for key, name in FLOORS.items():
        r = f[key]
        cells = [f"{r['erl']:.2f}", f"{r['erl_merge_penalized']:.2f}", f"{r['coverage']:.4f}", f"{r['precision']:.4f}",
                 str(r["splits"]), str(r["merges"]), str(r["n_pred_instances"])]  # fmt: skip
        if key == "oracle":
            cells = [f"*{c}*" for c in cells]
        elif cells[1] == "0.00":
            cells[1] = "**0.00**"
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    for size in (256, 512):
        print(f"### {size}³\n\n{table(size)}\n")
    for size in (256, 512):
        print(f"### held out, {size}³\n\n{table(size, held_out=True)}\n")
    for held_out in (False, True):
        for size in (256, 512):
            print(f"### villa's neural tracer, {'held out' if held_out else 'in sample'}, {size}³\n\n"
                  f"{entrant_table(size, held_out=held_out)}\n")
    for held_out in (False, True):
        for size in (256, 512):
            print(f"### villa's neural tracer with the scan around the cube, {'held out' if held_out else 'in sample'}, "
                  f"{size}³\n\n{entrant_table(size, held_out=held_out, row=SCAN_CONTEXT)}\n")
    print(f"### floors\n\n{floors_table()}\n")
