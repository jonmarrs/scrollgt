"""Maintainer tool: print the per-size-class fiber tables for baselines/BASELINES.md from the shipped meta.json files.

Generated, never hand-copied, so a published number cannot drift from the data that tests reproduce.

Usage: python tools/fiber_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(size: int) -> str:
    out = ["| cube | split | oracle ERL | tracer ERL | cc ERL | tracer ERLpen | cc ERLpen | tracer coverage |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for p in sorted((ROOT / "data").glob(f"fibers_*_{size}")):
        m = json.loads((p / "meta.json").read_text())
        f = m["floors"]
        o, c, t = f["oracle"], f["floor_connected_components"], f["tracer_strict_relink"]
        pen = f"**{t['erl_merge_penalized']:.2f}**" if t["erl_merge_penalized"] > c["erl_merge_penalized"] else f"{t['erl_merge_penalized']:.2f}"
        cube = p.name.removeprefix("fibers_").removesuffix(f"_{size}")
        out.append(f"| {cube} | {m['split']} | {o['erl']:.2f} | {t['erl']:.2f} | {c['erl']:.2f} | {pen} | "
                   f"{c['erl_merge_penalized']:.2f} | {t['coverage']:.4f} |")  # fmt: skip
    return "\n".join(out)


if __name__ == "__main__":
    for size in (256, 512):
        print(f"### {size}³\n\n{table(size)}\n")
