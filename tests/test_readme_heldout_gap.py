"""The README's held-out gap commands: one held-out Scroll-5 cube and one held-out Scroll-3 cube, scored with the
connected-components floor (the whole mask). Every number the README prints beside them is read from the target's
own `meta.json`, so a re-scored row or a swapped cube fails here.

The 256³ command is also run through the CLI. This suite checks only the 512³ command's published row;
test_fiber_target.py recomputes one 512³ target by default and marks the other ten as heavy. The 0.9.1 corpus
check records 2.55–2.57 GiB peak RSS for the 512³ connected-components CLI under a 7 GiB address-space limit.
"""

import json
import re
from pathlib import Path

from scrollgt.cli import main

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
README = ROOT / "README.md"

S5 = "s5_16997_03997_04497_256"
S3 = "s3_00994_00494_01994_512"
CMD = ".venv/bin/scrollgt score-fibers --floor connected_components data/fibers_{}  # coverage {}"


def _meta(cube):
    return json.loads((DATA / f"fibers_{cube}" / "meta.json").read_text())


def _cc_coverage(cube):
    return _meta(cube)["floors"]["floor_connected_components"]["coverage"]


def _flat(text):
    return " ".join(text.split())


def test_the_two_cubes_are_held_out_scroll_5_and_scroll_3():
    s5, s3 = _meta(S5), _meta(S3)
    assert (s5["split"], s5["scroll"].split()[:2], s5["size_class"]) == ("held_out", ["Scroll", "5"], 256)
    assert (s3["split"], s3["scroll"].split()[:2], s3["size_class"]) == ("held_out", ["Scroll", "3"], 512)


def test_readme_commands_print_each_cubes_published_coverage():
    text = README.read_text()
    for cube in (S5, S3):
        assert CMD.format(cube, f"{_cc_coverage(cube):.4f}") in text, cube


def test_readme_neural_tracer_coverage_is_the_published_row():
    neural = [_meta(c)["floors"]["villa_neural_tracer"]["coverage"] for c in (S5, S3)]
    assert (
        f"On the same two cubes villa's neural tracer, which draws its own centrelines, covers "
        f"{neural[0]:.4f} and {neural[1]:.4f}." in _flat(README.read_text())
    )


def test_the_held_out_scroll_5_command_reproduces_its_published_row(capsys):
    assert main(["score-fibers", "--floor", "connected_components", str(DATA / f"fibers_{S5}")]) == 0
    row = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("| **your labelling**"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3] == f"{_cc_coverage(S5):.4f}"


def test_the_mask_sentence_names_the_mask_not_the_neural_tracer():
    """0.8.0 put the neural tracer's sentence between "Reference mask: ..." and "It was trained on the eleven in-sample
    cubes", so "It" read as the neural tracer, which trained on no Scroll 5."""
    flat = _flat(README.read_text())
    assert "**The mask's model was trained on the eleven in-sample cubes**" in flat
    assert not re.search(r"presence input\. \*\*It was trained", flat)
