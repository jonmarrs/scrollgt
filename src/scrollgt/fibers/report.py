"""Scorecard formatting for fiber connectivity.

Three rules are enforced by construction rather than left to the caller: both ERL
variants always appear together, the tolerance always appears, and the cube's size class
appears with that class's oracle ERL. Raw ERL alone is gameable -- labelling an entire
cube as one instance scores within 23% of the oracle -- so a card showing one number
without the other would be actively misleading. The class matters for the same reason:
ERL is expected run length in voxels, so a 512 cube scores roughly double a 256 cube for
purely geometric reasons, and a score read against the wrong ceiling is read wrong.
"""

from __future__ import annotations

FLOOR_LABELS = {
    "floor_single_instance": "floor: one instance for everything",
    "floor_connected_components": "floor: connected components",
    "floor_voxel_instances": "floor: one instance per voxel",
    "floor_random_instances": "floor: 50 random instances",
}

COLUMNS = ["ERL", "ERLpen", "coverage", "precision", "splits", "merges", "n inst"]


def _row(name: str, r: dict) -> str:
    return (
        f"| {name} | {r['erl']:.2f} | {r['erl_merge_penalized']:.2f} | "
        f"{r['coverage']:.4f} | {r['precision']:.4f} | "
        f"{r['splits']} | {r['merges']} | {r['n_pred_instances']} |"
    )


def _class_line(card: dict) -> str:
    """`size_class` and `class_oracle_erl` rendered, not just carried on the JSON.

    Both have been on the card since size class became first-class, but only
    `--json-out` readers ever saw them; the terminal reader -- the one most likely to
    compare a 512 score against a 256 ceiling by eye -- got neither.
    """
    size = card.get("size_class")
    oracle = card.get("class_oracle_erl")
    cls = f"size class {size}³" if size is not None else "size class unknown"
    if oracle is None:
        return (
            f"{cls}; class oracle ERL unavailable — this target's meta.json carries no "
            f"floors.oracle.erl, so there is no ceiling to read the score against."
        )
    return (
        f"{cls}; class oracle ERL {float(oracle):.2f} — ERL is a length statistic and "
        f"does not compare across size classes."
    )


def fiber_markdown_report(card: dict) -> str:
    lines = [
        f"| {card['prediction']} vs {card['target']} | " + " | ".join(COLUMNS) + " |",
        "|---|" + "|".join(["---"] * len(COLUMNS)) + "|",
        _row("**your labelling**", card["metrics"]),
    ]
    for key, label in FLOOR_LABELS.items():
        if key in card.get("floors", {}):
            lines.append(_row(label, card["floors"][key]))

    split = "cross-scroll split" if card.get("split") == "cross_scroll" else "primary split"
    source = card.get("floors_source", "published")
    lines += [
        "",
        _class_line(card),
        f"tolerance {card['tolerance']} voxels ({split}); floors {source}"
        + (" — rerun with --recompute-floors to verify them from the shipped mask."
           if source == "published" else "."),
        "Splits and merges are reported separately and are never summed: a split "
        "fails to help, a merge corrupts the U/V parameterization.",
    ]

    if len(card.get("floors", {})) > 1:
        lines.append(
            "Every floor shows the same coverage and precision. That is the point, not a "
            "bug: both are properties of the shared fiber mask, not of the labelling, so "
            "they cannot rank a tracer. Only ERL and the merge count separate these rows."
        )

    if card.get("below_baseline"):
        cc = card["floors"]["floor_connected_components"]["erl"]
        lines.append(
            f"\n**BELOW the naive baseline.** Raw ERL {card['metrics']['erl']:.2f} "
            f"trails connected components at {cc:.2f}."
        )
    return "\n".join(lines)
