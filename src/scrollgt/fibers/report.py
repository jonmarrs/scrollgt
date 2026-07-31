"""Scorecard formatting for fiber connectivity.

Two rules are enforced by construction rather than left to the caller: both
ERL variants always appear together, and the tolerance always appears. Raw ERL
alone is gameable -- labelling an entire cube as one instance scores within
23% of the oracle -- so a card showing one number without the other would be
actively misleading.
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
