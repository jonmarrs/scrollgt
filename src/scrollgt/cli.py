"""ScrollGT command line: `scrollgt score` and `scrollgt check`."""

import argparse
import json
import sys

from .columns import score_columns
from .compliance import check_submission
from .fibers.report import fiber_markdown_report
from .fibers.target import score_fiber_prediction
from .score import markdown_report, score_prediction


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scrollgt",
        description="Score ink predictions against registered human ground truth "
        "on the open Vesuvius SOTA scroll data.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score", help="score a prediction against a target")
    p_score.add_argument("prediction", help="probability map (.png 8-bit or .npy in [0,1])")
    p_score.add_argument("target", help="target directory (contains gt_ink.png + meta.json)")
    p_score.add_argument("--json-out", default=None, help="write the scorecard JSON here")
    p_score.add_argument("--allow-non-scoring", action="store_true",
                         help="score a target explicitly marked non-scoring. Kept so the "
                              "historical published rows stay reproducible; the result is "
                              "not comparable to scoring targets.")
    p_score.add_argument("--allow-failing-placement", action="store_true",
                         help="score even if the target's ground truth fails its placement "
                              "check. The resulting number measures misalignment as much as "
                              "reading, and is not comparable to other targets.")

    p_cols = sub.add_parser(
        "score-columns",
        help="score a prediction against a column-level target (e.g. PHerc 1667 "
             "merged geometry — scholar-validated column GT, no pixel labels)",
    )
    p_cols.add_argument("prediction", help="probability map at target grid resolution "
                                           "(.png 8-bit or .npy in [0,1])")
    p_cols.add_argument("target", help="column target directory (meta.json + columns.json)")
    p_cols.add_argument("--origin", nargs=2, type=int, metavar=("Y", "X"),
                        default=[0, 0],
                        help="grid coordinate of the prediction's top-left corner "
                             "(for partial-extent predictions; default 0 0)")
    p_cols.add_argument("--json-out", default=None, help="write the scorecard JSON here")

    p_fib = sub.add_parser(
        "score-fibers",
        help="score a fiber instance labelling against hand-traced ground truth "
             "(ERL, splits, merges, and the anti-gaming floors)",
    )
    p_fib.add_argument("prediction",
                       help="instance labels (.npy of ints, 0 = background, cube-shaped)")
    p_fib.add_argument("target", help="fiber target directory (data/fibers_<cube>)")
    p_fib.add_argument("--recompute-floors", action="store_true",
                       help="recompute the floors from the shipped mask instead of "
                            "reading the published values (~50 s for a 256 cube, "
                            "several minutes for a 512 cube)")
    p_fib.add_argument("--json-out", default=None, help="write the scorecard JSON here")

    p_check = sub.add_parser("check", help="prize-compliance pre-check (window + overlap)")
    p_check.add_argument("--window-px", type=int, required=True,
                         help="ML window size in pixels (lateral)")
    p_check.add_argument("--scan-um", type=float, required=True,
                         help="scan resolution in microns per pixel (e.g. 8.0)")
    p_check.add_argument("--regions-json", default=None,
                         help='JSON file: {"train_regions": [...], "predict_region": {...}} '
                              "with y0/x0/h/w (+ optional volume) per region")

    args = parser.parse_args(argv)

    if args.cmd == "score":
        result = score_prediction(args.prediction, args.target,
                                  allow_failing_placement=args.allow_failing_placement,
                                  allow_non_scoring=args.allow_non_scoring)
        print(markdown_report([result]))
        print()
        m = result["metrics"]
        print(f"val_f1={m.get('val_f1', float('nan')):.4f}  "
              f"ap_prevalence_lift={m.get('ap_prevalence_lift', float('nan')):.4f}  "
              f"roc_auc={m.get('roc_auc', float('nan')):.4f}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(result, f, indent=2, default=float)
            print(f"scorecard written to {args.json_out}")
        return 0

    if args.cmd == "score-columns":
        result = score_columns(args.prediction, args.target,
                               origin=tuple(args.origin))
        m = result["metrics"]
        print(f"target: {result['target']}  cols scored: {m['cols_scored']}")
        print(f"col_gutter_auc={m.get('col_gutter_auc', float('nan')):.4f}  "
              f"pixel_auc={m.get('col_gutter_pixel_auc', float('nan')):.4f}  "
              f"line_period_peak_mean={m.get('line_period_peak_mean', float('nan')):.4f}")
        print(f"(text cols {m['n_text_cols']}, gutters {m['n_gutters']}, "
              f"excluded gutters {m['excluded_gutters']})")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(result, f, indent=2, default=float)
            print(f"scorecard written to {args.json_out}")
        return 0

    if args.cmd == "score-fibers":
        card = score_fiber_prediction(args.prediction, args.target,
                                      recompute_floors=args.recompute_floors)
        print(fiber_markdown_report(card))
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(card, f, indent=2, default=float)
        return 0

    if args.cmd == "check":
        train_regions, predict_region = [], {}
        if args.regions_json:
            with open(args.regions_json) as f:
                spec = json.load(f)
            train_regions = spec.get("train_regions", [])
            predict_region = spec.get("predict_region", {})
        ok, failures = check_submission(args.window_px, args.scan_um,
                                        train_regions, predict_region)
        if ok:
            print("COMPLIANT: window within 0.5mm cap"
                  + ("; no train/predict overlap" if train_regions else
                     " (no regions supplied — overlap not checked)"))
            return 0
        for fail in failures:
            print(f"FAIL: {fail}")
        return 1

    return 2  # unreachable with required=True


if __name__ == "__main__":
    sys.exit(main())
