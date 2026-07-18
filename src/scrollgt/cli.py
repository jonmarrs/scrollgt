"""ScrollGT command line: `scrollgt score` and `scrollgt check`."""

import argparse
import json
import sys

from .columns import score_columns
from .compliance import check_submission
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
        result = score_prediction(args.prediction, args.target)
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
