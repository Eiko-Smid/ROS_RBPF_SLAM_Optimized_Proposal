from pathlib import Path
import csv
import numbers
import math

from typing import List

from .optimizer import RankedRun


class ResultWriterScanMatching:
    @staticmethod
    def _format_csv_value(value, float_decimals: int):
        if isinstance(value, bool):
            return value

        if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
            return f"{float(value):.{float_decimals}f}"

        return value

    @staticmethod
    def create_path_and_check_if_file_exists(path: str) -> bool:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        return path_obj.exists()

    @staticmethod
    def write_ranked_runs_csv(
        path: str,
        ranked_runs: List[RankedRun],
        override: bool = False,
        float_decimals: int = 4,
    ) -> None:
        file_exists = ResultWriterScanMatching.create_path_and_check_if_file_exists(path=path)

        if file_exists and not override:
            print("\nScan-matching summary has not been saved because file exists and override=False.")
            return

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "score",
                    "tag",
                    "n_particles",
                    "every_nth_beam_filter",
                    "scan_match_failed_count",
                    "scan_match_fallback_failed_count",
                    "count_too_few_points",
                    "count_too_few_corresp",
                    "infinite_h_or_g",
                    "ill_cond_H",
                    "infinite_dtransform",
                    "infinite_mean_err",
                    "best_transf_too_large",
                    "best_mean_err_too_large",
                    "mean_trans_error",
                    "mean_rot_error_deg",
                    "rmse_trans_error",
                    "rmse_rot_error_deg",
                    "mean_step_duration_ms",
                    "n_steps",
                ]
            )

            for rank, run in enumerate(ranked_runs, start=1):
                summary = run.summary
                row = [
                    rank,
                    run.score,
                    run.params.tag,
                    summary.get("n_particles"),
                    run.params.every_nth_scan_filter,
                    summary.get("scan_match_failed_count"),
                    summary.get("scan_match_fallback_failed_count"),
                    summary.get("count_too_few_points", 0),
                    summary.get("count_too_few_corresp", 0),
                    summary.get("infinite_h_or_g", 0),
                    summary.get("ill_cond_H", 0),
                    summary.get("infinite_dtransform", 0),
                    summary.get("infinite_mean_err", 0),
                    summary.get("best_transf_too_large", 0),
                    summary.get("best_mean_err_too_large", 0),
                    summary.get("mean_translation_error"),
                    math.degrees(summary.get("mean_rotation_error")) if summary.get("mean_rotation_error") is not None else None,
                    summary.get("rmse_translation_error"),
                    math.degrees(summary.get("rmse_rotation_error")) if summary.get("rmse_rotation_error") is not None else None,
                    (summary.get("mean_step_duration", 0.0) or 0.0) * 1000.0,
                    summary.get("n_steps"),
                ]

                writer.writerow(
                    [
                        ResultWriterScanMatching._format_csv_value(value, float_decimals=float_decimals)
                        for value in row
                    ]
                )

        print(f"\nScan-matching summary has been saved to:\n{path}")

    @staticmethod
    def write_ranked_step_traces_csv(
        output_path: str,
        ranked_runs: List[RankedRun],
        override: bool = False,
        float_decimals: int = 6,
    ) -> None:
        output_file_path = Path(output_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        if output_file_path.exists() and not override:
            print(f"Skipping scan-matching step trace (exists, override=False): {output_file_path}")
            return

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "tag",
                    "step_id",
                    "trans_error",
                    "rot_error",
                ]
            )

            for rank, run in enumerate(ranked_runs, start=1):
                for step in run.step_results:
                    row = [
                        rank,
                        run.params.tag,
                        step.step_idx,
                        step.translation_error,
                        step.rotation_error,
                    ]

                    writer.writerow(
                        [
                            ResultWriterScanMatching._format_csv_value(value, float_decimals=float_decimals)
                            for value in row
                        ]
                    )

        print(f"\nScan-matching step trace has been saved to:\n{output_path}")
