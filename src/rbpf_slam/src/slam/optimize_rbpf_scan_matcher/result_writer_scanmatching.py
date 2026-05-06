from pathlib import Path
import csv
import numbers
import math

from typing import Any, List

from .optimizer_scanmatching import RankedRunScanMatching


class ResultWriterScanMatching:
    @staticmethod
    def _format_csv_value(value: Any, float_decimals: int) -> Any:
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
        ranked_runs: List[RankedRunScanMatching],
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
                    "icp_failed_count",
                    "count_too_few_points",
                    "count_too_few_corresp",
                    "infinite_h_or_g",
                    "ill_cond_H",
                    "infinite_dtransform",
                    "infinite_mean_err",
                    "best_transf_too_large",
                    "best_mean_err_too_large",
                    "mean_timing_sm_update_particle_ms",
                    "mean_timing_sm_scan_match_update_pose_ms",
                    "mean_timing_sm_map_extension_ms",
                    "mean_timing_sm_map_update_ms",
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
                    summary.get("icp_failed_count", 0),
                    summary.get("count_too_few_points", 0),
                    summary.get("count_too_few_corresp", 0),
                    summary.get("infinite_h_or_g", 0),
                    summary.get("ill_cond_H", 0),
                    summary.get("infinite_dtransform", 0),
                    summary.get("infinite_mean_err", 0),
                    summary.get("best_transf_too_large", 0),
                    summary.get("best_mean_err_too_large", 0),
                    (summary.get("mean_timing_sm_update_particle_s", 0.0) or 0.0) * 1000.0,
                    (summary.get("mean_timing_sm_scan_match_update_pose_s", 0.0) or 0.0) * 1000.0,
                    (summary.get("mean_timing_sm_map_extension_s", 0.0) or 0.0) * 1000.0,
                    (summary.get("mean_timing_sm_map_update_s", 0.0) or 0.0) * 1000.0,
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
        ranked_runs: List[RankedRunScanMatching],
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
                    "run_id",
                    "step",
                    "t",
                    "true_x",
                    "true_y",
                    "true_theta_deg",
                    "pred_x",
                    "pred_y",
                    "pred_theta_deg",
                    "corr_x",
                    "corr_y",
                    "corr_theta_deg",
                    "pred_trans_error",
                    "corr_trans_error",
                    "pred_rot_error_deg",
                    "corr_rot_error_deg",
                    "best_trans_norm",
                    "best_rot_abs_deg",
                    "pred_to_corr_dist",
                    "pred_to_corr_rot_deg",
                    "icp_iterations",
                    "icp_mean_error",
                    "n_correspondences",
                    "use_transformation",
                    "stop_reason",
                ]
            )

            for rank, run in enumerate(ranked_runs, start=1):
                for step in run.step_results:
                    true_x, true_y, true_theta = step.true_pose if step.true_pose is not None else (None, None, None)
                    pred_x, pred_y, pred_theta = step.pred_pose if step.pred_pose is not None else (None, None, None)
                    corr_x, corr_y, corr_theta = step.corr_pose if step.corr_pose is not None else (None, None, None)

                    row = [
                        run.params.tag,
                        step.step_idx,
                        step.t,
                        true_x,
                        true_y,
                        math.degrees(true_theta) if true_theta is not None else None,
                        pred_x,
                        pred_y,
                        math.degrees(pred_theta) if pred_theta is not None else None,
                        corr_x,
                        corr_y,
                        math.degrees(corr_theta) if corr_theta is not None else None,
                        step.pred_translation_error,
                        step.corr_translation_error,
                        math.degrees(step.pred_rotation_error) if step.pred_rotation_error is not None else None,
                        math.degrees(step.corr_rotation_error) if step.corr_rotation_error is not None else None,
                        step.best_trans_norm,
                        math.degrees(step.best_rot_abs) if step.best_rot_abs is not None else None,
                        step.pred_to_corr_dist,
                        math.degrees(step.pred_to_corr_rot) if step.pred_to_corr_rot is not None else None,
                        step.icp_iterations,
                        step.icp_mean_error,
                        step.n_correspondences,
                        step.use_transformation,
                        step.stop_reason,
                    ]

                    writer.writerow(
                        [
                            ResultWriterScanMatching._format_csv_value(value, float_decimals=float_decimals)
                            for value in row
                        ]
                    )

        print(f"\nScan-matching step trace has been saved to:\n{output_path}")
