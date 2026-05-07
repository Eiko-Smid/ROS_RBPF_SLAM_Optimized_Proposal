from pathlib import Path
import csv
import numbers
import math

from typing import Any, List

from .optimizer_scanmatching import RankedRunScanMatching

S_TO_MS = 1000.0

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
                    # "tag",
                    "every_nth_beam_filter",
                    "every_nth_beam_map",
                    
                    "scan_match_failed_count",                    
                    "icp_failed_count",
                    "icp_success_rate",

                    "mean_icp_iterations",
                    "count_too_few_points",
                    "count_too_few_corresp",
                    "infinite_h_or_g",
                    "ill_cond_H",
                    "infinite_dtransform",
                    "infinite_mean_err",
                    "best_transf_too_large",
                    "best_mean_err_too_large",

                    "mean_icp_error",
                    "mean_best_trans_norm",
                    "max_best_trans_norm",
                    "mean_best_rot_abs_deg",
                    "max_best_rot_abs_deg",
                    
                    "mean_pred_trans_error",
                    "mean_pred_rot_error_deg",
                    "mean_corr_trans_error",
                    "mean_corr_rot_error_deg",
                    "rmse_pred_trans_error",
                    "rmse_pred_rot_error",
                    "rmse_corr_trans_error",
                    "rmse_corr_rot_error",
                    "final_drift",                    

                    "mean_timing_sm_update_particle_ms",
                    "mean_timing_sm_scan_match_update_pose_ms",
                    "mean_timing_sm_map_extension_ms",
                    "mean_timing_sm_map_update_ms",
                    "mean_step_duration_ms",
                    
                    "n_steps",
                ]
            )

            for rank, run in enumerate(ranked_runs, start=1):
                summary = run.summary
                row = [
                    rank,
                    run.score,
                    # run.params.tag,
                    run.params.every_nth_scan_filter,
                    run.params.every_nth_scan_map,
                    
                    summary.scan_match_failed_count,                    
                    summary.icp_failed_count,
                    summary.icp_success_rate,

                    summary.mean_icp_iterations,
                    summary.count_too_few_points,
                    summary.count_too_few_corresp,
                    summary.infinite_h_or_g,
                    summary.ill_cond_H,
                    summary.infinite_dtransform,
                    summary.infinite_mean_err,
                    summary.best_transf_too_large,
                    summary.best_mean_err_too_large,

                    summary.mean_icp_error,
                    summary.mean_best_trans_norm,
                    summary.max_best_trans_norm,
                    math.degrees(summary.mean_best_rot_abs) if summary.mean_best_rot_abs is not None else None,
                    math.degrees(summary.max_best_rot_abs) if summary.max_best_rot_abs is not None else None,

                    summary.mean_pred_trans_error,
                    math.degrees(summary.mean_pred_rot_error) if summary.mean_pred_rot_error is not None else None,
                    summary.mean_corr_trans_error,
                    math.degrees(summary.mean_corr_rot_error) if summary.mean_corr_rot_error is not None else None,
                    summary.rmse_pred_trans_error,
                    summary.rmse_pred_rot_error,
                    summary.rmse_corr_trans_error,
                    summary.rmse_corr_rot_error,
                    summary.final_drift,

                    (summary.mean_timing_sm_update_particle_s or 0.0) * S_TO_MS,
                    (summary.mean_timing_sm_scan_match_update_pose_s or 0.0) * S_TO_MS,
                    (summary.mean_timing_sm_map_extension_s or 0.0) * S_TO_MS,
                    (summary.mean_timing_sm_map_update_s or 0.0) * S_TO_MS,
                    (summary.mean_step_duration or 0.0) * S_TO_MS,
                    
                    summary.n_steps,
                   
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

                    "scan_match_failed",
                    "icp_iterations",
                    "n_correspondences",
                    "use_transformation",
                    "stop_reason",

                    "n_measurements_total",
                    "n_valid_measurements_filter",
                    "n_valid_measurements_map_update",
                    "n_map_points_extracted",

                    "icp_best_trans_param",
                    "icp_best_rot_abs_deg",
                    "icp_mean_error",

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

                    "pred_to_corr_trans_err",
                    "pred_to_corr_rot_err_deg",                    
                    
                    "t_ogm_ms",
                    "t_scan_matching_ms",
                    "t_prediction_ms",
                    "t_map_extraction_ms",
                    "t_correct_pose_ms",
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

                        step.scan_match_failed,
                        step.icp_iterations if step.icp_iterations is not None else "None",
                        step.n_correspondences if step.n_correspondences is not None else "None",
                        step.use_transformation if step.use_transformation is not None else "None",
                        step.stop_reason,
                        
                        step.n_measurements_total,
                        step.n_valid_measurements_filter,
                        step.n_valid_measurements_map_update,
                        step.n_map_points_extracted if step.n_map_points_extracted is not None else "None",
                        
                        step.icp_best_trans_param if step.icp_best_trans_param is not None else "None",
                        math.degrees(step.icp_best_rot_abs_deg) if step.icp_best_rot_abs_deg is not None else "None",
                        step.icp_mean_error if step.icp_mean_error is not None else "None",

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

                        step.pred_to_corr_trans_err,
                        math.degrees(step.pred_to_corr_rot_err) if step.pred_to_corr_rot_err is not None else None,                        
                        
                        step.t_ogm * S_TO_MS if step.t_ogm is not None else 0.0,
                        step.t_scan_matching * S_TO_MS if step.t_scan_matching is not None else 0.0,
                        step.t_prediction * S_TO_MS if step.t_prediction is not None else 0.0,
                        step.t_map_extraction * S_TO_MS if step.t_map_extraction is not None else 0.0,
                        step.t_correct_pose * S_TO_MS if step.t_correct_pose is not None else 0.0,
                    ]

                    writer.writerow(
                        [
                            ResultWriterScanMatching._format_csv_value(value, float_decimals=float_decimals)
                            for value in row
                        ]
                    )

        print(f"\nScan-matching step trace has been saved to:\n{output_path}")
