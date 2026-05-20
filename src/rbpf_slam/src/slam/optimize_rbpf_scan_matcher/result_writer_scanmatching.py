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
    def _build_tag_from_params(params: Any) -> str:
        """Build a deterministic tag from the actual parameter values used for a run."""
        return (
            f"nf{params.every_nth_scan_filter}_nm{params.every_nth_scan_map}_"
            f"ip{params.occupancy_params.increasing_probability}_"
            f"dp{params.occupancy_params.decreasing_probability}_"
            f"lomin{params.occupancy_params.min_log_odds}_lomax{params.occupancy_params.max_log_odds}_"
            f"ot{params.scan_matcher_params.occ_thres}_dr{params.scan_matcher_params.delta_r}_"
            f"sr{params.scan_matcher_params.surface_radius_m}_mfr{params.scan_matcher_params.min_free_ratio}_"
            f"mnp{params.icp_params.max_n_points}_npca{params.icp_params.neighbors_pca}_"
            f"mi{params.icp_params.max_iterations}_mcd{params.icp_params.max_correspondence_distance}_"
            f"mc{params.icp_params.min_corresp}_mjt{params.icp_params.max_translation_jump}_"
            f"mjrd{math.degrees(params.icp_params.max_rotation_jump)}_"
            f"mae{params.icp_params.max_acceptable_mean_error}"
        )


    @staticmethod
    def create_path_and_check_if_file_exists(path: str) -> bool:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        return path_obj.exists()


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
                    "rank",
                    "tag",
                    "seed",
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
                    "icp_mean_err",

                    "true_x",
                    "true_y",
                    "true_theta_deg",
                    "pred_x",
                    "pred_y",
                    "pred_theta_deg",
                    "corr_x",
                    "corr_y",
                    "corr_theta_deg",

                    "pred_trans_err",
                    "corr_trans_err",
                    "pred_rot_err_deg",
                    "corr_rot_err_deg",

                    "pred_to_corr_trans_err",
                    "pred_to_corr_rot_err_deg",                    
                    
                    "t_ogm_ms",
                    "t_scan_matching_ms",
                    "t_prediction_ms",
                    "t_map_extraction_ms",
                    "t_correct_pose_ms",
                ]
            )

            ordered_runs = sorted(ranked_runs, key=lambda run: run.score)
            for rank, run in enumerate(ordered_runs, start=1):
                run_tag = ResultWriterScanMatching._build_tag_from_params(run.params)
                for step in run.step_results:
                    true_x, true_y, true_theta = step.true_pose if step.true_pose is not None else (None, None, None)
                    pred_x, pred_y, pred_theta = step.pred_pose if step.pred_pose is not None else (None, None, None)
                    corr_x, corr_y, corr_theta = step.corr_pose if step.corr_pose is not None else (None, None, None)

                    row = [
                        rank,
                        run_tag,
                        run.seed,
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
                        math.degrees(step.icp_best_rot_abs_rad) if step.icp_best_rot_abs_rad is not None else "None",
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

                        step.pred_trans_err,
                        step.corr_trans_err,
                        math.degrees(step.pred_rot_err) if step.pred_rot_err is not None else None,
                        math.degrees(step.corr_rot_err) if step.corr_rot_err is not None else None,

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


    @staticmethod
    def write_summary_runs_csv(
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
                    "seed",
                    "score",
                    "tag",

                    # Grid parameters: playback sampling
                    "every_nth_beam_filter",
                    "every_nth_beam_map",

                    # Grid parameters: OccupancyParams (OGM)
                    "increasing_probability",
                    "decreasing_probability",
                    "min_log_odds",
                    "max_log_odds",

                    # Grid parameters: ScanMatcherParams
                    "occ_thres",
                    "delta_r",
                    "surface_radius_m",
                    "min_free_ratio",

                    # Grid parameters: ICPParams
                    "max_n_points",
                    "neighbors_pca",
                    "max_iterations",
                    "max_correspondence_distance",
                    "min_corresp",
                    "max_translation_jump",
                    "max_rotation_jump_deg",
                    "max_acceptable_mean_error",
                    
                    "scan_match_failed_count",                    
                    "icp_failed_count",
                    "icp_success_rate",

                    # ICP error metrics
                    "mean_icp_iterations",
                    "count_too_few_points",
                    "count_too_few_corresp",
                    "infinite_h_or_g",
                    "ill_cond_H",
                    "infinite_dtransform",
                    "infinite_mean_err",
                    "best_transf_too_large",
                    "best_mean_err_too_large",


                    "mean_icp_err",
                    "mean_best_trans_norm",
                    "max_best_trans_norm",
                    "mean_best_rot_abs_deg",
                    "max_best_rot_abs_deg",
                    
                    # predicted pose error metrics
                    "mean_pred_trans_err",
                    "mean_pred_rot_err_deg",
                    "rmse_pred_trans_err",
                    "rmse_pred_rot_err_deg",

                    # CCorrected pose error metrics
                    "mean_corr_trans_err",
                    "mean_corr_rot_err_deg",                    
                    "rmse_corr_trans_err",
                    "rmse_corr_rot_err_deg",
                    "perc_95_corr_trans_err",
                    "perc_95_corr_rot_err",
                    
                    # Rolling rmse
                    "max_rolling_rmse_corr_trans_error",
                    "max_rolling_rmse_corr_rot_error",
                    "corr_worse_rate_trans",
                    "corr_worse_rate_rot",
                    # Scan match improvement metrics
                    "mean_corr_trans_improvm",
                    "mean_corr_rot_improvm_deg",

                    "final_drift_trans",   
                    "final_drift_rot_deg",                 

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
                run_tag = ResultWriterScanMatching._build_tag_from_params(run.params)
                row = [
                    rank,
                    run.seed,
                    run.score,
                    run_tag,

                    # Grid parameters: playback sampling
                    run.params.every_nth_scan_filter,
                    run.params.every_nth_scan_map,

                    # Grid parameters: OccupancyParams (OGM)
                    run.params.occupancy_params.increasing_probability,
                    run.params.occupancy_params.decreasing_probability,
                    run.params.occupancy_params.min_log_odds,
                    run.params.occupancy_params.max_log_odds,

                    # Grid parameters: ScanMatcherParams
                    run.params.scan_matcher_params.occ_thres,
                    run.params.scan_matcher_params.delta_r,
                    run.params.scan_matcher_params.surface_radius_m,
                    run.params.scan_matcher_params.min_free_ratio,

                    # Grid parameters: ICPParams
                    run.params.icp_params.max_n_points,
                    run.params.icp_params.neighbors_pca,
                    run.params.icp_params.max_iterations,
                    run.params.icp_params.max_correspondence_distance,
                    run.params.icp_params.min_corresp,
                    run.params.icp_params.max_translation_jump,
                    math.degrees(run.params.icp_params.max_rotation_jump),
                    run.params.icp_params.max_acceptable_mean_error,
                    
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

                    summary.mean_icp_err,
                    summary.mean_best_trans_norm,
                    summary.max_best_trans_norm,
                    math.degrees(summary.mean_best_rot_abs) if summary.mean_best_rot_abs is not None else None,
                    math.degrees(summary.max_best_rot_abs) if summary.max_best_rot_abs is not None else None,

                    summary.mean_pred_trans_err,
                    math.degrees(summary.mean_pred_rot_err) if summary.mean_pred_rot_err is not None else None,
                    summary.rmse_pred_trans_err,
                    math.degrees(summary.rmse_pred_rot_err) if summary.rmse_pred_rot_err is not None else None,
                                        
                    summary.mean_corr_trans_err,
                    math.degrees(summary.mean_corr_rot_err) if summary.mean_corr_rot_err is not None else None,                                        
                    summary.rmse_corr_trans_err,
                    math.degrees(summary.rmse_corr_rot_err) if summary.rmse_corr_rot_err is not None else None,
                    summary.perc_95_corr_trans_err,
                    math.degrees(summary.perc_95_corr_rot_err) if summary.perc_95_corr_rot_err is not None else None,
                    summary.max_rolling_rmse_corr_trans_error,
                    math.degrees(summary.max_rolling_rmse_corr_rot_error) if summary.max_rolling_rmse_corr_rot_error is not None else None,
                    summary.corr_worse_rate_trans,  
                    summary.corr_worse_rate_rot,    
                    summary.mean_corr_trans_improvm,
                    math.degrees(summary.mean_corr_rot_improvm) if summary.mean_corr_rot_improvm is not None else None,
                    
                    summary.final_drift_trans,
                    math.degrees(summary.final_drift_rot) if summary.final_drift_rot is not None else None,

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
