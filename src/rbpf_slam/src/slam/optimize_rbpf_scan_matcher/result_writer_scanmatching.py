from pathlib import Path
import csv
import numbers
import math
import pandas as pd
import numpy as np

from typing import Any, List

from .optimizer_scanmatching import RankedRunScanMatching

S_TO_MS = 1000.0

class ResultWriterScanMatching:
    @staticmethod
    def _nan() -> float:
        return float("nan")


    @staticmethod
    def _optional_deg(value_rad: Any) -> float:
        return math.degrees(value_rad) if value_rad is not None else ResultWriterScanMatching._nan()


    @staticmethod
    def _optional_ms(value_s: Any) -> float:
        return value_s * S_TO_MS if value_s is not None else ResultWriterScanMatching._nan()


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
    def write_dataframe_csv(
        path: str,
        df: pd.DataFrame,
        override: bool = False,
        float_decimals: int = 6,
        cols_to_use: List[str] = None,
        label: str = "DataFrame",
    ) -> None:
        file_exists = ResultWriterScanMatching.create_path_and_check_if_file_exists(path=path)

        if file_exists and not override:
            print(f"\n{label} has not been saved because file already exists and override is set to False!")
            return

        formatted_df = df.copy()

        if cols_to_use is not None:
            existing_cols = [col for col in cols_to_use if col in formatted_df.columns]
            formatted_df = formatted_df.loc[:, existing_cols]

        for col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map(
                lambda value: (
                    np.nan
                    if value is None
                    else ResultWriterScanMatching._format_csv_value(value, float_decimals=float_decimals)
                )
            )

        formatted_df.to_csv(path, index=False)
        print(f"\n{label} has been saved to:\n{path}")


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
                    "measurement_stddev",
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
                    
                    # Translational errors of raw odom, pred and corrected pose
                    "mean_raw_odom_trans_err",
                    "mean_pred_trans_err",
                    "mean_corr_trans_err",
                    "rmse_raw_odom_trans_err",
                    "rmse_pred_trans_err",
                    "rmse_corr_trans_err",

                    # Rotational errors of raw  odom, pred and corrected pose
                    "mean_raw_odom_rot_err_deg",
                    "mean_pred_rot_err_deg",
                    "mean_corr_rot_err_deg",                         
                    "rmse_raw_odom_rot_err_deg",
                    "rmse_pred_rot_err_deg",                                   
                    "rmse_corr_rot_err_deg",

                    # Percentil
                    "perc_95_corr_trans_err",
                    "perc_95_corr_rot_err",
                    
                    # Rolling rmse
                    "max_rolling_rmse_corr_trans_error",
                    "max_rolling_rmse_corr_rot_error",
                    
                    # RMSE rolling mean and improvement statistics
                    "corr_worse_rate_trans",
                    "corr_worse_rate_rot",                    
                    "mean_corr_trans_improvm",
                    "mean_corr_rot_improvm_deg",

                    # Raw odometry drift baseline before scan-matching drift.
                    "final_raw_odom_drift_trans",
                    "final_raw_odom_drift_rot_deg",
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
                    run.params.measurement_noise_stddev,
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
                    ResultWriterScanMatching._optional_deg(summary.mean_best_rot_abs),
                    ResultWriterScanMatching._optional_deg(summary.max_best_rot_abs),

                    # Translational errors of raw odom, pred and corrected pose
                    summary.mean_raw_odom_trans_err,
                    summary.mean_pred_trans_err,
                    summary.mean_corr_trans_err,
                    summary.rmse_raw_odom_trans_err,
                    summary.rmse_pred_trans_err,
                    summary.rmse_corr_trans_err,

                    # Rotational errors of raw  odom, pred and corrected pose
                    ResultWriterScanMatching._optional_deg(summary.mean_raw_odom_rot_err),
                    ResultWriterScanMatching._optional_deg(summary.mean_pred_rot_err),
                    ResultWriterScanMatching._optional_deg(summary.mean_corr_rot_err),                    
                    ResultWriterScanMatching._optional_deg(summary.rmse_raw_odom_rot_err),
                    ResultWriterScanMatching._optional_deg(summary.rmse_pred_rot_err),                    
                    ResultWriterScanMatching._optional_deg(summary.rmse_corr_rot_err),
                    
                    # Percentil
                    summary.perc_95_corr_trans_err,
                    ResultWriterScanMatching._optional_deg(summary.perc_95_corr_rot_err),

                    # RMSE rolling mean and improvement statistics
                    summary.max_rolling_rmse_corr_trans_error,
                    ResultWriterScanMatching._optional_deg(summary.max_rolling_rmse_corr_rot_error),
                    summary.corr_worse_rate_trans,  
                    summary.corr_worse_rate_rot,    
                    summary.mean_corr_trans_improvm,
                    ResultWriterScanMatching._optional_deg(summary.mean_corr_rot_improvm),
                    
                    summary.final_raw_odom_drift_trans,
                    ResultWriterScanMatching._optional_deg(summary.final_raw_odom_drift_rot),
                    summary.final_drift_trans,
                    ResultWriterScanMatching._optional_deg(summary.final_drift_rot),

                    ResultWriterScanMatching._optional_ms(summary.mean_timing_sm_update_particle_s),
                    ResultWriterScanMatching._optional_ms(summary.mean_timing_sm_scan_match_update_pose_s),
                    ResultWriterScanMatching._optional_ms(summary.mean_timing_sm_map_extension_s),
                    ResultWriterScanMatching._optional_ms(summary.mean_timing_sm_map_update_s),
                    ResultWriterScanMatching._optional_ms(summary.mean_step_duration),
                    
                    summary.n_steps,
                   
                ]

                writer.writerow(
                    [
                        ResultWriterScanMatching._format_csv_value(value, float_decimals=float_decimals)
                        for value in row
                    ]
                )

        print(f"\nScan-matching summary has been saved to:\n{path}")
