import math
from typing import Iterable, List

import pandas as pd
import numpy as np

from .optimizer import RankedRun, RBPFOptimizer
from .playback_defs import ExperimentParams


class RankedRunConverter:
    """
    Convert ranked optimization runs into pandas DataFrames.

    This version avoids the ColumnSpec/lambda approach and directly builds
    one dictionary per RankedRun.
    """

    @staticmethod
    def _deg(value):
        """Convert radians to degrees, while preserving None."""
        return math.degrees(value) if value is not None else None


    @staticmethod
    def _optional_ms(value_s):
        """Convert seconds to milliseconds, while preserving None."""
        return value_s * 1000.0 if value_s is not None else None


    @staticmethod
    def _read_from_summary(run: RankedRun, key: str, default=None):
        '''
        Safely read a value from run.summary. Raises error if run.summary is not a dict or if the key is not present. 
        '''
        if not isinstance(run.summary, dict):
            raise ValueError(f"\nExpected run.summary to be a dict, got {type(run.summary)}")

        if key not in run.summary.keys():
            raise KeyError(f"\nKey '{key}' not found in run.summary. Available keys: {list(run.summary.keys())}")

        value = run.summary.get(key, default)
            
        return value


    @classmethod
    def to_dataframe(cls, ranked_runs: Iterable[RankedRun]) -> pd.DataFrame:
        """
        Convert ranked runs to a pandas DataFrame.
        """
        rows = []

        for run in ranked_runs:
            params = run.params
            n_steps = cls._read_from_summary(run, "n_steps")
            scan_match_failed_count = cls._read_from_summary(run, "scan_match_failed_count", 0)
            scan_match_fallback_failed_count = cls._read_from_summary(
                run,
                "scan_match_fallback_failed_count",
                0,
            )

            # failed_rate = 1 - success_rate; equivalent to failed_count / n_steps
            scan_match_failed_rate = (
                float(scan_match_failed_count) / float(n_steps)
                if n_steps not in (None, 0)
                else None
            )
            scan_match_fallback_failed_rate = (
                float(scan_match_fallback_failed_count) / float(n_steps)
                if n_steps not in (None, 0)
                else None
            )

            row = {
                # Basic run information
                "score": run.score,
                "map": run.map_name,
                "dataset_id": run.dataset_id,
                "parameter_tag": run.parameter_tag,
                "parameter_hash": run.parameter_hash,
                "used_meas_model": getattr(params, "used_meas_model", None),
                "n_steps": n_steps,

                # Parameters
                "seed": run.seed,
                "measurement_stddev": params.measurement_noise_stddev,
                "every_nth_beam_filter": params.every_nth_scan_filter,
                "every_nth_beam_map": params.every_nth_scan_map,
                "n_particles": params.particle_params.n_particles,
                # TODO: Adapt measurement model here!
                # "sigma_measurement": params.measurement_model_params.sigma_measurement,
                # "sigma_measurement":params.measurement_model_params.sigma_hit,
                # "proposal_alpha": params.proposal_alpha,
                # "proposal_beta": params.proposal_beta,
                "occ_thresh": params.measurement_model_params.occ_thresh,
                "free_thresh": params.measurement_model_params.free_thresh,
                "unknown_ratio_thresh": params.measurement_model_params.unknown_ratio_thresh,
                "known_free_ratio_thresh": params.measurement_model_params.known_free_ratio_thresh,

                "sigma_hit": params.measurement_model_params.sigma_hit,
                "w_hit": params.measurement_model_params.w_hit,
                "w_short": params.measurement_model_params.w_short,
                "lambda_short": params.measurement_model_params.lambda_short,
                "w_max": params.measurement_model_params.w_max,
                "w_rand": params.measurement_model_params.w_rand,
                
                "p_unknown": params.measurement_model_params.p_unknown,
                "p_out_of_map": params.measurement_model_params.p_out_of_map,
                "p_unexpected_known_free": params.measurement_model_params.p_unexpected_known_free,
                "p_pred_below_min": params.measurement_model_params.p_pred_below_min,
                
                "alpha_meas": params.measurement_model_params.alpha_meas,
                "beam_step": params.measurement_model_params.beam_step,
                "eps": params.measurement_model_params.eps,

                "sigma_x_motion": params.motion_model_params.sigma_x,
                "sigma_y_motion": params.motion_model_params.sigma_y,
                "sigma_theta_motion": params.motion_model_params.sigma_theta,
                "ctrl_motion_fac": params.motion_model_params.ctrl_motion_fac,
                "ctrl_turn_fac": params.motion_model_params.ctrl_turn_fac,
                "neff_threshold": params.neff_threshold,
                "proposal_sigma_xy": params.proposal_sigma_xy,
                "proposal_sigma_theta": params.proposal_sigma_theta,
                "n_samples_dir": params.proposal_n_samples,
                "meas_kernel_size": params.meas_kernel_size,

                # Scan matcher information metrics
                "scan_match_failed_count": cls._read_from_summary(run, "scan_match_failed_count", scan_match_failed_count),
                "scan_match_fallback_failed_count": cls._read_from_summary(run, "scan_match_fallback_failed_count", scan_match_fallback_failed_count),
                "scan_match_failed_rate": cls._read_from_summary(run, "scan_match_failed_rate", scan_match_failed_rate),
                "scan_match_fallback_failed_rate": cls._read_from_summary(run, "scan_match_fallback_failed_rate", scan_match_fallback_failed_rate),
                # "median_extracted_map_points": cls._read_from_summary(run, "median_extracted_map_points"),
                # "median_map_point_keep_ratio": cls._read_from_summary(run, "median_map_point_keep_ratio"),
                "count_too_few_points": cls._read_from_summary(run, "count_too_few_points", 0),
                "count_too_few_corresp": cls._read_from_summary(run, "count_too_few_corresp", 0),
                "infinite_h_or_g": cls._read_from_summary(run, "infinite_h_or_g", 0),
                "ill_cond_H": cls._read_from_summary(run, "ill_cond_H", 0),
                "infinite_dtransform": cls._read_from_summary(run, "infinite_dtransform", 0),
                "infinite_mean_err": cls._read_from_summary(run, "infinite_mean_err", 0),
                "best_transf_too_large": cls._read_from_summary(run, "best_transf_too_large", 0),
                "best_mean_err_too_large": cls._read_from_summary(run, "best_mean_err_too_large", 0),

                # Pose errs
                # trans errs
                # Mean trans errs
                "mean_trans_err_raw_odom": cls._read_from_summary(run, "mean_trans_err_raw_odom"),
                "mean_trans_err_weighted_mean": cls._read_from_summary(run, "mean_trans_err_weighted_mean"),
                "mean_trans_err_best_particle": cls._read_from_summary(run, "mean_trans_err_best_particle"),
                "mean_trans_err_closest_p_before_resampling": cls._read_from_summary(run, "mean_trans_err_closest_p_before_resampling"),
                "mean_trans_err_map_traj": cls._read_from_summary(run, "mean_trans_err_map_traj"),

                # RMSE trans errs
                "rmse_trans_err_raw_odom": cls._read_from_summary(run, "rmse_trans_err_raw_odom"),
                "rmse_trans_err_weighted_mean": cls._read_from_summary(run, "rmse_trans_err_weighted_mean"),
                "rmse_trans_err_best_particle": cls._read_from_summary(run, "rmse_trans_err_best_particle"),
                "rmse_trans_err_closest_p_before_resampling": cls._read_from_summary(run, "rmse_trans_err_closest_p_before_resampling"),
                "rmse_trans_err_map_traj": cls._read_from_summary(run, "rmse_trans_err_map_traj"),

                # Worst trans errs
                "worst_trans_err_raw_odom": cls._read_from_summary(run, "worst_trans_err_raw_odom"),
                "worst_trans_err_weighted_mean": cls._read_from_summary(run, "worst_trans_err_weighted_mean"),
                "worst_trans_err_best_particle": cls._read_from_summary(run, "worst_trans_err_best_particle"),
                "worst_trans_err_closest_p_before_resampling": cls._read_from_summary(run, "worst_trans_err_closest_p_before_resampling"),
                "worst_trans_err_map_traj": cls._read_from_summary(run, "worst_trans_err_map_traj"),

                # rot errs
                # Mean rot errs
                "mean_rot_err_raw_odom_deg": cls._deg(
                    cls._read_from_summary(run, "mean_rot_err_raw_odom")
                ),
                "mean_rot_err_weighted_mean_deg": cls._deg(
                    cls._read_from_summary(run, "mean_rot_err_weighted_mean")
                ),
                "mean_rot_err_best_particle_deg": cls._deg(
                    cls._read_from_summary(run, "mean_rot_err_best_particle")
                ),
                "mean_rot_err_closest_p_before_resampling_deg": cls._deg(
                    cls._read_from_summary(run, "mean_rot_err_closest_p_before_resampling")
                ),
                "mean_rot_err_map_traj_deg": cls._deg(
                    cls._read_from_summary(run, "mean_rot_err_map_traj")
                ),

                # RMSE rot errs
                "rmse_rot_err_raw_odom_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_rot_err_raw_odom")
                ),
                "rmse_rot_err_weighted_mean_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_rot_err_weighted_mean")
                ),
                "rmse_rot_err_best_particle_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_rot_err_best_particle")
                ),
                "rmse_rot_err_closest_p_before_resampling_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_rot_err_closest_p_before_resampling")
                ),
                "rmse_rot_err_map_traj_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_rot_err_map_traj")
                ),

                # Worst rot errs
                "worst_rot_err_raw_odom_deg": cls._deg(
                    cls._read_from_summary(run, "worst_rot_err_raw_odom")
                ),
                "worst_rot_err_weighted_mean_deg": cls._deg(
                    cls._read_from_summary(run, "worst_rot_err_weighted_mean")
                ),
                "worst_rot_err_best_particle_deg": cls._deg(
                    cls._read_from_summary(run, "worst_rot_err_best_particle")
                ),
                "worst_rot_err_closest_p_before_resampling_deg": cls._deg(
                    cls._read_from_summary(run, "worst_rot_err_closest_p_before_resampling")
                ),
                "worst_rot_err_map_traj_deg": cls._deg(
                    cls._read_from_summary(run, "worst_rot_err_map_traj")
                ),

                # Final drift
                # Trans drift
                "final_trans_drift_trans_err_raw_odom": cls._read_from_summary(
                    run, "final_trans_drift_trans_err_raw_odom"
                ),
                "final_trans_drift_trans_err_weighted_mean": cls._read_from_summary(
                    run, "final_trans_drift_trans_err_weighted_mean"
                ),
                "final_trans_drift_trans_err_best_particle": cls._read_from_summary(
                    run, "final_trans_drift_trans_err_best_particle"
                ),
                "final_trans_drift_trans_err_closest_p_before_resampling": cls._read_from_summary(
                    run, "final_trans_drift_trans_err_closest_p_before_resampling"
                ),
                "final_trans_drift_trans_err_map_traj": cls._read_from_summary(
                    run, "final_trans_drift_trans_err_map_traj"
                ),
                # Rot drift
                "final_rot_drift_rot_err_raw_odom_deg": cls._deg(
                    cls._read_from_summary(run, "final_rot_drift_rot_err_raw_odom")
                ),
                "final_rot_drift_rot_err_weighted_mean_deg": cls._deg(
                    cls._read_from_summary(run, "final_rot_drift_rot_err_weighted_mean")
                ),
                "final_rot_drift_rot_err_best_particle_deg": cls._deg(
                    cls._read_from_summary(run, "final_rot_drift_rot_err_best_particle")
                ),
                "final_rot_drift_rot_err_closest_p_before_resampling_deg": cls._deg(
                    cls._read_from_summary(run, "final_rot_drift_rot_err_closest_p_before_resampling")
                ),
                "final_rot_drift_rot_err_map_traj_deg": cls._deg(
                    cls._read_from_summary(run, "final_rot_drift_rot_err_map_traj")
                ),


                # Translational/rot rates of map traj
                "rate_above_thres_trans_err_map_traj": cls._read_from_summary(
                    run, "rate_above_thres_trans_err_map_traj"
                ),
                "rate_above_thres_rot_err_map_traj": cls._read_from_summary(
                    run, "rate_above_thres_rot_err_map_traj"
                ),

                # Raw odom improvement
                "median_trans_err_map_traj_impr_over_raw_odom": cls._read_from_summary(
                    run, "median_trans_err_map_traj_impr_over_raw_odom"
                ),
                "median_rot_err_map_traj_impr_over_raw_odom": cls._read_from_summary(
                    run, "median_rot_err_map_traj_impr_over_raw_odom"
                ),

                # Positive Error slopes
                "p90_pos_trans_err_slopes_map_traj": cls._read_from_summary(
                    run, "p90_pos_trans_err_slopes_map_traj"
                ),
                "p90_pos_rot_err_slopes_map_traj_deg": cls._deg(
                    cls._read_from_summary(run, "p90_pos_rot_err_slopes_map_traj")
                ),

                # Motion errors
                "p90_trans_motion_err_map_traj": cls._read_from_summary(
                    run, "p90_trans_motion_err_map_traj"
                ),
                "p90_rot_motion_err_map_traj_deg": cls._deg(
                    cls._read_from_summary(run, "p90_rot_motion_err_map_traj")
                ),


                # Correlations
                "median_corr_trans_weights_pos": cls._read_from_summary(
                    run, "median_corr_trans_weights_pos"
                ),
                "median_corr_rot_weights_pos": cls._read_from_summary(
                    run, "median_corr_rot_weights_pos"
                ),
                "rate_gap_trans_best_to_min_before_resamp_above_max_trans_gap": cls._read_from_summary(
                    run, "rate_gap_trans_best_to_min_before_resamp_above_max_trans_gap"
                ),
                "rate_gap_rot_best_to_min_before_resamp_above_max_rot_gap": cls._read_from_summary(
                    run, "rate_gap_rot_best_to_min_before_resamp_above_max_rot_gap"
                ),

            }

            rows.append(row)

        return pd.DataFrame(rows)


class ResultAggregator:
    """
    Aggregate the given dataframe run results into the desired output types.
    """
    @staticmethod
    def _groupby(df: pd.DataFrame, by, dropna: bool = False):
        """
        Pandas compatibility wrapper.

        Older pandas versions (e.g. in some ROS Python environments) do not
        support the dropna argument for DataFrame.groupby.
        """
        try:
            return df.groupby(by, dropna=dropna)
        except TypeError:
            return df.groupby(by)


    @staticmethod
    def _place_col_after_col(df: pd.DataFrame, col: str, col_after: str) -> pd.DataFrame:
        extract_col = df.pop(col)
        df.insert(df.columns.get_loc(col_after) + 1, col, extract_col)
        return df


    def rank_by_score(self, ranked_run_df: pd.DataFrame, score_col: str, ascending=True) -> pd.DataFrame:
        '''
        Sorts the given df by score in the desired order (asc or desc).
        '''   
        # Sort df by score values 
        ranked_df: pd.DataFrame = ranked_run_df.sort_values(by=score_col, ascending=ascending).reset_index(drop=True)

        # Check if rank col exists, if so delete
        if "rank" in ranked_run_df.columns:
            ranked_df = ranked_df.drop(columns=["rank"])

        # Add rank col as first column
        ranked_df.insert(0, "rank", ranked_df.index + 1)
        
        return ranked_df


    def aggregate_by_dataset_and_param(self, ranked_run_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Groups the run-level dataframe by dataset_id and parameter_hash and aggregates all metrics that are
        currently written by RankedRunConverter.to_dataframe(...).

        The output is one row per dataset/parameter-set/meas-model combination. The method keeps the same basic
        structure as the old aggregator:
            1) define required columns
            2) check missing columns
            3) group and aggregate
            4) compute pooled rates where raw counts and denominators are available
            5) compute dataset_param_score
            6) rank by dataset_param_score
        '''
        
        # 1) Required columns
        # _____________________________________________________________________________________

        # Define required columns needed for aggrgation stage
        required_cols = [
            # Grouping columns
            "dataset_id",
            "parameter_hash",
            "used_meas_model",

            # General information
            "map",
            "parameter_tag",
            "score",
            "seed",
            "n_steps",

            # Scan matcher metrics
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

            # Translation errors
            "mean_trans_err_raw_odom",
            "mean_trans_err_weighted_mean",
            "mean_trans_err_best_particle",
            "mean_trans_err_closest_p_before_resampling",
            "mean_trans_err_map_traj",
            "rmse_trans_err_raw_odom",
            "rmse_trans_err_weighted_mean",
            "rmse_trans_err_best_particle",
            "rmse_trans_err_closest_p_before_resampling",
            "rmse_trans_err_map_traj",
            "worst_trans_err_raw_odom",
            "worst_trans_err_weighted_mean",
            "worst_trans_err_best_particle",
            "worst_trans_err_closest_p_before_resampling",
            "worst_trans_err_map_traj",

            # Rotation errors, already converted to degree by to_dataframe()
            "mean_rot_err_raw_odom_deg",
            "mean_rot_err_weighted_mean_deg",
            "mean_rot_err_best_particle_deg",
            "mean_rot_err_closest_p_before_resampling_deg",
            "mean_rot_err_map_traj_deg",
            "rmse_rot_err_raw_odom_deg",
            "rmse_rot_err_weighted_mean_deg",
            "rmse_rot_err_best_particle_deg",
            "rmse_rot_err_closest_p_before_resampling_deg",
            "rmse_rot_err_map_traj_deg",
            "worst_rot_err_raw_odom_deg",
            "worst_rot_err_weighted_mean_deg",
            "worst_rot_err_best_particle_deg",
            "worst_rot_err_closest_p_before_resampling_deg",
            "worst_rot_err_map_traj_deg",

            # Final drift
            "final_trans_drift_trans_err_raw_odom",
            "final_trans_drift_trans_err_weighted_mean",
            "final_trans_drift_trans_err_best_particle",
            "final_trans_drift_trans_err_closest_p_before_resampling",
            "final_trans_drift_trans_err_map_traj",
            "final_rot_drift_rot_err_raw_odom_deg",
            "final_rot_drift_rot_err_weighted_mean_deg",
            "final_rot_drift_rot_err_best_particle_deg",
            "final_rot_drift_rot_err_closest_p_before_resampling_deg",
            "final_rot_drift_rot_err_map_traj_deg",

            # Scorer metrics / additional evaluator metrics
            "rate_above_thres_trans_err_map_traj",
            "rate_above_thres_rot_err_map_traj",
            "median_trans_err_map_traj_impr_over_raw_odom",
            "median_rot_err_map_traj_impr_over_raw_odom",
            "p90_pos_trans_err_slopes_map_traj",
            "p90_pos_rot_err_slopes_map_traj_deg",
            "p90_trans_motion_err_map_traj",
            "p90_rot_motion_err_map_traj_deg",
            "median_corr_trans_weights_pos",
            "median_corr_rot_weights_pos",
            "rate_gap_trans_best_to_min_before_resamp_above_max_trans_gap",
            "rate_gap_rot_best_to_min_before_resamp_above_max_rot_gap",
        ]

        # Estimate missing columns
        missing = [col for col in required_cols if col not in ranked_run_df.columns]
        if missing:
            raise ValueError(
                "aggregate_by_dataset_and_param missing required columns: " + ", ".join(missing)
            )
        
        
        # 2) Build aggregation specification in the desired output order
        # _____________________________________________________________________________________

        agg_spec = {}

        # Define agg methods to use for the columns
        agg_spec["map"] = ("map", "first")
        agg_spec["parameter_tag"] = ("parameter_tag", "first")
        agg_spec["n_runs"] = ("score", "size")
        agg_spec["n_seeds"] = ("seed", "nunique")
        agg_spec["total_n_steps"] = ("n_steps", "sum")


        # Score statistics. The aggregation score is intentionally based on the scorer output itself.
        agg_spec["mean_score"] = ("score", "mean")
        agg_spec["best_score"] = ("score", "min")
        agg_spec["worst_score"] = ("score", "max")
        agg_spec["std_score"] = ("score", "std")

        # Scan matcher information.
        # Counts are summed because they are event counts. Rates are also aggregated as mean/worst because
        # n_steps may not be present in the dataframe. If n_steps exists, pooled rates are computed after groupby.
        agg_spec["total_scan_match_failed_count"] = ("scan_match_failed_count", "sum")
        agg_spec["total_scan_match_fallback_failed_count"] = ("scan_match_fallback_failed_count", "sum")
        
        scan_match_count_cols = [
            "count_too_few_points",
            "count_too_few_corresp",
            "infinite_h_or_g",
            "ill_cond_H",
            "infinite_dtransform",
            "infinite_mean_err",
            "best_transf_too_large",
            "best_mean_err_too_large",
        ]
        for col in scan_match_count_cols:
            agg_spec[f"total_{col}"] = (col, "sum")

        # Pose error trajectories.
        pose_sources = [
            "raw_odom",
            "weighted_mean",
            "best_particle",
            "closest_p_before_resampling",
            "map_traj",
        ]

        # Aggregate translational errors
        # trans
        for src in pose_sources:
            agg_spec[f"mean_trans_err_{src}"] = (f"mean_trans_err_{src}", "mean")
        # rmse
        for src in pose_sources:
            agg_spec[f"mean_rmse_trans_err_{src}"] = (f"rmse_trans_err_{src}", "mean")
            agg_spec[f"worst_rmse_trans_err_{src}"] = (f"rmse_trans_err_{src}", "max")
        # worst
        # for src in pose_sources:
        #     agg_spec[f"worst_trans_err_{src}"] = (f"worst_trans_err_{src}", "max")

        # Aggregate Rotational errors in degrees.
        for src in pose_sources:
            agg_spec[f"mean_rot_err_{src}_deg"] = (f"mean_rot_err_{src}_deg", "mean")
        for src in pose_sources:
            agg_spec[f"mean_rmse_rot_err_{src}_deg"] = (f"rmse_rot_err_{src}_deg", "mean")
            agg_spec[f"worst_rmse_rot_err_{src}_deg"] = (f"rmse_rot_err_{src}_deg", "max")
        # for src in pose_sources:
        #     agg_spec[f"max_rot_err_{src}_deg"] = (f"worst_rot_err_{src}_deg", "max")

        # Aggregate Final drift 
        # trans
        for src in pose_sources:
            agg_spec[f"mean_final_trans_drift_trans_err_{src}"] = (
                f"final_trans_drift_trans_err_{src}",
                "mean",
            )
            agg_spec[f"worst_final_trans_drift_trans_err_{src}"] = (
                f"final_trans_drift_trans_err_{src}",
                "max",
            )
        # rot
        for src in pose_sources:
            agg_spec[f"mean_final_rot_drift_rot_err_{src}_deg"] = (
                f"final_rot_drift_rot_err_{src}_deg",
                "mean",
            )
            agg_spec[f"worst_final_rot_drift_rot_err_{src}_deg"] = (
                f"final_rot_drift_rot_err_{src}_deg",
                "max",
            )

        # MAP trajectory threshold rates. No raw counts are stored, so we aggregate the run-level rates.
        agg_spec["mean_rate_above_thres_trans_err_map_traj"] = (
            "rate_above_thres_trans_err_map_traj",
            "mean",
        )
        agg_spec["worst_rate_above_thres_trans_err_map_traj"] = (
            "rate_above_thres_trans_err_map_traj",
            "max",
        )
        agg_spec["mean_rate_above_thres_rot_err_map_traj"] = (
            "rate_above_thres_rot_err_map_traj",
            "mean",
        )
        agg_spec["worst_rate_above_thres_rot_err_map_traj"] = (
            "rate_above_thres_rot_err_map_traj",
            "max",
        )

        # Improvement metrics. Higher is better, so the worst run is the minimum improvement.
        agg_spec["mean_median_trans_err_map_traj_impr_over_raw_odom"] = (
            "median_trans_err_map_traj_impr_over_raw_odom",
            "mean",
        )
        agg_spec["worst_median_trans_err_map_traj_impr_over_raw_odom"] = (
            "median_trans_err_map_traj_impr_over_raw_odom",
            "min",
        )
        agg_spec["mean_median_rot_err_map_traj_impr_over_raw_odom"] = (
            "median_rot_err_map_traj_impr_over_raw_odom",
            "mean",
        )
        agg_spec["worst_median_rot_err_map_traj_impr_over_raw_odom"] = (
            "median_rot_err_map_traj_impr_over_raw_odom",
            "min",
        )

        # Positive error slopes. Lower is better, so the worst run is the maximum value.
        agg_spec["mean_p90_pos_trans_err_slopes_map_traj"] = (
            "p90_pos_trans_err_slopes_map_traj",
            "mean",
        )
        agg_spec["worst_p90_pos_trans_err_slopes_map_traj"] = (
            "p90_pos_trans_err_slopes_map_traj",
            "max",
        )
        agg_spec["mean_p90_pos_rot_err_slopes_map_traj_deg"] = (
            "p90_pos_rot_err_slopes_map_traj_deg",
            "mean",
        )
        agg_spec["worst_p90_pos_rot_err_slopes_map_traj_deg"] = (
            "p90_pos_rot_err_slopes_map_traj_deg",
            "max",
        )

        # Motion errors. Lower is better, so the worst run is the maximum value.
        agg_spec["mean_p90_trans_motion_err_map_traj"] = (
            "p90_trans_motion_err_map_traj",
            "mean",
        )
        agg_spec["worst_p90_trans_motion_err_map_traj"] = (
            "p90_trans_motion_err_map_traj",
            "max",
        )
        agg_spec["mean_p90_rot_motion_err_map_traj_deg"] = (
            "p90_rot_motion_err_map_traj_deg",
            "mean",
        )
        agg_spec["worst_p90_rot_motion_err_map_traj_deg"] = (
            "p90_rot_motion_err_map_traj_deg",
            "max",
        )

        # Particle weighting / selection quality. These are already badness-like rates/metrics, so max is worst.
        agg_spec["mean_median_corr_trans_weights_pos"] = ("median_corr_trans_weights_pos", "mean")
        agg_spec["worst_median_corr_trans_weights_pos"] = ("median_corr_trans_weights_pos", "max")
        agg_spec["mean_median_corr_rot_weights_pos"] = ("median_corr_rot_weights_pos", "mean")
        agg_spec["worst_median_corr_rot_weights_pos"] = ("median_corr_rot_weights_pos", "max")
        agg_spec["mean_rate_gap_trans_best_to_min_before_resamp_above_max_trans_gap"] = (
            "rate_gap_trans_best_to_min_before_resamp_above_max_trans_gap",
            "mean",
        )
        agg_spec["worst_rate_gap_trans_best_to_min_before_resamp_above_max_trans_gap"] = (
            "rate_gap_trans_best_to_min_before_resamp_above_max_trans_gap",
            "max",
        )
        agg_spec["mean_rate_gap_rot_best_to_min_before_resamp_above_max_rot_gap"] = (
            "rate_gap_rot_best_to_min_before_resamp_above_max_rot_gap",
            "mean",
        )
        agg_spec["worst_rate_gap_rot_best_to_min_before_resamp_above_max_rot_gap"] = (
            "rate_gap_rot_best_to_min_before_resamp_above_max_rot_gap",
            "max",
        )
        
        # 3) Group and aggregate
        # _____________________________________________________________________________________

        agg_dataset_param_df = self._groupby(
            ranked_run_df,
            ["dataset_id", "parameter_hash", "used_meas_model"],
        ).agg(**agg_spec).reset_index()

     
        # Place map name directly after dataset id, like in the old aggregator.
        agg_dataset_param_df = self._place_col_after_col(
            df=agg_dataset_param_df,
            col="map",
            col_after="dataset_id",
        )

        
        # 4) Compute pooled rates where possible
        # _____________________________________________________________________________________

        def _safe_div(numerator_col: str, denominator_col: str) -> pd.Series:
            numerator = pd.to_numeric(agg_dataset_param_df[numerator_col], errors="coerce")
            denominator = pd.to_numeric(agg_dataset_param_df[denominator_col], errors="coerce")
            result = pd.Series(np.nan, index=agg_dataset_param_df.index, dtype=float)
            valid = denominator.gt(0) & numerator.ge(0)
            result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
            return result

        # Best option: pooled count / pooled number of steps.
        agg_dataset_param_df["scan_match_failed_rate"] = _safe_div(
            "total_scan_match_failed_count",
            "total_n_steps",
        )
        agg_dataset_param_df["scan_match_fallback_failed_rate"] = _safe_div(
            "total_scan_match_fallback_failed_count",
            "total_n_steps",
        )

        # Place main pooled/fallback scan-match rates next to their counts.
        agg_dataset_param_df = self._place_col_after_col(
            df=agg_dataset_param_df,
            col="scan_match_failed_rate",
            col_after="total_scan_match_fallback_failed_count",
        )
        agg_dataset_param_df = self._place_col_after_col(
            df=agg_dataset_param_df,
            col="scan_match_fallback_failed_rate",
            col_after="scan_match_failed_rate",
        )

        
        # 5) Compute dataset-level parameter score
        # _____________________________________________________________________________________

        agg_dataset_param_df["std_score"] = agg_dataset_param_df["std_score"].fillna(0.0)

        dataset_param_score = (
            1.0 * agg_dataset_param_df["mean_score"]
            + 0.5 * agg_dataset_param_df["worst_score"]
            + 0.2 * agg_dataset_param_df["std_score"]
        )

        agg_dataset_param_df.insert(0, "dataset_param_score", dataset_param_score)

        return self.rank_by_score(agg_dataset_param_df, "dataset_param_score", ascending=True)


    def aggregate_by_params(self, agg_dataset_param_df: pd.DataFrame):
        # Estimate if all needed columns exist
        required_cols = [
            # Grouping column
            "parameter_hash",

            # General information
            "parameter_tag",
            "used_meas_model",
            "dataset_id",
            "dataset_param_score",
            "total_n_steps",
            "measurement_stddev",

            # Translational pose errors
            "mean_trans_err",
            "mean_trans_err_sm_true",
            "mean_trans_err_mu_true",

            "mean_rmse_trans_err_sm_true",
            "mean_rmse_trans_err_mu_true",
            "mean_rmse_trans_error",

            "worst_rmse_trans_err_sm_true",
            "worst_rmse_trans_err_mu_true",
            "worst_rmse_trans_error",

            "max_trans_err_sm_true",
            "max_trans_err_mu_true",
            "max_trans_err",

            # Rotational pose errors
            "mean_rot_err_sm_true_deg",
            "mean_rot_err_mu_true_deg",
            "mean_rot_error_deg",

            "mean_rmse_rot_err_sm_true_deg",
            "mean_rmse_rot_err_mu_true_deg",
            "mean_rmse_rot_error_deg",

            "worst_rmse_rot_err_sm_true_deg",
            "worst_rmse_rot_err_mu_true_deg",
            "worst_rmse_rot_error_deg",

            "max_rot_err_sm_true_deg",
            "max_rot_err_mu_true_deg",
            "max_rot_err_deg",

            # Scan-matcher information
            "scan_match_failed_rate",
            "scan_match_fallback_failed_rate",
            "median_extracted_map_points",
            "median_map_point_keep_ratio",

            # Proposal metrics
            "mean_mu_true_better_than_sm_true_rate",
            "worst_mu_true_better_than_sm_true_rate",

            "mean_min_xj_is_best_xj",
            "worst_min_xj_is_best_xj",

            "mean_min_xj_pose_err_true",
            "worst_min_xj_pose_err_true",

            "mean_best_weighted_xj_pose_err_true",
            "worst_best_weighted_xj_pose_err_true",

            "mean_min_xj_true_err_improves_over_sm_true",
            "worst_min_xj_true_err_improves_over_sm_true",

            "mean_min_xj_better_sm_pose_rate",
            "worst_min_xj_better_sm_pose_rate",

            "mean_best_xj_true_err_improves_over_sm_true",
            "worst_best_xj_true_err_improves_over_sm_true",

            "mean_best_xj_better_sm_pose_rate",
            "worst_best_xj_better_sm_pose_rate",

            "median_log_motion_range",
            "median_log_meas_range",

            # XJ error metrics
            "mean_min_trans_err_xjs",
            "mean_min_xj_trans_err_true",
            "mean_best_xj_trans_err_true",
            "mean_rmse_min_trans_err_xjs",
            "mean_rmse_min_xj_trans_err_true",
            "mean_rmse_best_xj_trans_err_true",
            "mean_min_rot_err_xjs_deg",
            "mean_min_xj_rot_err_true_deg",
            "mean_best_xj_rot_err_true_deg",
            "mean_rmse_min_rot_err_xjs_deg",
            "mean_rmse_min_xj_rot_err_true_deg",
            "mean_rmse_best_xj_rot_err_true_deg",

            "mean_corr_xjs_motion",
            "worst_corr_xjs_motion",
            "mean_corr_xjs_meas",
            "worst_corr_xjs_meas",

            "mean_corr_xj_trans_motion",
            "worst_corr_xj_trans_motion",
            "mean_corr_xj_trans_meas",
            "worst_corr_xj_trans_meas",
            "mean_corr_xj_rot_motion",
            "worst_corr_xj_rot_motion",
            "mean_corr_xj_rot_meas",
            "worst_corr_xj_rot_meas",

            # Measurement model metrics
            # Proposal rates
            "meas_model_prop_map_hit_rate",
            "meas_model_prop_out_of_map_rate",
            "meas_model_prop_no_map_hit_rate",
            "meas_model_prop_unknown_no_map_hit_rate",
            "meas_model_prop_known_free_no_map_hit_rate",
            "meas_model_prop_unexpected_known_free_rate",

            # Fallback rates
            "meas_model_fallback_map_hit_rate",
            "meas_model_fallback_out_of_map_rate",
            "meas_model_fallback_no_map_hit_rate",
            "meas_model_fallback_unknown_no_map_hit_rate",
            "meas_model_fallback_known_free_no_map_hit_rate",
            "meas_model_fallback_unexpected_known_free_rate",
        ]

        # Estimate missing columns
        missing = [col for col in required_cols if col not in agg_dataset_param_df.columns]
        if missing:
            raise ValueError(
                "aggregate_by_params missing required columns: " + ", ".join(missing)
            )
        
        # Compute metrics from grouped columns
        agg_param_df = self._groupby(agg_dataset_param_df, ["parameter_hash"])
        agg_param_df: pd.DataFrame = agg_param_df.agg(
            # General info
            parameter_tag=("parameter_tag", "first"),
            used_meas_model=("used_meas_model", "first"),
            n_datasets=("dataset_id", "nunique"),
            n_results=("dataset_param_score", "size"),

            total_n_steps=("total_n_steps", "sum"),    
            measurement_stddev=("measurement_stddev", "first"),        
            
            # Metrics for score computation
            mean_score=("dataset_param_score", "mean"),
            worst_score=("dataset_param_score", "max"),
            std_score=("dataset_param_score", "std"),

            # Pose errors
            # Translational err
            # mean
            mean_trans_err=("mean_trans_err", "mean"),
            mean_trans_err_sm_true=("mean_trans_err_sm_true", "mean"),
            mean_trans_err_mu_true=("mean_trans_err_mu_true", "mean"),
            # rmse
            mean_rmse_trans_err_sm_true=("mean_rmse_trans_err_sm_true", "mean"),
            mean_rmse_trans_err_mu_true=("mean_rmse_trans_err_mu_true", "mean"),
            mean_rmse_trans_error=("mean_rmse_trans_error", "mean"),
            # worst rmse
            worst_rmse_trans_err_sm_true=("worst_rmse_trans_err_sm_true", "max"),
            worst_rmse_trans_err_mu_true=("worst_rmse_trans_err_mu_true", "max"),
            worst_rmse_trans_error=("worst_rmse_trans_error", "max"),
            # max values (These are the max vals over all steps for all runs)
            max_trans_err_sm_true=("max_trans_err_sm_true", "max"),
            max_trans_err_mu_true=("max_trans_err_mu_true", "max"),
            max_trans_err=("max_trans_err", "max"),

            # Rotational errors
            # mean
            mean_rot_err_sm_true_deg=("mean_rot_err_sm_true_deg", "mean"),
            mean_rot_err_mu_true_deg=("mean_rot_err_mu_true_deg", "mean"),
            mean_rot_error_deg=("mean_rot_error_deg", "mean"),
            # rmse
            mean_rmse_rot_err_sm_true_deg=("mean_rmse_rot_err_sm_true_deg", "mean"),
            mean_rmse_rot_err_mu_true_deg=("mean_rmse_rot_err_mu_true_deg", "mean"),
            mean_rmse_rot_error_deg=("mean_rmse_rot_error_deg", "mean"),
            # worst
            worst_rmse_rot_err_sm_true_deg=("worst_rmse_rot_err_sm_true_deg", "max"),
            worst_rmse_rot_err_mu_true_deg=("worst_rmse_rot_err_mu_true_deg", "max"),
            worst_rmse_rot_error_deg=("worst_rmse_rot_error_deg", "max"),
            # max values
            max_rot_err_sm_true_deg=("max_rot_err_sm_true_deg", "max"),
            max_rot_err_mu_true_deg=("max_rot_err_mu_true_deg", "max"),
            max_rot_err_deg=("max_rot_err_deg", "max"),

            # Scan matcher info
            # Here we dont compute the rates manually because we aggregate over different maps and don't
            # want to have big influence in unioned metrics because if big maps comapred to small maps
            mean_scan_match_failed_rate=("scan_match_failed_rate", "mean"),
            worst_scan_match_failed_rate=("scan_match_failed_rate", "max"),
            
            mean_scan_match_fallback_failed_rate=("scan_match_fallback_failed_rate", "mean"),
            worst_scan_match_fallback_failed_rate=("scan_match_fallback_failed_rate", "max"),

            median_extracted_map_points=("median_extracted_map_points", "median"),
            median_map_point_keep_ratio=("median_map_point_keep_ratio", "median"),
                        
            # Proposal information metrics
            mean_mu_true_better_than_sm_true_rate=("mean_mu_true_better_than_sm_true_rate", "mean"),
            worst_mu_true_better_than_sm_true_rate=("worst_mu_true_better_than_sm_true_rate", "min"),

            mean_min_xj_is_best_xj=("mean_min_xj_is_best_xj", "mean"),
            worst_min_xj_is_best_xj=("worst_min_xj_is_best_xj", "min"),

            mean_min_xj_pose_err_true=("mean_min_xj_pose_err_true", "mean"),
            worst_min_xj_pose_err_true=("worst_min_xj_pose_err_true", "max"),

            mean_best_weighted_xj_pose_err_true=("mean_best_weighted_xj_pose_err_true", "mean"),
            worst_best_weighted_xj_pose_err_true=("worst_best_weighted_xj_pose_err_true", "max"),

            mean_min_xj_true_err_improves_over_sm_true =("mean_min_xj_true_err_improves_over_sm_true", "mean"),
            worst_min_xj_true_err_improves_over_sm_true =("worst_min_xj_true_err_improves_over_sm_true", "min"),

            mean_min_xj_better_sm_pose_rate =("mean_min_xj_better_sm_pose_rate", "mean"),
            worst_min_xj_better_sm_pose_rate =("worst_min_xj_better_sm_pose_rate", "min"),

            mean_best_xj_true_err_improves_over_sm_true =("mean_best_xj_true_err_improves_over_sm_true", "mean"),
            worst_best_xj_true_err_improves_over_sm_true =("worst_best_xj_true_err_improves_over_sm_true", "min"),

            mean_best_xj_better_sm_pose_rate =("mean_best_xj_better_sm_pose_rate", "mean"),
            worst_best_xj_better_sm_pose_rate =("worst_best_xj_better_sm_pose_rate", "min"),

            median_log_motion_range=("median_log_motion_range", "median"),
            median_log_meas_range=("median_log_meas_range", "median"),

            # XJ error metrics
            mean_min_trans_err_xjs=("mean_min_trans_err_xjs", "mean"),
            mean_min_xj_trans_err_true=("mean_min_xj_trans_err_true", "mean"),
            mean_best_xj_trans_err_true=("mean_best_xj_trans_err_true", "mean"),
            mean_rmse_min_trans_err_xjs=("mean_rmse_min_trans_err_xjs", "mean"),
            mean_rmse_min_xj_trans_err_true=("mean_rmse_min_xj_trans_err_true", "mean"),
            mean_rmse_best_xj_trans_err_true=("mean_rmse_best_xj_trans_err_true", "mean"),
            mean_min_rot_err_xjs_deg=("mean_min_rot_err_xjs_deg", "mean"),
            mean_min_xj_rot_err_true_deg=("mean_min_xj_rot_err_true_deg", "mean"),
            mean_best_xj_rot_err_true_deg=("mean_best_xj_rot_err_true_deg", "mean"),
            mean_rmse_min_rot_err_xjs_deg=("mean_rmse_min_rot_err_xjs_deg", "mean"),
            mean_rmse_min_xj_rot_err_true_deg=("mean_rmse_min_xj_rot_err_true_deg", "mean"),
            mean_rmse_best_xj_rot_err_true_deg=("mean_rmse_best_xj_rot_err_true_deg", "mean"),

            mean_corr_xjs_motion=("mean_corr_xjs_motion", "mean"),
            worst_corr_xjs_motion=("worst_corr_xjs_motion", "min"),
            mean_corr_xjs_meas=("mean_corr_xjs_meas", "mean"),
            worst_corr_xjs_meas=("worst_corr_xjs_meas", "min"),        

            mean_corr_xj_trans_motion=("mean_corr_xj_trans_motion", "mean"),
            mean_corr_xj_trans_meas=("mean_corr_xj_trans_meas", "mean"),
            worst_corr_xj_trans_motion=("worst_corr_xj_trans_motion", "min"),
            worst_corr_xj_trans_meas=("worst_corr_xj_trans_meas", "min"),

            mean_corr_xj_rot_motion=("mean_corr_xj_rot_motion", "mean"),
            mean_corr_xj_rot_meas=("mean_corr_xj_rot_meas", "mean"),
            worst_corr_xj_rot_motion=("worst_corr_xj_rot_motion", "min"),
            worst_corr_xj_rot_meas=("worst_corr_xj_rot_meas", "min"),

            # measurement model metrics
            # Proposal rates
            mean_meas_model_prop_map_hit_rate=(
                "meas_model_prop_map_hit_rate",
                "mean",
            ),
            mean_meas_model_prop_out_of_map_rate=(
                "meas_model_prop_out_of_map_rate",
                "mean",
            ),
            mean_meas_model_prop_no_map_hit_rate=(
                "meas_model_prop_no_map_hit_rate",
                "mean",
            ),
            mean_meas_model_prop_unknown_no_map_hit_rate=(
                "meas_model_prop_unknown_no_map_hit_rate",
                "mean",
            ),
            mean_meas_model_prop_known_free_no_map_hit_rate=(
                "meas_model_prop_known_free_no_map_hit_rate",
                "mean",
            ),
            mean_meas_model_prop_unexpected_known_free_rate=(
                "meas_model_prop_unexpected_known_free_rate",
                "mean",
            ),
            # Worst proposal rates
            worst_meas_model_prop_out_of_map_rate=(
                "meas_model_prop_out_of_map_rate",
                "max",
            ),
            worst_meas_model_prop_unknown_no_map_hit_rate=(
                "meas_model_prop_unknown_no_map_hit_rate",
                "max",
            ),
            worst_meas_model_prop_unexpected_known_free_rate=(
                "meas_model_prop_unexpected_known_free_rate",
                "max",
            ),

            # Fallback rates
            mean_meas_model_fallback_map_hit_rate=(
                "meas_model_fallback_map_hit_rate",
                "mean",
            ),
            mean_meas_model_fallback_out_of_map_rate=(
                "meas_model_fallback_out_of_map_rate",
                "mean",
            ),
            mean_meas_model_fallback_no_map_hit_rate=(
                "meas_model_fallback_no_map_hit_rate",
                "mean",
            ),
            mean_meas_model_fallback_unknown_no_map_hit_rate=(
                "meas_model_fallback_unknown_no_map_hit_rate",
                "mean",
            ),
            mean_meas_model_fallback_known_free_no_map_hit_rate=(
                "meas_model_fallback_known_free_no_map_hit_rate",
                "mean",
            ),
            mean_meas_model_fallback_unexpected_known_free_rate=(
                "meas_model_fallback_unexpected_known_free_rate",
                "mean",
            ),

            # Worst fallback rates
            worst_meas_model_fallback_out_of_map_rate=(
                "meas_model_fallback_out_of_map_rate",
                "max",
            ),
            worst_meas_model_fallback_unknown_no_map_hit_rate=(
                "meas_model_fallback_unknown_no_map_hit_rate",
                "max",
            ),
            worst_meas_model_fallback_unexpected_known_free_rate=(
                "meas_model_fallback_unexpected_known_free_rate",
                "max",
            ),

            # Info on how many datasets needed to use fallback 
            n_datasets_with_fallback=(
                "meas_model_fallback_map_hit_rate",
                "count",
            ),

        ).reset_index()
        
        agg_param_df["std_score"] = agg_param_df["std_score"].fillna(0.0)
        
        # Compute score
        global_score = (
            1.0 * agg_param_df["mean_score"]
            + 0.5 * agg_param_df["worst_score"]
            + 0.2 * agg_param_df["std_score"]   
        )

        agg_param_df.insert(0, "global_score", global_score)

        return self.rank_by_score(agg_param_df, "global_score", ascending=True)


    def build_ranked_parameter_overview(
        self,
        agg_param_df: pd.DataFrame,
        ranked_runs: Iterable[RankedRun],
    ) -> pd.DataFrame:
        """
        Build a compact parameter overview table keyed by parameter_hash.

        Output columns are:
        rank, global_score, parameter_hash, followed by all ExperimentParams fields
        used for parameter identity (excluding start_pose and tag).
        """
        # Check if required columns exist
        required_cols = {"rank", "global_score", "parameter_hash"}
        missing = sorted(col for col in required_cols if col not in agg_param_df.columns)
        if missing:
            raise ValueError(
                "build_ranked_parameter_overview missing required columns: " + ", ".join(missing)
            )
        
        params_by_hash = {}
        for run in ranked_runs:
            # Check if param hash exists
            if run.parameter_hash is None:
                continue            
            
            # Generate parameter overview and store it
            params_for_hash = RBPFOptimizer.generate_params_for_hash(run.params)
            existing = params_by_hash.get(run.parameter_hash)

            if existing is None:
                params_by_hash[run.parameter_hash] = params_for_hash
            elif existing != params_for_hash:
                raise ValueError(
                    f"Inconsistent params detected for parameter_hash '{run.parameter_hash}'."
                )

        if not params_by_hash:
            return pd.DataFrame(columns=["rank", "global_score", "parameter_hash"])

        # Keep parameter column order stable and readable.
        param_cols = list(next(iter(params_by_hash.values())).keys())

        rows = []
        for _, row in agg_param_df.sort_values(by="rank", ascending=True).iterrows():
            param_hash = row["parameter_hash"]
            row_payload = {
                "rank": row["rank"],
                "global_score": row["global_score"],
                "parameter_hash": param_hash,
            }
            row_payload.update(params_by_hash.get(param_hash, {}))
            rows.append(row_payload)

        overview_df = pd.DataFrame(rows)

        ordered_cols = ["rank", "global_score", "parameter_hash"] + param_cols
        ordered_existing_cols = [col for col in ordered_cols if col in overview_df.columns]
        return overview_df[ordered_existing_cols]
