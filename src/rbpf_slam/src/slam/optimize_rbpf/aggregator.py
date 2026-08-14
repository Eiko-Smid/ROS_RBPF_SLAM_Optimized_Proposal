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
        Transform the given ranked runs into a pandas DataFrame for further analysis and reporting.
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

                # Parameters
                "seed": run.seed,
                "measurement_stddev": params.measurement_noise_stddev,
                "every_nth_beam_filter": params.every_nth_scan_filter,
                "every_nth_beam_map": params.every_nth_scan_map,
                "n_particles": params.particle_params.n_particles,
                # TODO: Adapt measurement model here!
                # "sigma_measurement": params.measurement_model_params.sigma_measurement,
                # "sigma_measurement":params.measurement_model_params.sigma_hit,
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
                "scan_match_failed_count": scan_match_failed_count,
                "scan_match_fallback_failed_count": scan_match_fallback_failed_count,
                "scan_match_failed_rate": scan_match_failed_rate,
                "scan_match_fallback_failed_rate": scan_match_fallback_failed_rate,
                "median_extracted_map_points": cls._read_from_summary(run, "median_extracted_map_points"),
                "median_map_point_keep_ratio": cls._read_from_summary(run, "median_map_point_keep_ratio"),
                "count_too_few_points": cls._read_from_summary(run, "count_too_few_points", 0),
                "count_too_few_corresp": cls._read_from_summary(run, "count_too_few_corresp", 0),
                "infinite_h_or_g": cls._read_from_summary(run, "infinite_h_or_g", 0),
                "ill_cond_H": cls._read_from_summary(run, "ill_cond_H", 0),
                "infinite_dtransform": cls._read_from_summary(run, "infinite_dtransform", 0),
                "infinite_mean_err": cls._read_from_summary(run, "infinite_mean_err", 0),
                "best_transf_too_large": cls._read_from_summary(run, "best_transf_too_large", 0),
                "best_mean_err_too_large": cls._read_from_summary(run, "best_mean_err_too_large", 0),

                # Pose errors
                # Translational errors
                # mean
                "mean_trans_error_raw_odom": cls._read_from_summary(run, "mean_translation_error_raw_odom"),                
                "mean_trans_err_sm_true": cls._read_from_summary(run, "mean_trans_err_sm_true"),
                "mean_trans_err_mu_true": cls._read_from_summary(run, "mean_trans_err_mu_true"),
                "mean_trans_error": cls._read_from_summary(run, "mean_translation_error"),
                # "mean_trans_err_mu_sm": cls._summary(run, "mean_trans_err_mu_sm"),
                
                # rmse
                "rmse_trans_error_raw_odom": cls._read_from_summary(run, "rmse_translation_error_raw_odom"),
                "rmse_trans_err_sm_true": cls._read_from_summary(run, "rmse_trans_err_sm_true"),
                "rmse_trans_err_mu_true": cls._read_from_summary(run, "rmse_trans_err_mu_true"),
                "rmse_trans_error": cls._read_from_summary(run, "rmse_translation_error"),            
                # "rmse_trans_err_mu_sm": cls._summary(run, "rmse_trans_err_mu_sm"),

                # worst
                "worst_trans_err_sm_true": cls._read_from_summary(run, "worst_trans_err_sm_true"),
                "worst_trans_err_mu_true": cls._read_from_summary(run, "worst_trans_err_mu_true"),
                "worst_translation_error": cls._read_from_summary(run, "worst_translation_error"),

                # Rotational errors
                # mean
                "mean_rot_error_raw_odom_deg": cls._deg(
                    cls._read_from_summary(run, "mean_rotation_error_raw_odom")
                ),
                "mean_rot_err_sm_true_deg": cls._deg(
                    cls._read_from_summary(run, "mean_rot_err_sm_true")
                ),
                "mean_rot_err_mu_true_deg": cls._deg(
                    cls._read_from_summary(run, "mean_rot_err_mu_true")
                ),
                "mean_rot_error_deg": cls._deg(
                    cls._read_from_summary(run, "mean_rotation_error")
                ),        
                # TODO: Add mean rot err sm true        
                # "mean_rot_err_mu_sm_deg": cls._deg(
                #     cls._summary(run, "mean_rot_err_mu_sm")
                # ),
                
                # rmse
                "rmse_rot_error_raw_odom_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_rotation_error_raw_odom")
                ),                
                "rmse_rot_err_sm_true_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_rot_err_sm_true")
                ),
                "rmse_rot_err_mu_true_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_rot_err_mu_true")
                ),
                "rmse_rot_error_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_rotation_error")
                ),

                # worst
                "worst_rot_err_sm_true_deg": cls._deg(
                    cls._read_from_summary(run, "worst_rot_err_sm_true")
                ),
                "worst_rot_err_mu_true_deg": cls._deg(
                    cls._read_from_summary(run, "worst_rot_err_mu_true")
                ),
                "worst_rot_error_deg": cls._deg(
                    cls._read_from_summary(run, "worst_rotation_error")
                ),

                # XJ errors
                # trans err
                # mean
                "mean_min_trans_err_xjs": cls._read_from_summary(run, "mean_min_trans_err_xjs"),
                "mean_min_xj_trans_err_true": cls._read_from_summary(run, "mean_min_xj_trans_err_true"),
                "mean_best_xj_trans_err_true": cls._read_from_summary(run, "mean_best_xj_trans_err_true"),
                # rmse
                "rmse_min_trans_err_xjs": cls._read_from_summary(run, "rmse_min_trans_err_xjs"),
                "rmse_min_xj_trans_err_true": cls._read_from_summary(run, "rmse_min_xj_trans_err_true"),
                "rmse_best_xj_trans_err_true": cls._read_from_summary(run, "rmse_best_xj_trans_err_true"),
                # rot err
                # mean
                "mean_min_rot_err_xjs_deg": cls._deg(
                    cls._read_from_summary(run, "mean_min_rot_err_xjs")
                ),
                "mean_min_xj_rot_err_true_deg": cls._deg(
                    cls._read_from_summary(run, "mean_min_xj_rot_err_true")
                ),
                "mean_best_xj_rot_err_true_deg": cls._deg(
                    cls._read_from_summary(run, "mean_best_xj_rot_err_true")
                ),
                # rmse
                "rmse_min_rot_err_xjs_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_min_rot_err_xjs")
                ),
                "rmse_min_xj_rot_err_true_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_min_xj_rot_err_true")
                ),
                "rmse_best_xj_rot_err_true_deg": cls._deg(
                    cls._read_from_summary(run, "rmse_best_xj_rot_err_true")
                ),

                # pose errors
                "mean_pose_err_sm_true": cls._read_from_summary(run, "mean_pose_err_sm_true"),
                "mean_pose_err_mu_true": cls._read_from_summary(run, "mean_pose_err_mu_true"),
                "mean_min_xj_pose_err_true": cls._read_from_summary(run, "mean_min_xj_pose_err_true"),
                "mean_weight_min_xj_err": cls._read_from_summary(run, "mean_weight_min_xj_err"),
                "mean_best_weighted_xj_pose_err_true": cls._read_from_summary(
                    run, "mean_best_weighted_xj_pose_err_true"
                ),
                "mean_weight_best_xj": cls._read_from_summary(run, "mean_weight_best_xj"),

                # Final drift metrics
                # trans           
                "drift_trans_err_raw_odom": cls._read_from_summary(run, "drift_trans_err_raw_odom"),
                "drift_trans_err": cls._read_from_summary(run, "drift_trans_err"),
                # rot
                "drift_rot_err_raw_odom_deg": cls._deg(
                    cls._read_from_summary(run, "drift_rot_err_raw_odom")            
                ),
                "drift_rot_err_deg": cls._deg(
                    cls._read_from_summary(run, "drift_rot_err")            
                ),                                                

                # Improvement metrics
                "mean_min_xj_true_err_improves_over_sm_true": cls._read_from_summary(
                    run, "mean_min_xj_true_err_improves_over_sm_true"
                ),
                "mean_best_xj_true_err_improves_over_sm_true": cls._read_from_summary(
                    run, "mean_best_xj_true_err_improves_over_sm_true"
                ),
                "mean_mu_true_err_improves_over_sm_true": cls._read_from_summary(
                    run, "mean_mu_true_err_improves_over_sm_true"
                ),                
                "mu_true_better_than_sm_true_rate": cls._read_from_summary(
                    run, "mu_true_better_than_sm_true_rate"
                ),

                # Covariance and correlation metrics of proposal                
                "mean_xj_eff": cls._read_from_summary(run, "mean_xj_eff"),
                "mean_xj_eff_motion": cls._read_from_summary(run, "mean_xj_eff_motion"),
                "mean_xj_eff_meas": cls._read_from_summary(run, "mean_xj_eff_meas"),
                
                "mean_weight_best_xj": cls._read_from_summary(run, "mean_weight_best_xj"),
                "mean_weight_ratio_min_best_weight": cls._read_from_summary(
                    run, "mean_weight_ratio_min_best_weight"
                ),
                "median_weight_ratio_min_best_weight": cls._read_from_summary(
                    run, "median_weight_ratio_min_best_weight"
                ),
                "mean_log_motion_range": cls._read_from_summary(run, "mean_log_motion_range"),
                "median_log_motion_range": cls._read_from_summary(run, "median_log_motion_range"),
                "mean_log_meas_range": cls._read_from_summary(run, "mean_log_meas_range"),
                "median_log_meas_range": cls._read_from_summary(run, "median_log_meas_range"),
                "mean_log_weight_range": cls._read_from_summary(run, "mean_log_weight_range"),
                
                # xj and weight analysis metrics
                "mean_min_xj_is_best_xj": cls._read_from_summary(run, "mean_min_xj_is_best_xj"),
                "min_xj_better_sm_pose_rate": cls._read_from_summary(run, "min_xj_better_sm_pose_rate"),
                "best_xj_better_sm_pose_rate": cls._read_from_summary(run, "best_xj_better_sm_pose_rate"),

                "mean_min_xj_true_err_weight_score": cls._read_from_summary(
                    run, "mean_min_xj_true_err_weight_score"
                ),

                # Compute correlations (Does high weight, prob, etc correlate with low xj pose errors to true pose)
                # corr for pose errors
                "mean_corr_xjs_weights": cls._read_from_summary(run, "mean_corr_xjs_weights"),
                "median_corr_xjs_weights": cls._read_from_summary(run, "median_corr_xjs_weights"),
                "mean_corr_xjs_motion": cls._read_from_summary(run, "mean_corr_xjs_motion"),
                "median_corr_xjs_motion": cls._read_from_summary(run, "median_corr_xjs_motion"),
                "mean_corr_xjs_meas": cls._read_from_summary(run, "mean_corr_xjs_meas"),
                "median_corr_xjs_meas": cls._read_from_summary(run, "median_corr_xjs_meas"),
                "mean_corr_weights_motion": cls._read_from_summary(run, "mean_corr_weights_motion"),
                "median_corr_weights_motion": cls._read_from_summary(run, "median_corr_weights_motion"),
                "mean_corr_weights_meas": cls._read_from_summary(run, "mean_corr_weights_meas"),
                "median_corr_weights_meas": cls._read_from_summary(run, "median_corr_weights_meas"),
                # corr for trans and rot errors
                "mean_corr_xj_trans_weights": cls._read_from_summary(run, "mean_corr_xj_trans_weights"),
                "median_corr_xj_trans_weights": cls._read_from_summary(run, "median_corr_xj_trans_weights"),
                "mean_corr_xj_trans_motion": cls._read_from_summary(run, "mean_corr_xj_trans_motion"),
                "median_corr_xj_trans_motion": cls._read_from_summary(run, "median_corr_xj_trans_motion"),
                "mean_corr_xj_trans_meas": cls._read_from_summary(run, "mean_corr_xj_trans_meas"),
                "median_corr_xj_trans_meas": cls._read_from_summary(run, "median_corr_xj_trans_meas"),
                "mean_corr_xj_rot_weights": cls._read_from_summary(run, "mean_corr_xj_rot_weights"),
                "median_corr_xj_rot_weights": cls._read_from_summary(run, "median_corr_xj_rot_weights"),
                "mean_corr_xj_rot_motion": cls._read_from_summary(run, "mean_corr_xj_rot_motion"),
                "median_corr_xj_rot_motion": cls._read_from_summary(run, "median_corr_xj_rot_motion"),
                "mean_corr_xj_rot_meas": cls._read_from_summary(run, "mean_corr_xj_rot_meas"),
                "median_corr_xj_rot_meas": cls._read_from_summary(run, "median_corr_xj_rot_meas"),

                # "mean_best_xj_score": cls._summary(run, "mean_best_xj_score"),
                # "rmse_best_xj_score": cls._summary(run, "rmse_best_xj_score"),
                # "mean_motion_rank_score": cls._summary(run, "mean_motion_rank_score"),
                # "mean_meas_rank_score": cls._summary(run, "mean_meas_rank_score"),

                # Covariance matrix information of proposal distribution
                "mean_prop_std_xy": cls._read_from_summary(run, "mean_prop_std_xy"),
                "mean_prop_std_theta_deg": cls._deg(
                    cls._read_from_summary(run, "mean_prop_std_theta")
                ),
                "mean_prop_corr_xy": cls._read_from_summary(run, "mean_prop_corr_xy"),
                "mean_prop_corr_x_theta": cls._read_from_summary(run, "mean_prop_corr_x_theta"),
                "mean_prop_corr_y_theta": cls._read_from_summary(run, "mean_prop_corr_y_theta"),
                
                # Step information
                "n_steps": cls._read_from_summary(run, "n_steps"),
                "mean_step_duration_ms": cls._optional_ms(
                    cls._read_from_summary(run, "mean_step_duration")
                ),

                # Measurement model metrics
                # Proposal 
                # Counters
                "meas_model_prop_call_count": cls._read_from_summary(
                    run, "meas_model_prop_call_count"
                ),
                "meas_model_prop_valid_beam_count": cls._read_from_summary(
                    run, "meas_model_prop_valid_beam_count"
                ),
                "meas_model_prop_map_hit_count": cls._read_from_summary(
                    run, "meas_model_prop_map_hit_count"
                ),
                "meas_model_prop_no_map_hit_count": cls._read_from_summary(
                    run, "meas_model_prop_no_map_hit_count"
                ),
                "meas_model_prop_out_of_map_count": cls._read_from_summary(
                    run, "meas_model_prop_out_of_map_count"
                ),
                "meas_model_prop_unknown_ray_count": cls._read_from_summary(
                    run, "meas_model_prop_unknown_ray_count"
                ),
                "meas_model_prop_known_free_ray_count": cls._read_from_summary(
                    run, "meas_model_prop_known_free_ray_count"
                ),
                "meas_model_prop_unexpected_known_free_count": cls._read_from_summary(
                    run, "meas_model_prop_unexpected_known_free_count"
                ),
                # Rates
                "meas_model_prop_map_hit_rate": cls._read_from_summary(
                    run, "meas_model_prop_map_hit_rate"
                ),
                "meas_model_prop_out_of_map_rate": cls._read_from_summary(
                    run, "meas_model_prop_out_of_map_rate"
                ),
                "meas_model_prop_no_map_hit_rate": cls._read_from_summary(
                    run, "meas_model_prop_no_map_hit_rate"
                ),
                "meas_model_prop_unknown_no_map_hit_rate": cls._read_from_summary(
                    run, "meas_model_prop_unknown_no_map_hit_rate"
                ),
                "meas_model_prop_known_free_no_map_hit_rate": cls._read_from_summary(
                    run, "meas_model_prop_known_free_no_map_hit_rate"
                ),
                "meas_model_prop_unexpected_known_free_rate": cls._read_from_summary(
                    run, "meas_model_prop_unexpected_known_free_rate"
                ),
                # Fallback
                # Counters
                "meas_model_fallback_call_count": cls._read_from_summary(
                    run, "meas_model_fallback_call_count"
                ),
                "meas_model_fallback_valid_beam_count": cls._read_from_summary(
                    run, "meas_model_fallback_valid_beam_count"
                ),
                "meas_model_fallback_map_hit_count": cls._read_from_summary(
                    run, "meas_model_fallback_map_hit_count"
                ),
                "meas_model_fallback_no_map_hit_count": cls._read_from_summary(
                    run, "meas_model_fallback_no_map_hit_count"
                ),
                "meas_model_fallback_out_of_map_count": cls._read_from_summary(
                    run, "meas_model_fallback_out_of_map_count"
                ),
                "meas_model_fallback_unknown_ray_count": cls._read_from_summary(
                    run, "meas_model_fallback_unknown_ray_count"
                ),
                "meas_model_fallback_known_free_ray_count": cls._read_from_summary(
                    run, "meas_model_fallback_known_free_ray_count"
                ),
                "meas_model_fallback_unexpected_known_free_count": cls._read_from_summary(
                    run, "meas_model_fallback_unexpected_known_free_count"
                ),
                # Rates
                "meas_model_fallback_map_hit_rate": cls._read_from_summary(
                    run, "meas_model_fallback_map_hit_rate"
                ),
                "meas_model_fallback_out_of_map_rate": cls._read_from_summary(
                    run, "meas_model_fallback_out_of_map_rate"
                ),
                "meas_model_fallback_no_map_hit_rate": cls._read_from_summary(
                    run, "meas_model_fallback_no_map_hit_rate"
                ),
                "meas_model_fallback_unknown_no_map_hit_rate": cls._read_from_summary(
                    run, "meas_model_fallback_unknown_no_map_hit_rate"
                ),
                "meas_model_fallback_known_free_no_map_hit_rate": cls._read_from_summary(
                    run, "meas_model_fallback_known_free_no_map_hit_rate"
                ),
                "meas_model_fallback_unexpected_known_free_rate": cls._read_from_summary(
                    run, "meas_model_fallback_unexpected_known_free_rate"
                ),
                
            }

            rows.append(row)

        return pd.DataFrame(rows)


class ResultAggregator:
    """
    Class to aggregate summary results of the optimization pipeline.
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
        """
        Place the given column after the column named by col_after.
        """
        extract_col = df.pop(col)
        df.insert(df.columns.get_loc(col_after) + 1, col, extract_col)
        return df


    def rank_by_score(self, ranked_run_df: pd.DataFrame, score_col: str, ascending=True) -> pd.DataFrame:
        '''
        Rank the given DataFrame by the specified score column. The ranking is ascending if
        ascending is True and descending otherwise. The rank is added as a new column named
        "rank" at the beginning of the DataFrame.
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
        First aggregation stage that groups the given DataFrame by dataset_id, parameter_hash,
        and used_meas_model, then aggregates the run metrics. It also computes derived failure
        and measurement-model rates plus a dataset_param_score. The resulting DataFrame is ranked
        by dataset_param_score in ascending order (lower is better).

        Parameters
        ----------
        ranked_run_df : pd.DataFrame
            DataFrame containing the ranked runs with the required columns.

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame grouped by dataset_id, parameter_hash, and used_meas_model,
            with additional metrics, scores, and ranking.
        '''
        # Define the required columns for aggregation
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
            "measurement_stddev",

            # Translational pose errors
            "mean_trans_error",
            "mean_trans_err_sm_true",
            "mean_trans_err_mu_true",

            "rmse_trans_err_sm_true",
            "rmse_trans_err_mu_true",
            "rmse_trans_error",

            "worst_trans_err_sm_true",
            "worst_trans_err_mu_true",
            "worst_translation_error",

            # Rotational pose errors
            "mean_rot_err_sm_true_deg",
            "mean_rot_err_mu_true_deg",
            "mean_rot_error_deg",

            "rmse_rot_err_sm_true_deg",
            "rmse_rot_err_mu_true_deg",
            "rmse_rot_error_deg",

            "worst_rot_err_sm_true_deg",
            "worst_rot_err_mu_true_deg",
            "worst_rot_error_deg",

            # Scan-matcher information
            "scan_match_failed_count",
            "scan_match_fallback_failed_count",
            "median_extracted_map_points",
            "median_map_point_keep_ratio",

            # Proposal metrics
            "mu_true_better_than_sm_true_rate",

            "mean_min_xj_is_best_xj",
            "mean_min_xj_pose_err_true",
            "mean_best_weighted_xj_pose_err_true",

            "mean_min_xj_true_err_improves_over_sm_true",
            "min_xj_better_sm_pose_rate",

            "mean_best_xj_true_err_improves_over_sm_true",
            "best_xj_better_sm_pose_rate",

            "median_log_motion_range",
            "median_log_meas_range",

            # XJ error metrics
            "mean_min_trans_err_xjs",
            "mean_min_xj_trans_err_true",
            "mean_best_xj_trans_err_true",
            "rmse_min_trans_err_xjs",
            "rmse_min_xj_trans_err_true",
            "rmse_best_xj_trans_err_true",
            "mean_min_rot_err_xjs_deg",
            "mean_min_xj_rot_err_true_deg",
            "mean_best_xj_rot_err_true_deg",
            "rmse_min_rot_err_xjs_deg",
            "rmse_min_xj_rot_err_true_deg",
            "rmse_best_xj_rot_err_true_deg",
            
            # Correlations
            # corr for pose errors
            "mean_corr_xjs_motion",
            "mean_corr_xjs_meas",
            # corr for trans errors
            "mean_corr_xj_trans_weights",
            "mean_corr_xj_trans_motion",
            "mean_corr_xj_trans_meas",
            # corr for rot errors
            "mean_corr_xj_rot_weights",
            "mean_corr_xj_rot_motion",
            "mean_corr_xj_rot_meas",

            # Measurement model metrics
            "meas_model_prop_call_count",
            "meas_model_prop_valid_beam_count",
            "meas_model_prop_map_hit_count",
            "meas_model_prop_no_map_hit_count",
            "meas_model_prop_out_of_map_count",
            "meas_model_prop_unknown_ray_count",
            "meas_model_prop_known_free_ray_count",
            "meas_model_prop_unexpected_known_free_count",
            # "meas_model_prop_map_hit_rate",
            # "meas_model_prop_out_of_map_rate",
            # "meas_model_prop_no_map_hit_rate",
            # "meas_model_prop_unknown_no_map_hit_rate",
            # "meas_model_prop_known_free_no_map_hit_rate",
            # "meas_model_prop_unexpected_known_free_rate",
            "meas_model_fallback_call_count",
            "meas_model_fallback_valid_beam_count",
            "meas_model_fallback_map_hit_count",
            "meas_model_fallback_no_map_hit_count",
            "meas_model_fallback_out_of_map_count",
            "meas_model_fallback_unknown_ray_count",
            "meas_model_fallback_known_free_ray_count",
            "meas_model_fallback_unexpected_known_free_count",
            # "meas_model_fallback_map_hit_rate",
            # "meas_model_fallback_out_of_map_rate",
            # "meas_model_fallback_no_map_hit_rate",
            # "meas_model_fallback_unknown_no_map_hit_rate",
            # "meas_model_fallback_known_free_no_map_hit_rate",
            # "meas_model_fallback_unexpected_known_free_rate"
        ]

        # Check if the given DataFrame includes the required columns, raising an error if not
        missing = [col for col in required_cols if col not in ranked_run_df.columns]
        if missing:
            raise ValueError(
                "aggregate_by_dataset_and_param missing required columns: " + ", ".join(missing)
            )

        # Group the DataFrame by dataset, parameter hash, and measurement model, then aggregate the metrics
        agg_dataset_param_df = self._groupby(
            ranked_run_df,
            ["dataset_id", "parameter_hash", "used_meas_model"],
        )
        agg_dataset_param_df: pd.DataFrame = agg_dataset_param_df.agg(
            # general information
            map=("map", "first"),
            parameter_tag=("parameter_tag", "first"),
            # used_meas_model=("used_meas_model", "first"),
            n_runs=("score", "size"),
            n_seeds=("seed", "nunique"),

            total_n_steps=("n_steps", "sum"),
            measurement_stddev=("measurement_stddev", "first"),

            # Metrics for score computation
            mean_score=("score", "mean"),
            worst_score=("score", "max"),
            std_score=("score", "std"),

            # Pose errors
            # Translational err
            # mean
            mean_trans_err=("mean_trans_error", "mean"),
            mean_trans_err_sm_true=("mean_trans_err_sm_true", "mean"),
            mean_trans_err_mu_true=("mean_trans_err_mu_true", "mean"),
            # rmse
            mean_rmse_trans_err_sm_true=("rmse_trans_err_sm_true", "mean"),
            mean_rmse_trans_err_mu_true=("rmse_trans_err_mu_true", "mean"),
            mean_rmse_trans_error=("rmse_trans_error", "mean"),
            # worst rmse
            worst_rmse_trans_err_sm_true=("rmse_trans_err_sm_true", "max"),
            worst_rmse_trans_err_mu_true=("rmse_trans_err_mu_true", "max"),
            worst_rmse_trans_error=("rmse_trans_error", "max"),
            # max values (These are the max vals over all steps for all runs)
            max_trans_err_sm_true=("worst_trans_err_sm_true", "max"),
            max_trans_err_mu_true=("worst_trans_err_mu_true", "max"),
            max_trans_err=("worst_translation_error", "max"),

            # Rotational errors
            # mean
            mean_rot_err_sm_true_deg=("mean_rot_err_sm_true_deg", "mean"),
            mean_rot_err_mu_true_deg=("mean_rot_err_mu_true_deg", "mean"),
            mean_rot_error_deg=("mean_rot_error_deg", "mean"),
            # rmse
            mean_rmse_rot_err_sm_true_deg=("rmse_rot_err_sm_true_deg", "mean"),
            mean_rmse_rot_err_mu_true_deg=("rmse_rot_err_mu_true_deg", "mean"),
            mean_rmse_rot_error_deg=("rmse_rot_error_deg", "mean"),
            # worst
            worst_rmse_rot_err_sm_true_deg=("rmse_rot_err_sm_true_deg", "max"),
            worst_rmse_rot_err_mu_true_deg=("rmse_rot_err_mu_true_deg", "max"),
            worst_rmse_rot_error_deg=("rmse_rot_error_deg", "max"),
            # max values
            max_rot_err_sm_true_deg=("worst_rot_err_sm_true_deg", "max"),
            max_rot_err_mu_true_deg=("worst_rot_err_mu_true_deg", "max"),
            max_rot_err_deg=("worst_rot_error_deg", "max"),

            # Scan matcher info            
            total_scan_match_failed_count=("scan_match_failed_count", "sum"),
            total_scan_match_fallback_failed_count=("scan_match_fallback_failed_count", "sum"),
            
            median_extracted_map_points=("median_extracted_map_points", "median"),
            median_map_point_keep_ratio=("median_map_point_keep_ratio", "median"),                          

            # Proposal information metrics
            mean_mu_true_better_than_sm_true_rate=("mu_true_better_than_sm_true_rate", "mean"),
            worst_mu_true_better_than_sm_true_rate=("mu_true_better_than_sm_true_rate", "min"),

            mean_min_xj_is_best_xj=("mean_min_xj_is_best_xj", "mean"),
            worst_min_xj_is_best_xj=("mean_min_xj_is_best_xj", "min"),

            mean_min_xj_pose_err_true=("mean_min_xj_pose_err_true", "mean"),
            worst_min_xj_pose_err_true=("mean_min_xj_pose_err_true", "max"),

            mean_best_weighted_xj_pose_err_true=("mean_best_weighted_xj_pose_err_true", "mean"),
            worst_best_weighted_xj_pose_err_true=("mean_best_weighted_xj_pose_err_true", "max"),

            mean_min_xj_true_err_improves_over_sm_true =("mean_min_xj_true_err_improves_over_sm_true", "mean"),
            worst_min_xj_true_err_improves_over_sm_true =("mean_min_xj_true_err_improves_over_sm_true", "min"),

            mean_min_xj_better_sm_pose_rate =("min_xj_better_sm_pose_rate", "mean"),
            worst_min_xj_better_sm_pose_rate =("min_xj_better_sm_pose_rate", "min"),
            
            mean_best_xj_true_err_improves_over_sm_true =("mean_best_xj_true_err_improves_over_sm_true", "mean"),
            worst_best_xj_true_err_improves_over_sm_true =("mean_best_xj_true_err_improves_over_sm_true", "min"),

            mean_best_xj_better_sm_pose_rate =("best_xj_better_sm_pose_rate", "mean"),
            worst_best_xj_better_sm_pose_rate =("best_xj_better_sm_pose_rate", "min"),

            median_log_motion_range=("median_log_motion_range", "median"),
            median_log_meas_range=("median_log_meas_range", "median"),

            # XJ error metrics
            mean_min_trans_err_xjs=("mean_min_trans_err_xjs", "mean"),
            mean_min_xj_trans_err_true=("mean_min_xj_trans_err_true", "mean"),
            mean_best_xj_trans_err_true=("mean_best_xj_trans_err_true", "mean"),
            mean_rmse_min_trans_err_xjs=("rmse_min_trans_err_xjs", "mean"),
            mean_rmse_min_xj_trans_err_true=("rmse_min_xj_trans_err_true", "mean"),
            mean_rmse_best_xj_trans_err_true=("rmse_best_xj_trans_err_true", "mean"),
            mean_min_rot_err_xjs_deg=("mean_min_rot_err_xjs_deg", "mean"),
            mean_min_xj_rot_err_true_deg=("mean_min_xj_rot_err_true_deg", "mean"),
            mean_best_xj_rot_err_true_deg=("mean_best_xj_rot_err_true_deg", "mean"),
            mean_rmse_min_rot_err_xjs_deg=("rmse_min_rot_err_xjs_deg", "mean"),
            mean_rmse_min_xj_rot_err_true_deg=("rmse_min_xj_rot_err_true_deg", "mean"),
            mean_rmse_best_xj_rot_err_true_deg=("rmse_best_xj_rot_err_true_deg", "mean"),

            # Correlations
            # corr for pose errors
            mean_corr_xjs_motion=("mean_corr_xjs_motion", "mean"),            
            mean_corr_xjs_meas=("mean_corr_xjs_meas", "mean"),
            worst_corr_xjs_motion=("mean_corr_xjs_motion", "min"),
            worst_corr_xjs_meas=("mean_corr_xjs_meas", "min"),
            # corr for trans errors
            mean_corr_xj_trans_motion=("mean_corr_xj_trans_motion", "mean"),            
            mean_corr_xj_trans_meas=("mean_corr_xj_trans_meas", "mean"),
            worst_corr_xj_trans_meas=("mean_corr_xj_trans_meas", "min"),
            worst_corr_xj_trans_motion=("mean_corr_xj_trans_motion", "min"),
            # corr for rot errors
            mean_corr_xj_rot_motion=("mean_corr_xj_rot_motion", "mean"),
            mean_corr_xj_rot_meas=("mean_corr_xj_rot_meas", "mean"),
            worst_corr_xj_rot_meas=("mean_corr_xj_rot_meas", "min"),
            worst_corr_xj_rot_motion=("mean_corr_xj_rot_motion", "min"),

            # Measurement model metrics
            # Proposal
            meas_model_prop_call_count=("meas_model_prop_call_count", "sum"),
            meas_model_prop_valid_beam_count=("meas_model_prop_valid_beam_count", "sum"),
            meas_model_prop_map_hit_count=("meas_model_prop_map_hit_count", "sum"),
            meas_model_prop_no_map_hit_count=("meas_model_prop_no_map_hit_count", "sum"),
            meas_model_prop_out_of_map_count=("meas_model_prop_out_of_map_count", "sum"),
            meas_model_prop_unknown_ray_count=("meas_model_prop_unknown_ray_count", "sum"),
            meas_model_prop_known_free_ray_count=("meas_model_prop_known_free_ray_count", "sum"),
            meas_model_prop_unexpected_known_free_count=("meas_model_prop_unexpected_known_free_count", "sum"),
            # Fallback
            meas_model_fallback_call_count=("meas_model_fallback_call_count", "sum"),
            meas_model_fallback_valid_beam_count=("meas_model_fallback_valid_beam_count", "sum"),
            meas_model_fallback_map_hit_count=("meas_model_fallback_map_hit_count", "sum"),
            meas_model_fallback_no_map_hit_count=("meas_model_fallback_no_map_hit_count", "sum"),
            meas_model_fallback_out_of_map_count=("meas_model_fallback_out_of_map_count", "sum"),
            meas_model_fallback_unknown_ray_count=("meas_model_fallback_unknown_ray_count", "sum"),
            meas_model_fallback_known_free_ray_count=("meas_model_fallback_known_free_ray_count", "sum"),
            meas_model_fallback_unexpected_known_free_count=("meas_model_fallback_unexpected_known_free_count", "sum"),

        ).reset_index()

        # Place map name directly after dataset id
        agg_dataset_param_df = self._place_col_after_col(
            df=agg_dataset_param_df,
            col="map",
            col_after="dataset_id"
        )

        # Compute scan-matching failure rates
        # Here we don't compute mean from rates before because here we can safely compute this by the n_steps. This
        # is because we aggregate per map. This is the correct computation
        n_steps = agg_dataset_param_df["total_n_steps"]
        agg_dataset_param_df["scan_match_failed_rate"] = (
            agg_dataset_param_df["total_scan_match_failed_count"] / n_steps
        )

        agg_dataset_param_df["scan_match_fallback_failed_rate"] = (
            agg_dataset_param_df["total_scan_match_fallback_failed_count"] / n_steps
        )

        # Place scan-matching failure rates directly after their corresponding counts
        agg_dataset_param_df = self._place_col_after_col(
            df=agg_dataset_param_df,
            col="scan_match_failed_rate",
            col_after="total_scan_match_failed_count"
        )
        agg_dataset_param_df = self._place_col_after_col(
            df=agg_dataset_param_df,
            col="scan_match_fallback_failed_rate",
            col_after="total_scan_match_fallback_failed_count"
        )

        # Compute measurement model metrics
        def _rate(numerator_col: str, denominator_col: str, call_count_col: str):
            numerator = pd.to_numeric(agg_dataset_param_df[numerator_col], errors="coerce")
            denominator = pd.to_numeric(agg_dataset_param_df[denominator_col], errors="coerce")
            call_count = pd.to_numeric(agg_dataset_param_df[call_count_col], errors="coerce")
            valid = call_count.gt(0) & denominator.gt(0) & numerator.ge(0)
            result = pd.Series(np.nan, index=agg_dataset_param_df.index, dtype=float)
            result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
            return result

        # Proposal rates
        agg_dataset_param_df["meas_model_prop_map_hit_rate"] = _rate(
            "meas_model_prop_map_hit_count", "meas_model_prop_valid_beam_count", "meas_model_prop_call_count"
        )
        agg_dataset_param_df["meas_model_prop_out_of_map_rate"] = _rate(
            "meas_model_prop_out_of_map_count", "meas_model_prop_valid_beam_count", "meas_model_prop_call_count"
        )
        agg_dataset_param_df["meas_model_prop_no_map_hit_rate"] = _rate(
            "meas_model_prop_no_map_hit_count", "meas_model_prop_valid_beam_count", "meas_model_prop_call_count"
        )
        agg_dataset_param_df["meas_model_prop_unknown_no_map_hit_rate"] = _rate(
            "meas_model_prop_unknown_ray_count", "meas_model_prop_no_map_hit_count", "meas_model_prop_call_count"
        )
        agg_dataset_param_df["meas_model_prop_known_free_no_map_hit_rate"] = _rate(
            "meas_model_prop_known_free_ray_count", "meas_model_prop_no_map_hit_count", "meas_model_prop_call_count"
        )
        agg_dataset_param_df["meas_model_prop_unexpected_known_free_rate"] = _rate(
            "meas_model_prop_unexpected_known_free_count", "meas_model_prop_known_free_ray_count", "meas_model_prop_call_count"
        )

        # Fallback rates
        agg_dataset_param_df["meas_model_fallback_map_hit_rate"] = _rate(
            "meas_model_fallback_map_hit_count", "meas_model_fallback_valid_beam_count", "meas_model_fallback_call_count"
        )
        agg_dataset_param_df["meas_model_fallback_out_of_map_rate"] = _rate(
            "meas_model_fallback_out_of_map_count", "meas_model_fallback_valid_beam_count", "meas_model_fallback_call_count"
        )
        agg_dataset_param_df["meas_model_fallback_no_map_hit_rate"] = _rate(
            "meas_model_fallback_no_map_hit_count", "meas_model_fallback_valid_beam_count", "meas_model_fallback_call_count"
        )
        agg_dataset_param_df["meas_model_fallback_unknown_no_map_hit_rate"] = _rate(
            "meas_model_fallback_unknown_ray_count", "meas_model_fallback_no_map_hit_count", "meas_model_fallback_call_count"
        )
        agg_dataset_param_df["meas_model_fallback_known_free_no_map_hit_rate"] = _rate(
            "meas_model_fallback_known_free_ray_count", "meas_model_fallback_no_map_hit_count", "meas_model_fallback_call_count"
        )
        agg_dataset_param_df["meas_model_fallback_unexpected_known_free_rate"] = _rate(
            "meas_model_fallback_unexpected_known_free_count", "meas_model_fallback_known_free_ray_count", "meas_model_fallback_call_count"
        )

        # Fill NaN values in std_score with 0.0 to avoid issues in score computation
        agg_dataset_param_df["std_score"] = agg_dataset_param_df["std_score"].fillna(0.0)

        # Compute and insert the dataset-parameter score
        dataset_param_score = (
            1.0 * agg_dataset_param_df["mean_score"]
            + 0.5 * agg_dataset_param_df["worst_score"]
            + 0.2 * agg_dataset_param_df["std_score"]
        )

        agg_dataset_param_df.insert(0, "dataset_param_score", dataset_param_score)

        return self.rank_by_score(agg_dataset_param_df, "dataset_param_score", ascending=True)


    def aggregate_by_params(self, agg_dataset_param_df: pd.DataFrame):
        '''
        Second aggregation stage that groups the dataset-level results by parameter_hash, then
        aggregates the metrics across datasets. It also computes additional statistics and a
        global_score for each parameter hash. The resulting DataFrame is ranked by global_score
        in ascending order (lower is better).

        Parameters
        ----------
        agg_dataset_param_df : pd.DataFrame
            DataFrame containing the results from the first aggregation stage, grouped by dataset,
            parameter hash, and measurement model.

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame grouped by parameter_hash, with additional metrics, global scores,
            and ranking.
        '''
        # Define the required columns for aggregation
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

        # Check if the given DataFrame includes the required columns, raising an error if not
        missing = [col for col in required_cols if col not in agg_dataset_param_df.columns]
        if missing:
            raise ValueError(
                "aggregate_by_params missing required columns: " + ", ".join(missing)
            )
        
        # Group the DataFrame by parameter_hash, then aggregate the metrics across datasets
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
        
        # Fill NaN values in std_score with 0.0 to avoid issues in score computation
        agg_param_df["std_score"] = agg_param_df["std_score"].fillna(0.0)
        
        # Compute and insert the global score
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
