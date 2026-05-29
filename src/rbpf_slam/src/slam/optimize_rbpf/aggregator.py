import math
from typing import Iterable, List

import pandas as pd

from .optimizer import RankedRun


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
    def _summary(run: RankedRun, key: str, default=None):
        """Safely read a value from run.summary."""
        return run.summary.get(key, default)

    @classmethod
    def to_dataframe(cls, ranked_runs: Iterable[RankedRun]) -> pd.DataFrame:
        """
        Convert ranked runs to a pandas DataFrame.
        """
        rows = []

        for run in ranked_runs:
            params = run.params

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
                "sigma_measurement": params.measurement_model_params.sigma_measurement,
                "proposal_alpha": params.proposal_alpha,
                "proposal_beta": params.proposal_beta,
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
                "scan_match_failed_count": cls._summary(run, "scan_match_failed_count"),
                "scan_match_fallback_failed_count": cls._summary(run, "scan_match_fallback_failed_count"),
                "count_too_few_points": cls._summary(run, "count_too_few_points", 0),
                "count_too_few_corresp": cls._summary(run, "count_too_few_corresp", 0),
                "infinite_h_or_g": cls._summary(run, "infinite_h_or_g", 0),
                "ill_cond_H": cls._summary(run, "ill_cond_H", 0),
                "infinite_dtransform": cls._summary(run, "infinite_dtransform", 0),
                "infinite_mean_err": cls._summary(run, "infinite_mean_err", 0),
                "best_transf_too_large": cls._summary(run, "best_transf_too_large", 0),
                "best_mean_err_too_large": cls._summary(run, "best_mean_err_too_large", 0),

                # End pose errors
                "mean_trans_error_raw_odom": cls._summary(run, "mean_translation_error_raw_odom"),
                "mean_rot_error_raw_odom_deg": cls._deg(
                    cls._summary(run, "mean_rotation_error_raw_odom")
                ),
                "rmse_trans_error_raw_odom": cls._summary(run, "rmse_translation_error_raw_odom"),
                "rmse_rot_error_raw_odom_deg": cls._deg(
                    cls._summary(run, "rmse_rotation_error_raw_odom")
                ),
                "mean_trans_error": cls._summary(run, "mean_translation_error"),
                "mean_rot_error_deg": cls._deg(
                    cls._summary(run, "mean_rotation_error")
                ),
                "rmse_trans_error": cls._summary(run, "rmse_translation_error"),
                "rmse_rot_error_deg": cls._deg(
                    cls._summary(run, "rmse_rotation_error")
                ),
                "drift_trans_err_raw_odom": cls._summary(run, "drift_trans_err_raw_odom"),

                # Intentionally mirrored from previous CSV writer behavior:
                # these are not converted here.
                "drift_rot_err_raw_odom_deg": cls._summary(run, "drift_rot_err_raw_odom"),
                "drift_trans_err": cls._summary(run, "drift_trans_err"),
                "drift_rot_err_deg": cls._summary(run, "drift_rot_err"),

                # Mu errors of proposal
                "mean_trans_err_mu_true": cls._summary(run, "mean_trans_err_mu_true"),
                "mean_rot_err_mu_true_deg": cls._deg(
                    cls._summary(run, "mean_rot_err_mu_true")
                ),
                "mean_trans_err_mu_sm": cls._summary(run, "mean_trans_err_mu_sm"),
                "mean_rot_err_mu_sm_deg": cls._deg(
                    cls._summary(run, "mean_rot_err_mu_sm")
                ),
                "rmse_trans_err_mu_sm": cls._summary(run, "rmse_trans_err_mu_sm"),
                "rmse_rot_err_mu_sm_deg": cls._deg(
                    cls._summary(run, "rmse_rot_err_mu_sm")
                ),

                # Covariance and correlation metrics of proposal
                "mean_prop_std_xy": cls._summary(run, "mean_prop_std_xy"),
                "mean_prop_std_theta_deg": cls._deg(
                    cls._summary(run, "mean_prop_std_theta")
                ),
                "mean_prop_corr_xy": cls._summary(run, "mean_prop_corr_xy"),
                "mean_prop_corr_x_theta": cls._summary(run, "mean_prop_corr_x_theta"),
                "mean_prop_corr_y_theta": cls._summary(run, "mean_prop_corr_y_theta"),
                "mean_xj_eff": cls._summary(run, "mean_xj_eff"),
                "mean_xj_eff_motion": cls._summary(run, "mean_xj_eff_motion"),
                "mean_xj_eff_meas": cls._summary(run, "mean_xj_eff_meas"),

                # Pose errors metrics and xj weights
                "mean_pose_err_sm_true": cls._summary(run, "mean_pose_err_sm_true"),
                "mean_pose_err_mu_true": cls._summary(run, "mean_pose_err_mu_true"),
                "mean_min_xj_pose_err_true": cls._summary(run, "mean_min_xj_pose_err_true"),
                "mean_weight_min_xj_err": cls._summary(run, "mean_weight_min_xj_err"),
                "mean_best_weighted_xj_pose_err_true": cls._summary(
                    run, "mean_best_weighted_xj_pose_err_true"
                ),
                "mean_weight_best_xj": cls._summary(run, "mean_weight_best_xj"),
                "mean_weight_ratio_min_best_weight": cls._summary(
                    run, "mean_weight_ratio_min_best_weight"
                ),
                "median_weight_ratio_min_best_weight": cls._summary(
                    run, "median_weight_ratio_min_best_weight"
                ),
                "mean_log_motion_range": cls._summary(run, "mean_log_motion_range"),
                "median_log_motion_range": cls._summary(run, "median_log_motion_range"),
                "mean_log_meas_range": cls._summary(run, "mean_log_meas_range"),
                "median_log_meas_range": cls._summary(run, "median_log_meas_range"),
                "mean_log_weight_range": cls._summary(run, "mean_log_weight_range"),

                # Improvement metrics
                "mean_min_xj_true_err_improves_over_sm_true": cls._summary(
                    run, "mean_min_xj_true_err_improves_over_sm_true"
                ),
                "rmse_min_xj_true_err_improves_over_sm_true": cls._summary(
                    run, "rmse_min_xj_true_err_improves_over_sm_true"
                ),
                "mean_best_xj_true_err_improves_over_sm_true": cls._summary(
                    run, "mean_best_xj_true_err_improves_over_sm_true"
                ),
                "rmse_best_xj_true_err_improves_over_sm_true": cls._summary(
                    run, "rmse_best_xj_true_err_improves_over_sm_true"
                ),
                "mean_mu_true_err_improves_over_sm_true": cls._summary(
                    run, "mean_mu_true_err_improves_over_sm_true"
                ),
                "rmse_mu_true_err_improves_over_sm_true": cls._summary(
                    run, "rmse_mu_true_err_improves_over_sm_true"
                ),

                # xj and weight analysis metrics
                "mean_min_xj_is_best_xj": cls._summary(run, "mean_min_xj_is_best_xj"),
                "mean_min_xj_true_err_weight_score": cls._summary(
                    run, "mean_min_xj_true_err_weight_score"
                ),
                "rmse_min_xj_true_err_weight_score": cls._summary(
                    run, "rmse_min_xj_true_err_weight_score"
                ),
                "mean_corr_xjs_weights": cls._summary(run, "mean_corr_xjs_weights"),
                "median_corr_xjs_weights": cls._summary(run, "median_corr_xjs_weights"),
                "mean_corr_xjs_motion": cls._summary(run, "mean_corr_xjs_motion"),
                "median_corr_xjs_motion": cls._summary(run, "median_corr_xjs_motion"),
                "mean_corr_xjs_meas": cls._summary(run, "mean_corr_xjs_meas"),
                "median_corr_xjs_meas": cls._summary(run, "median_corr_xjs_meas"),
                "mean_corr_weights_motion": cls._summary(run, "mean_corr_weights_motion"),
                "median_corr_weights_motion": cls._summary(run, "median_corr_weights_motion"),
                "mean_corr_weights_meas": cls._summary(run, "mean_corr_weights_meas"),
                "median_corr_weights_meas": cls._summary(run, "median_corr_weights_meas"),
                "mean_best_xj_score": cls._summary(run, "mean_best_xj_score"),
                "rmse_best_xj_score": cls._summary(run, "rmse_best_xj_score"),
                "mean_motion_rank_score": cls._summary(run, "mean_motion_rank_score"),
                "mean_meas_rank_score": cls._summary(run, "mean_meas_rank_score"),

                # Step information
                "n_steps": cls._summary(run, "n_steps"),
                "mean_step_duration_ms": cls._optional_ms(
                    cls._summary(run, "mean_step_duration")
                ),
            }

            rows.append(row)

        return pd.DataFrame(rows)


class ResultAggregator:
    """
    Aggregate the given dataframe run results into the desried output types.
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

    def rank_by_score(self, ranked_run_df: pd.DataFrame, score_col, ascending=True) -> pd.DataFrame:
        '''
        Sorts the given df by score in the desired order (asc or desc).
        '''   
        # Sort df by score values 
        ranked_df: pd.DataFrame = ranked_run_df.sort_values(by=score_col, ascending=ascending).reset_index(drop=True)

        # Check if rank col exists, if so delete
        if "rank" in ranked_run_df.columns:
            ranked_df = ranked_df.drop(columns=["rank"])
        
        # Add rank col
        ranked_df["rank"] = ranked_df.index + 1
        
        return ranked_df


    def aggregate_by_data_and_seed(self, ranked_run_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Aggregate runs by map and seed and compute a weighted group score.
        '''
        # Estimate if all needed columns exist
        required_cols = {
                "dataset_id",
                "seed",
                "score",
                "rmse_trans_error",
                "rmse_rot_error_deg",
                "mean_min_xj_is_best_xj",
                "parameter_tag",
                "parameter_hash",
                "used_meas_model",
            }

        missing = sorted(col for col in required_cols if col not in ranked_run_df.columns)
        if missing:
            raise ValueError(
                "aggregate_by_dataset_id_and_seed missing required columns: " + ", ".join(missing)
            )

        # Compute metrics from columns
        agg_dataset_seed_df = self._groupby(
            ranked_run_df,
            ["dataset_id", "parameter_hash"],
        )
        agg_dataset_seed_df: pd.DataFrame = agg_dataset_seed_df.agg(
            parameter_tag=("parameter_tag", "first"),
            used_meas_model=("used_meas_model", "first"),
            n_runs=("score", "size"),
            n_seeds=("seed", "nunique"),

            mean_score=("score", "mean"),
            worst_score=("score", "max"),
            std_score=("score", "std"),

            mean_rmse_trans_error=("rmse_trans_error", "mean"),
            worst_rmse_trans_error=("rmse_trans_error", "max"),

            mean_rmse_rot_error_deg=("rmse_rot_error_deg", "mean"),
            worst_rmse_rot_error_deg=("rmse_rot_error_deg", "max"),

            mean_min_xj_is_best_xj=("mean_min_xj_is_best_xj", "mean"),
            worst_min_xj_is_best_xj=("mean_min_xj_is_best_xj", "min"),
        ).reset_index()
        
        # Use 0.0 score if score is none
        agg_dataset_seed_df["std_score"] = agg_dataset_seed_df["std_score"].fillna(0.0)

        # Compute score 
        agg_dataset_seed_df["dataset_param_score"] = (
            1.0 * agg_dataset_seed_df["mean_score"]
            + 0.5 * agg_dataset_seed_df["worst_score"]
            + 0.2 * agg_dataset_seed_df["std_score"]
        )

        return self.rank_by_score(agg_dataset_seed_df, "dataset_param_score", ascending=True)


    def aggregate_by_params(self, agg_dataset_seed_df: pd.DataFrame):
        # Estimate if all needed columns exist
        required_cols = {
            "dataset_param_score",
            "parameter_hash",
            "parameter_tag",
        }

        missing = sorted(col for col in required_cols if col not in agg_dataset_seed_df.columns)
        if missing:
            raise ValueError(
                "aggregate_by_params missing required columns: " + ", ".join(missing)
            )
        
        # Compute metrics from columns
        agg_param_df = self._groupby(agg_dataset_seed_df, ["parameter_hash"])
        agg_param_df: pd.DataFrame = agg_param_df.agg(
            parameter_tag=("parameter_tag", "first"),
            used_meas_model=("used_meas_model", "first"),
            n_datasets=("dataset_id", "nunique"),
            n_results=("dataset_param_score", "size"),

            mean_score=("dataset_param_score", "mean"),
            worst_score=("dataset_param_score", "max"),
            std_score=("dataset_param_score", "std"),

            mean_rmse_trans_error=("mean_rmse_trans_error", "mean"),
            worst_rmse_trans_error=("worst_rmse_trans_error", "max"),

            mean_rmse_rot_error_deg=("mean_rmse_rot_error_deg", "mean"),
            worst_rmse_rot_error_deg=("worst_rmse_rot_error_deg", "max"),

            mean_min_xj_is_best_xj=("mean_min_xj_is_best_xj", "mean"),
            worst_min_xj_is_best_xj=("worst_min_xj_is_best_xj", "min"),
        ).reset_index()
        
        agg_param_df["std_score"] = agg_param_df["std_score"].fillna(0.0)
        
        # Compute score
        agg_param_df["global_score"] = (
            1.0 * agg_param_df["mean_score"]
            + 0.5 * agg_param_df["worst_score"]
            + 0.2 * agg_param_df["std_score"]
        )

        # Sort df by score
        agg_param_df = agg_param_df.sort_values(by="global_score", ascending=True).reset_index(drop=True)
        agg_param_df["rank"] = agg_param_df.index + 1

        return self.rank_by_score(agg_param_df, "global_score", ascending=True)
