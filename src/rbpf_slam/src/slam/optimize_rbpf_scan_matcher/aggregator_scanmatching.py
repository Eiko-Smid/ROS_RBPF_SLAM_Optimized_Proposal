import math
from typing import Iterable

import pandas as pd

from .optimizer_scanmatching import RankedRunScanMatching, ScanMatchingOptimizer


class RankedRunConverterScanMatching:
    """Convert scan-matching ranked runs into pandas DataFrames."""

    @staticmethod
    def _deg(value):
        return math.degrees(value) if value is not None else None

    @staticmethod
    def _ms(value):
        return value * 1000.0 if value is not None else None

    @classmethod
    def to_dataframe(cls, ranked_runs: Iterable[RankedRunScanMatching]) -> pd.DataFrame:
        rows = []

        legacy_columns = [
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

            # Rotational errors of raw odom, pred and corrected pose
            "mean_raw_odom_rot_err_deg",
            "mean_pred_rot_err_deg",
            "mean_corr_rot_err_deg",
            "rmse_raw_odom_rot_err_deg",
            "rmse_pred_rot_err_deg",
            "rmse_corr_rot_err_deg",

            # Percentile
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
        metadata_columns = ["map", "dataset_id", "parameter_tag", "parameter_hash"]
        ordered_columns = legacy_columns + metadata_columns

        for run in ranked_runs:
            params = run.params
            summary = run.summary

            rows.append(
                {                    
                    "score": run.score,
                    "map": run.map_name,
                    "dataset_id": run.dataset_id,
                    "parameter_tag": run.parameter_tag,
                    "parameter_hash": run.parameter_hash,
                    "seed": run.seed,

                    # Grid parameters: playback sampling
                    "every_nth_beam_filter": params.every_nth_scan_filter,
                    "every_nth_beam_map": params.every_nth_scan_map,

                    # Grid parameters: OccupancyParams (OGM)
                    "increasing_probability": params.occupancy_params.increasing_probability,
                    "decreasing_probability": params.occupancy_params.decreasing_probability,
                    "min_log_odds": params.occupancy_params.min_log_odds,
                    "max_log_odds": params.occupancy_params.max_log_odds,

                    # Grid parameters: ScanMatcherParams
                    "occ_thres": params.scan_matcher_params.occ_thres,
                    "delta_r": params.scan_matcher_params.delta_r,
                    "surface_radius_m": params.scan_matcher_params.surface_radius_m,
                    "min_free_ratio": params.scan_matcher_params.min_free_ratio,

                    # Grid parameters: ICPParams
                    "max_n_points": params.icp_params.max_n_points,
                    "neighbors_pca": params.icp_params.neighbors_pca,
                    "max_iterations": params.icp_params.max_iterations,
                    "max_correspondence_distance": params.icp_params.max_correspondence_distance,
                    "min_corresp": params.icp_params.min_corresp,
                    "max_translation_jump": params.icp_params.max_translation_jump,
                    "max_rotation_jump_deg": cls._deg(params.icp_params.max_rotation_jump),
                    "max_acceptable_mean_error": params.icp_params.max_acceptable_mean_error,

                    "n_steps": summary.n_steps,
                    "scan_match_failed_count": summary.scan_match_failed_count,
                    "icp_failed_count": summary.icp_failed_count,
                    "icp_success_rate": summary.icp_success_rate,
                    "mean_icp_iterations": summary.mean_icp_iterations,
                    "count_too_few_points": summary.count_too_few_points,
                    "count_too_few_corresp": summary.count_too_few_corresp,
                    "infinite_h_or_g": summary.infinite_h_or_g,
                    "ill_cond_H": summary.ill_cond_H,
                    "infinite_dtransform": summary.infinite_dtransform,
                    "infinite_mean_err": summary.infinite_mean_err,
                    "best_transf_too_large": summary.best_transf_too_large,
                    "best_mean_err_too_large": summary.best_mean_err_too_large,
                    "mean_icp_err": summary.mean_icp_err,
                    "mean_best_trans_norm": summary.mean_best_trans_norm,
                    "max_best_trans_norm": summary.max_best_trans_norm,
                    "mean_best_rot_abs_deg": cls._deg(summary.mean_best_rot_abs),
                    "max_best_rot_abs_deg": cls._deg(summary.max_best_rot_abs),
                    "mean_raw_odom_trans_err": summary.mean_raw_odom_trans_err,
                    "mean_pred_trans_err": summary.mean_pred_trans_err,
                    "mean_corr_trans_err": summary.mean_corr_trans_err,
                    "rmse_raw_odom_trans_err": summary.rmse_raw_odom_trans_err,
                    "rmse_pred_trans_err": summary.rmse_pred_trans_err,
                    "rmse_corr_trans_err": summary.rmse_corr_trans_err,
                    "mean_raw_odom_rot_err_deg": cls._deg(summary.mean_raw_odom_rot_err),
                    "mean_pred_rot_err_deg": cls._deg(summary.mean_pred_rot_err),
                    "mean_corr_rot_err_deg": cls._deg(summary.mean_corr_rot_err),
                    "rmse_raw_odom_rot_err_deg": cls._deg(summary.rmse_raw_odom_rot_err),
                    "rmse_pred_rot_err_deg": cls._deg(summary.rmse_pred_rot_err),
                    "rmse_corr_rot_err_deg": cls._deg(summary.rmse_corr_rot_err),
                    "perc_95_corr_trans_err": summary.perc_95_corr_trans_err,
                    "perc_95_corr_rot_err": cls._deg(summary.perc_95_corr_rot_err),
                    "max_rolling_rmse_corr_trans_error": summary.max_rolling_rmse_corr_trans_error,
                    "max_rolling_rmse_corr_rot_error": cls._deg(summary.max_rolling_rmse_corr_rot_error),
                    "corr_worse_rate_trans": summary.corr_worse_rate_trans,
                    "corr_worse_rate_rot": summary.corr_worse_rate_rot,
                    "mean_corr_trans_improvm": summary.mean_corr_trans_improvm,
                    "mean_corr_rot_improvm_deg": cls._deg(summary.mean_corr_rot_improvm),
                    "final_raw_odom_drift_trans": summary.final_raw_odom_drift_trans,
                    "final_raw_odom_drift_rot_deg": cls._deg(summary.final_raw_odom_drift_rot),
                    "final_drift_trans": summary.final_drift_trans,
                    "final_drift_rot_deg": cls._deg(summary.final_drift_rot),
                    "mean_timing_sm_update_particle_ms": cls._ms(summary.mean_timing_sm_update_particle_s),
                    "mean_timing_sm_scan_match_update_pose_ms": cls._ms(summary.mean_timing_sm_scan_match_update_pose_s),
                    "mean_timing_sm_map_extension_ms": cls._ms(summary.mean_timing_sm_map_extension_s),
                    "mean_timing_sm_map_update_ms": cls._ms(summary.mean_timing_sm_map_update_s),
                    "mean_step_duration_ms": cls._ms(summary.mean_step_duration),

                    
                }
            )

        return pd.DataFrame(rows, columns=ordered_columns)


class ResultAggregatorScanMatching:
    @staticmethod
    def _groupby(df: pd.DataFrame, by, dropna: bool = False):
        try:
            return df.groupby(by, dropna=dropna)
        except TypeError:
            return df.groupby(by)

    def rank_by_score(self, ranked_run_df: pd.DataFrame, score_col: str, ascending: bool = True) -> pd.DataFrame:
        ranked_df = ranked_run_df.sort_values(by=score_col, ascending=ascending).reset_index(drop=True)

        if "rank" in ranked_df.columns:
            ranked_df = ranked_df.drop(columns=["rank"])

        ranked_df.insert(0, "rank", ranked_df.index + 1)
        return ranked_df

    def aggregate_by_dataset_and_param(self, ranked_run_df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {
            "dataset_id",
            "seed",
            "score",
            "rmse_corr_trans_err",
            "rmse_corr_rot_err_deg",
            "scan_match_success_rate",
            "parameter_tag",
            "parameter_hash",
        }
        missing = sorted(col for col in required_cols if col not in ranked_run_df.columns)
        if missing:
            raise ValueError(
                "aggregate_by_dataset_and_param missing required columns: " + ", ".join(missing)
            )

        agg_df = self._groupby(ranked_run_df, ["dataset_id", "parameter_hash"]).agg(
            parameter_tag=("parameter_tag", "first"),
            n_runs=("score", "size"),
            n_seeds=("seed", "nunique"),
            mean_score=("score", "mean"),
            worst_score=("score", "max"),
            std_score=("score", "std"),
            mean_rmse_corr_trans_err=("rmse_corr_trans_err", "mean"),
            worst_rmse_corr_trans_err=("rmse_corr_trans_err", "max"),
            mean_rmse_corr_rot_err_deg=("rmse_corr_rot_err_deg", "mean"),
            worst_rmse_corr_rot_err_deg=("rmse_corr_rot_err_deg", "max"),
            mean_scan_match_success_rate=("scan_match_success_rate", "mean"),
            worst_scan_match_success_rate=("scan_match_success_rate", "min"),
        ).reset_index()

        agg_df["std_score"] = agg_df["std_score"].fillna(0.0)
        agg_df["dataset_param_score"] = (
            1.0 * agg_df["mean_score"]
            + 0.5 * agg_df["worst_score"]
            + 0.2 * agg_df["std_score"]
        )

        return self.rank_by_score(agg_df, "dataset_param_score", ascending=True)

    def aggregate_by_params(self, agg_dataset_param_df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"dataset_param_score", "dataset_id", "parameter_hash", "parameter_tag"}
        missing = sorted(col for col in required_cols if col not in agg_dataset_param_df.columns)
        if missing:
            raise ValueError("aggregate_by_params missing required columns: " + ", ".join(missing))

        agg_param_df = self._groupby(agg_dataset_param_df, ["parameter_hash"]).agg(
            parameter_tag=("parameter_tag", "first"),
            n_datasets=("dataset_id", "nunique"),
            n_results=("dataset_param_score", "size"),
            mean_score=("dataset_param_score", "mean"),
            worst_score=("dataset_param_score", "max"),
            std_score=("dataset_param_score", "std"),
            mean_rmse_corr_trans_err=("mean_rmse_corr_trans_err", "mean"),
            worst_rmse_corr_trans_err=("worst_rmse_corr_trans_err", "max"),
            mean_rmse_corr_rot_err_deg=("mean_rmse_corr_rot_err_deg", "mean"),
            worst_rmse_corr_rot_err_deg=("worst_rmse_corr_rot_err_deg", "max"),
            mean_scan_match_success_rate=("mean_scan_match_success_rate", "mean"),
            worst_scan_match_success_rate=("worst_scan_match_success_rate", "min"),
        ).reset_index()

        agg_param_df["std_score"] = agg_param_df["std_score"].fillna(0.0)
        agg_param_df["global_score"] = (
            1.0 * agg_param_df["mean_score"]
            + 0.5 * agg_param_df["worst_score"]
            + 0.2 * agg_param_df["std_score"]
        )

        return self.rank_by_score(agg_param_df, "global_score", ascending=True)

    def build_ranked_parameter_overview(
        self,
        agg_param_df: pd.DataFrame,
        ranked_runs: Iterable[RankedRunScanMatching],
    ) -> pd.DataFrame:
        required_cols = {"rank", "global_score", "parameter_hash"}
        missing = sorted(col for col in required_cols if col not in agg_param_df.columns)
        if missing:
            raise ValueError(
                "build_ranked_parameter_overview missing required columns: " + ", ".join(missing)
            )

        params_by_hash = {}
        for run in ranked_runs:
            if run.parameter_hash is None:
                continue

            params_for_hash = ScanMatchingOptimizer.generate_params_for_hash(run.params)
            existing = params_by_hash.get(run.parameter_hash)
            if existing is None:
                params_by_hash[run.parameter_hash] = params_for_hash
            elif existing != params_for_hash:
                raise ValueError(
                    f"Inconsistent params detected for parameter_hash '{run.parameter_hash}'."
                )

        if not params_by_hash:
            return pd.DataFrame(columns=["rank", "global_score", "parameter_hash"])

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
