from pathlib import Path
import csv
import numbers
import math
import numpy as np

from typing import List

from .optimizer import RankedRun


class ResultWriter:
	@staticmethod
	def _pose_to_csv_value(pose):
		if pose is None:
			return ""
		return f"{float(pose[0]):.8f};{float(pose[1]):.8f};{float(pose[2]):.8f}"

	@staticmethod
	def _safe_filename(value: str) -> str:
		allowed = []
		for c in value:
			if c.isalnum() or c in ("-", "_"):
				allowed.append(c)
			else:
				allowed.append("_")
		return "".join(allowed).strip("_") or "run"

	@staticmethod
	def _format_csv_value(value, float_decimals: int):
		"""
		Format float-like values with fixed decimal places while leaving other values unchanged.
		"""
		if value is None:
			return np.nan

		if isinstance(value, bool):
			return value

		if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
			return f"{float(value):.{float_decimals}f}"

		return value

	@staticmethod
	def create_path_and_check_if_file_exists(path: str) -> bool:
		"""
		Ensures parent directory exists and returns whether the file already exists.
		"""
		path_obj = Path(path)
		path_obj.parent.mkdir(parents=True, exist_ok=True)
		return path_obj.exists()



	@staticmethod
	def write_run_steps_csv(
		output_path: str,
		ranked_runs: List[RankedRun],
		override: bool = False,
		float_decimals: int = 6,
	) -> None:
		"""
		Writes one per-step trace CSV per ranked run.

		This export is independent from RBPF internals and uses stored step results.
		"""
		output_file_path = Path(output_path)
		output_file_path.parent.mkdir(parents=True, exist_ok=True)

		if output_file_path.exists() and not override:
			print(f"Skipping step trace (exists, override=False): {output_file_path}")
			return

		with open(output_file_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(
				[
					"rank",

					# Parameters
					"seed",
					"tag",
					"every_nth_beam_filter",
					"every_nth_beam_map",
					"step_id",
					# "neff",
					# "trans_error_best_p",
					# "rot_error_best_p_deg",

					# Pose errors
					"trans_error",
					"rot_error_deg",
					# "particle_weight_min",
					# "particle_weight_max",
					# "particle_weight_mean",
					
					# Scan matcher information metrics
					"scan_match_failed",
					"scan_match_fallback_failed",
					"trans_err_sm_true",
					"rot_err_sm_true_deg",
					"pose_err_sm_true",
					
					# Proposal Pose errors metrics and xj weights
					"pose_err_mu_true",
					"min_xj_pose_err_true",
					"weight_min_xj_err",
					"best_weighted_xj_pose_err_true",
					"weight_best_xj",

					# Improvement metrics (Did Proposal beat sm)
					"min_xj_true_err_improves_over_sm_true",
					"best_xj_true_err_improves_over_sm_true",
					"mu_true_err_improves_over_sm_true",
					"min_xj_true_err_weight_score",
					"corr_xjs_weights",
					"best_xj_score",					
					"motion_rank_score",
					"meas_rank_score",
					"weight_ratio_min_best_weight",
					"log_motion_range",
					"log_meas_range",
					"log_weight_range",
					"xj_eff",

					# Mu errors of proposal
					"trans_err_mu_true",
					"rot_err_mu_true_deg",
					"trans_err_mu_sm",
					"rot_err_mu_sm_deg",
					"trans_err_mu_pred",
					"rot_err_mu_pred_deg",

					# Covariance and correlation metrics of proposal
					"prop_std_x",
					"prop_std_y",
					"prop_std_theta_deg",
					"prop_corr_xy",
					"prop_corr_x_theta",
					"prop_corr_y_theta",
					
					# Step information
					"step_duration_ms",
				]
			)

			for rank, run in enumerate(ranked_runs, start=1):
				for step in run.step_results:
					row = [
						rank,

						# Parameters
						run.seed,
						run.params.tag,
						run.params.every_nth_scan_filter,
						run.params.every_nth_scan_map,
						step.step_idx,
						# step.neff,
						# step.translation_error_best_p,
						# math.degrees(step.rotation_error_best_p) if step.rotation_error_best_p is not None else None,

						# Pose errors
						step.translation_error,
						math.degrees(step.rotation_error) if step.rotation_error is not None else None,
						# step.particle_weight_min,
						# step.particle_weight_max,
						# step.particle_weight_mean,

						# Scan matcher information metrics
						step.scan_match_failed,
						step.scan_match_fallback_failed,
						step.trans_err_sm_true,
						math.degrees(step.rot_err_sm_true) if step.rot_err_sm_true is not None else None,
						step.pose_err_sm_true,
						
						# Proposal Pose errors metrics and xj weights
						step.pose_err_mu_true,
						step.min_xj_pose_err_true,
						step.weight_min_xj_err,
						step.best_weighted_xj_pose_err_true,
						step.weight_best_xj,

						# Improvement metrics (Did Proposal beat sm)
						step.min_xj_true_err_improves_over_sm_true,
						step.best_xj_true_err_improves_over_sm_true,
						step.mu_true_err_improves_over_sm_true,
						step.min_xj_true_err_weight_score,
						step.corr_xjs_weights,
						step.best_xj_score,						
						step.motion_rank_score,
						step.meas_rank_score,
						step.weight_ratio_min_best_weight,
						step.log_motion_range,
						step.log_meas_range,
						step.log_weight_range,
						step.xj_eff,

						# Mu errors of proposal
						step.trans_err_mu_true,
						math.degrees(step.rot_err_mu_true) if step.rot_err_mu_true is not None else None,
						step.trans_err_mu_sm,
						math.degrees(step.rot_err_mu_sm) if step.rot_err_mu_sm is not None else None,
						step.trans_err_mu_pred,
						math.degrees(step.rot_err_mu_pred) if step.rot_err_mu_pred is not None else None,
						
						# Covariance and correlation metrics of proposal
						step.prop_std_x,
						step.prop_std_y,
						math.degrees(step.prop_std_theta) if step.prop_std_theta is not None else None,
						step.corr_xy,
						step.corr_x_theta,
						step.corr_y_theta,
						
						# Step information
						(step.step_duration or 0.0) * 1000.0,
					]

					writer.writerow(
						[
							ResultWriter._format_csv_value(value, float_decimals=float_decimals)
							for value in row
						]
					)

		print(f"\nStep trace CSV files written to:\n{output_path}")



	@staticmethod
	def write_run_summary_csv(
		path: str,
		ranked_runs: List[RankedRun],
		override: bool = False,
		float_decimals: int = 4,
	) -> None:
		"""
		Writes ranked RBPF optimization results to CSV.

		If the target file exists and override=False, nothing is written.

		Parameters
		----------
		float_decimals: int
			Number of decimal places used for floating-point values in the CSV.
		"""
		file_exists = ResultWriter.create_path_and_check_if_file_exists(path=path)

		if file_exists and not override:
			print("\nOptimization has not been saved because file already exists and override is set to False!")
			return

		with open(path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(
				[
					"rank",
					"score",
					
					# Parameters
					"seed",					
					"every_nth_beam_filter",
					"every_nth_beam_map",
					"n_particles",
					"sigma_measurement",
					"sigma_x_motion",
					"sigma_y_motion",
					"sigma_theta_motion",
					"ctrl_motion_fac",
					"ctrl_turn_fac",
					"neff_threshold",
					"proposal_sigma_xy",
					"proposal_sigma_theta",
					"proposal_n_samples",

					# Scan matcher information metrics
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

					# trans and rot errros of end pose
					"mean_trans_error",
					"mean_rot_error_deg",
					"rmse_trans_error",
					"rmse_rot_error_deg",
					# "mean_trans_err_best_p",
					# "mean_rot_err_deg_best_p",
					# "rmse_trans_error_best_p",
					# "rmse_rot_error_deg_best_p",
					"drift",
					
					# "mean_neff",
					# "mean_particle_weight_min",
					# "mean_particle_weight_max",					
					# "mean_particle_weight_mean",

					# Mu errors of proposal
					"mean_trans_err_mu_true",
					"mean_rot_err_mu_true_deg",
					"mean_trans_err_mu_sm",
					"mean_rot_err_mu_sm_deg",					
					"rmse_trans_err_mu_sm",
					"rmse_rot_err_mu_sm_deg",
					# "mean_trans_err_mu_pred",
					# "mean_rot_err_mu_pred_deg",
					# "rmse_trans_err_mu_pred",
					# "rmse_rot_err_mu_pred_deg",

					# Covariance and correlation metrics of proposal
					"mean_prop_std_xy",
					"mean_prop_std_theta_deg",
					"mean_prop_corr_xy",
					"mean_prop_corr_x_theta",
					"mean_prop_corr_y_theta",
					"mean_xj_eff",

					# Pose errrors metrics and xj weights
					"mean_pose_err_sm_true",	
					"mean_pose_err_mu_true",			
					"mean_min_xj_pose_err_true",
					# "rmse_min_xj_pose_err_true",
					"mean_weight_min_xj_err",
					"mean_best_weighted_xj_pose_err_true",
					"mean_weight_best_xj",
					"mean_weight_ratio_min_best_weight",
					"median_weight_ratio_min_best_weight",
					"mean_log_motion_range",
					"median_log_motion_range",
					"mean_log_meas_range",
					"median_log_meas_range",
					"mean_log_weight_range",

					# Improvement metrics (Did Proposal beat sm)
					"mean_min_xj_true_err_improves_over_sm_true",
					"rmse_min_xj_true_err_improves_over_sm_true",
					"mean_best_xj_true_err_improves_over_sm_true",
					"rmse_best_xj_true_err_improves_over_sm_true",

					# xj and weight analysis metrics
					"mean_min_xj_true_err_weight_score",
					"rmse_min_xj_true_err_weight_score",
					"mean_corr_xjs_weights",
					"rmse_corr_xjs_weights",
					"mean_best_xj_score",
					"rmse_best_xj_score",
					"mean_motion_rank_score",
					"mean_meas_rank_score",
					"mean_mu_true_err_improves_over_sm_true",
					"rmse_mu_true_err_improves_over_sm_true",
					
					# Step information
					"n_steps",
					"mean_step_duration_ms",
				]
			)

			for rank, run in enumerate(ranked_runs, start=1):
				summary = run.summary

				row = [
					rank,
					run.score,
					
					# Parameters
					run.seed,					
					run.params.every_nth_scan_filter,
					run.params.every_nth_scan_map,
					run.params.particle_params.n_particles,
					run.params.measurement_model_params.sigma_measurement,
					run.params.motion_model_params.sigma_x,
					run.params.motion_model_params.sigma_y,
					run.params.motion_model_params.sigma_theta,
					run.params.motion_model_params.ctrl_motion_fac,
					run.params.motion_model_params.ctrl_turn_fac,
					run.params.neff_threshold,
					run.params.proposal_sigma_xy,
					run.params.proposal_sigma_theta,
					run.params.proposal_n_samples,

					# Scan matcher information metrics
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

					# trans and rot errors of end pose
					summary.get("mean_translation_error"),
					math.degrees(summary.get("mean_rotation_error")) if summary.get("mean_rotation_error") is not None else None,
					summary.get("rmse_translation_error"),
					math.degrees(summary.get("rmse_rotation_error")) if summary.get("rmse_rotation_error") is not None else None,
					# summary.get("mean_trans_err_best_p"),
					# math.degrees(summary.get("mean_rot_err_best_p")) if summary.get("mean_rot_err_best_p") is not None else None,
					# summary.get("rmse_trans_error_best_p"),
					# math.degrees(summary.get("rmse_rot_error_best_p")) if summary.get("rmse_rot_error_best_p") is not None else None,
					summary.get("drift"),
					
					# summary.get("mean_neff"),
					# summary.get("mean_particle_weight_min"),
					# summary.get("mean_particle_weight_max"),					
					# summary.get("mean_particle_weight_mean"),
					
					# Mu errors of proposal
					summary.get("mean_trans_err_mu_true"),
					math.degrees(summary.get("mean_rot_err_mu_true")) if summary.get("mean_rot_err_mu_true") is not None else None,		
					summary.get("mean_trans_err_mu_sm"),
					math.degrees(summary.get("mean_rot_err_mu_sm")) if summary.get("mean_rot_err_mu_sm") is not None else None,
					summary.get("rmse_trans_err_mu_sm"),
					math.degrees(summary.get("rmse_rot_err_mu_sm")) if summary.get("rmse_rot_err_mu_sm") is not None else None,
					# summary.get("mean_trans_err_mu_pred"),
					# math.degrees(summary.get("mean_rot_err_mu_pred")) if summary.get("mean_rot_err_mu_pred") is not None else None,
					# summary.get("rmse_trans_err_mu_pred"),
					# math.degrees(summary.get("rmse_rot_err_mu_pred")) if summary.get("rmse_rot_err_mu_pred") is not None else None,
					
					# Covariance and correlation metrics of proposal
					summary.get("mean_prop_std_xy"),
					math.degrees(summary.get("mean_prop_std_theta")) if summary.get("mean_prop_std_theta") is not None else None,
					summary.get("mean_prop_corr_xy"),
					summary.get("mean_prop_corr_x_theta"),
					summary.get("mean_prop_corr_y_theta"),
					summary.get("mean_xj_eff"),
					
					# Pose errors metrics and xj weights
					summary.get("mean_pose_err_sm_true"),
					summary.get("mean_pose_err_mu_true"),
					summary.get("mean_min_xj_pose_err_true"),
					# summary.get("rmse_min_xj_pose_err_true"),
					summary.get("mean_weight_min_xj_err"),
					summary.get("mean_best_weighted_xj_pose_err_true"),
					summary.get("mean_weight_best_xj"),
					summary.get("mean_weight_ratio_min_best_weight"),
					summary.get("median_weight_ratio_min_best_weight"),
					summary.get("mean_log_motion_range"),
					summary.get("median_log_motion_range"),
					summary.get("mean_log_meas_range"),
					summary.get("median_log_meas_range"),
					summary.get("mean_log_weight_range"),
					
					# Improvement metrics (Did Proposal beat sm)
					summary.get("mean_min_xj_true_err_improves_over_sm_true"),
					summary.get("rmse_min_xj_true_err_improves_over_sm_true"),
					summary.get("mean_best_xj_true_err_improves_over_sm_true"),
					summary.get("rmse_best_xj_true_err_improves_over_sm_true"),

					# xj and weight analysis metrics
					summary.get("mean_min_xj_true_err_weight_score"),
					summary.get("rmse_min_xj_true_err_weight_score"),
					summary.get("mean_corr_xjs_weights"),
					summary.get("rmse_corr_xjs_weights"),
					summary.get("mean_best_xj_score"),
					summary.get("rmse_best_xj_score"),
					summary.get("mean_motion_rank_score"),
					summary.get("mean_meas_rank_score"),
					summary.get("mean_mu_true_err_improves_over_sm_true"),
					summary.get("rmse_mu_true_err_improves_over_sm_true"),
					
					# Step information
					summary.get("n_steps"),
					(summary.get("mean_step_duration", 0.0) or 0.0) * 1000.0,
				]

				writer.writerow(
					[
						ResultWriter._format_csv_value(value, float_decimals=float_decimals)
						for value in row
					]
				)

		print(f"\nOptimization run has been saved to:\n{path}")
