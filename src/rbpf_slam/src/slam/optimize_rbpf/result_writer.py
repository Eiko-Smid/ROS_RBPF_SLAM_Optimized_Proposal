from pathlib import Path
import csv
import numbers
import math

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
	def write_ranked_runs_csv(
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
					"mean_trans_err_best_p",
					"mean_rot_err_deg_best_p",
					"rmse_trans_error_best_p",
					"rmse_rot_error_deg_best_p",
					"drift",
					"mean_neff",
					"mean_particle_weight_min",
					"mean_particle_weight_max",
					"mean_particle_weight_mean",
					"mean_step_duration_ms",
					"n_steps",
				]
			)

			for rank, run in enumerate(ranked_runs, start=1):
				summary = run.summary

				row = [
					rank,
					run.score,
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
					summary.get("mean_trans_err_best_p"),
					math.degrees(summary.get("mean_rot_err_best_p")) if summary.get("mean_rot_err_best_p") is not None else None,
					summary.get("rmse_trans_error_best_p"),
					math.degrees(summary.get("rmse_rot_error_best_p")) if summary.get("rmse_rot_error_best_p") is not None else None,
					summary.get("drift"),
					summary.get("mean_neff"),
					summary.get("mean_particle_weight_min"),
					summary.get("mean_particle_weight_max"),
					summary.get("mean_particle_weight_mean"),
					(summary.get("mean_step_duration", 0.0) or 0.0) * 1000.0,
					summary.get("n_steps"),
				]

				writer.writerow(
					[
						ResultWriter._format_csv_value(value, float_decimals=float_decimals)
						for value in row
					]
				)

		print(f"\nOptimization run has been saved to:\n{path}")


	@staticmethod
	def write_ranked_step_traces_csv(
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
					"seed",
					"tag",
					"every_nth_beam_filter",
					"every_nth_beam_map",
					"step_id",
					"neff",
					"trans_error_best_p",
					"rot_error_best_p",
					"trans_error",
					"rot_error",
					"particle_weight_min",
					"particle_weight_max",
					"particle_weight_mean",
					"scan_match_failed",
					"scan_match_fallback_failed",					
					"step_duration_ms",
				]
			)

			for rank, run in enumerate(ranked_runs, start=1):
				for step in run.step_results:
					row = [
						rank,
						run.seed,
						run.params.tag,
						run.params.every_nth_scan_filter,
						run.params.every_nth_scan_map,
						step.step_idx,
						step.neff,
						step.translation_error_best_p,
						step.rotation_error_best_p,
						step.translation_error,
						step.rotation_error,
						step.particle_weight_min,
						step.particle_weight_max,
						step.particle_weight_mean,
						step.scan_match_failed,
						step.scan_match_fallback_failed,
						(step.step_duration or 0.0) * 1000.0,
					]

					writer.writerow(
						[
							ResultWriter._format_csv_value(value, float_decimals=float_decimals)
							for value in row
						]
					)

		print(f"\nStep trace CSV files written to:\n{output_path}")
