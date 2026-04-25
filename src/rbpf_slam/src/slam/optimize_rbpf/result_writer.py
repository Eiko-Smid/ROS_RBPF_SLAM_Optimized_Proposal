from pathlib import Path
import csv
import numbers
import math

from typing import List

from .optimizer import RankedRun


class ResultWriter:
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
					"tag",
					"n_particles",
					"sigma_measurement",
					"every_nth_beam",
					"proposal_sigma_xy",
					"proposal_sigma_theta",
					"proposal_n_samples",
					"neff_threshold",
					"scan_match_failed_count",
					"scan_match_fallback_failed_count",
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
					summary.get("sigma_measurement"),
					run.params.measurement_model_params.every_nth_scan,
					run.params.proposal_sigma_xy,
					run.params.proposal_sigma_theta,
					run.params.proposal_n_samples,
					summary.get("neff_threshold"),
					summary.get("scan_match_failed_count"),
					summary.get("scan_match_fallback_failed_count"),
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
