from pathlib import Path
import csv

from typing import List

from .optimizer import RankedRun


class ResultWriter:
	@staticmethod
	def create_path_and_check_if_file_exists(path: str) -> bool:
		"""
		Ensures parent directory exists and returns whether the file already exists.
		"""
		path_obj = Path(path)
		path_obj.parent.mkdir(parents=True, exist_ok=True)
		return path_obj.exists()


	@staticmethod
	def write_ranked_runs_csv(path: str, ranked_runs: List[RankedRun], override: bool = False) -> None:
		"""
		Writes ranked RBPF optimization results to CSV.

		If the target file exists and override=False, nothing is written.
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
					"neff_threshold",
					"rmse_translation_error",
					"rmse_rotation_error",
					"drift",
					"mean_neff",
					"mean_step_duration_ms",
					"n_steps",
				]
			)

			for rank, run in enumerate(ranked_runs, start=1):
				summary = run.summary

				writer.writerow(
					[
						rank,
						run.score,
						run.params.tag,
						summary.get("n_particles"),
						summary.get("sigma_measurement"),
						summary.get("neff_threshold"),
						summary.get("rmse_translation_error"),
						summary.get("rmse_rotation_error"),
						summary.get("drift"),
						summary.get("mean_neff"),
						(summary.get("mean_step_duration", 0.0) or 0.0) * 1000.0,
						summary.get("n_steps"),
					]
				)

		print(f"\nOptimization run has been saved to:\n{path}")
