from pathlib import Path
import csv
import numbers
import math
import numpy as np
import pandas as pd

from typing import List

from .optimizer import RankedRun


class ResultWriter:

	@staticmethod
	def _optional_ms(value_s):
		return value_s * 1000.0 if value_s is not None else None


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
	def write_dataframe_csv(
		path: str,
		df: pd.DataFrame,
		override: bool = False,
		float_decimals: int = 6,
		cols_to_use: List[str] = None,
		cols_to_exclude: List[str] = None,
		label: str = "DataFrame",
	) -> None:
		"""
		Write a DataFrame to CSV with optional float formatting and overwrite protection. If cols_to_use is 
		provided, only those columns will be written. If only cols_to_exclude is provided, those columns will
		be excluded. If both are given then cols_to_exclude will be ignored. 

		Parameters
		----------
		path : str
			Path to the CSV file.
		df : pd.DataFrame
			DataFrame to write.
		override : bool, optional
			If True, overwrite existing file. Default is False.
		float_decimals : int, optional
			Number of decimal places for float formatting. Default is 6.
		cols_to_use : List[str], optional
			List of columns to include in the output. If None, all columns are included.
		cols_to_exclude : List[str], optional
			List of columns to exclude from the output. Ignored if cols_to_use is provided.
		label : str, optional
			Label for the DataFrame in print statements. Default is "DataFrame".
		"""
		formatted_df = df.copy()

		file_exists = ResultWriter.create_path_and_check_if_file_exists(path=path)

		if file_exists and not override:
			print(f"\n{label} has not been saved because file already exists and override is set to False!\n{path}")
			return

		# Exclude columns from df
		if cols_to_use is None and cols_to_exclude:
			formatted_df = formatted_df.drop(columns=cols_to_exclude, errors="ignore")

		# Use only specific columns
		if cols_to_use is not None:
			existing_cols = [col for col in cols_to_use if col in df.columns]
			formatted_df = formatted_df.loc[:, existing_cols]
		
		# Format columns of df. _format_csv_value can be adapted such that the columns adapt to it, too.
		for col in formatted_df.columns:
			formatted_df[col] = formatted_df[col].map(
				lambda value: ResultWriter._format_csv_value(value, float_decimals=float_decimals)
			)

		formatted_df.to_csv(path, index=False)
		print(f"\n{label} has been saved to:\n{path}")


	@staticmethod
	def write_proposal_weights_csv(
		output_path: str,
		ranked_runs: List[RankedRun],
		override: bool = False,
		float_decimals: int = 6,
	) -> None:
		"""
		Writes one row per step and per proposal sample j with raw (not normalized) values.
		"""
		output_file_path = Path(output_path)
		output_file_path.parent.mkdir(parents=True, exist_ok=True)

		if output_file_path.exists() and not override:
			print(f"Skipping proposal weights trace (exists, override=False): {output_file_path}")
			return

		with open(output_file_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(
				[
					"rank",
					"seed",
					"step_id",
					"j",
					"xj_pose_err",
					"xj_weight",
					"xj_motion",
					"xj_meas",
				]
			)

			for rank, run in enumerate(ranked_runs, start=1):
				for step in run.step_results:
					indices = step.xj_indices
					pose_err = step.xj_pose_err
					weights = step.xj_weight
					motion = step.xj_motion
					meas = step.xj_meas

					if not indices or not pose_err or not weights or not motion or not meas:
						row = [
							rank,
							run.seed,
							step.step_idx,
							"none",
							"none",
							"none",
							"none",
							"none",
						]
						writer.writerow(
							[
								ResultWriter._format_csv_value(value, float_decimals=float_decimals)
								for value in row
							]
						)
						continue

					n = min(len(indices), len(pose_err), len(weights), len(motion), len(meas))
					if n <= 0:
						row = [
							rank,
							run.seed,
							step.step_idx,
							"none",
							"none",
							"none",
							"none",
							"none",
						]
						writer.writerow(
							[
								ResultWriter._format_csv_value(value, float_decimals=float_decimals)
								for value in row
							]
						)
						continue
					for i in range(n):
						row = [
							rank,
							run.seed,
							step.step_idx,
							indices[i],
							pose_err[i],
							weights[i],
							motion[i],
							meas[i],
						]
						writer.writerow(
							[
								ResultWriter._format_csv_value(value, float_decimals=float_decimals)
								for value in row
							]
						)

		print(f"\nProposal weights CSV files written to:\n{output_path}")

