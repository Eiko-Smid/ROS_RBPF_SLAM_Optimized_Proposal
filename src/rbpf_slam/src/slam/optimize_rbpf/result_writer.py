from pathlib import Path
import csv

from dataclasses import asdict
from typing import List
from .optimizer import RankedRun


class ResultWriter:
    @staticmethod
    def create_path_and_check_if_file_exists(path: str):
        '''
        Creates the given path if it does not exist and checks if the file exists.
        Returns information if file exists or not.

        Parameters
        ----------
        path: str
            The path to create and check.
        
        Returns
        -------
        bool
            True if file exists, False if not.
        '''
        # Check if path exists, if not create it
        path_ = Path(path)
        path_.parent.mkdir(parents=True, exist_ok=True)

        # Check if path exists
        if path_.exists():
            return True
        else:
            return False


    @staticmethod
    def write_ranked_runs_csv(path: str, ranked_runs: List[RankedRun], override: bool=False) -> None:
        '''
        Writes the ranked runs to a CSV file. If override is True an existing file will be overridden, otherwise no 
        data will be written. 

        Parameters
        ----------
        path: str
            The path to the CSV file.
        ranked_runs: List[RankedRun]
            The list of ranked runs to write.
        override: bool, optional
            Whether to override the file if it exists. Default is False.

        '''
        # Check if path exists 
        file_exists = ResultWriter.create_path_and_check_if_file_exists(path=path)

        if (not file_exists) or (file_exists and override):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "score",
                    "tag",
                    "max_corr_dist",
                    "neighbors_pca",
                    "occ_thres",
                    "delta_r",
                    "max_n_points",
                    "mean_translation_error",
                    "rmse_translation_error",
                    "mean_rotation_error",
                    "fallback_count",
                    "mean_step_duration",
                    "mean_icp_iterations",
                ])

                for r in ranked_runs:
                    writer.writerow([
                        r.score,
                        r.params.tag,
                        r.params.icp.max_correspondence_distance,
                        r.params.icp.neighbors_pca,
                        r.params.scan_matcher.occ_thres,
                        r.params.scan_matcher.delta_r,
                        r.params.icp.max_n_points,
                        r.summary["mean_translation_error"],
                        r.summary["rmse_translation_error"],
                        r.summary["mean_rotation_error"],
                        r.summary["fallback_count"],
                        r.summary["mean_step_duration"] *1000,  # convert to ms
                        r.summary["mean_icp_iterations"],
                    ])
            print(f"\nOptimization run has been saved to:\n{path}")
        else:
            print(f"\nOptimization has not been saved cause file already exists and override is set to False!")

