#!/usr/bin/env python3

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

import numpy as np
import pandas as pd
import json

from dataclasses import asdict, dataclass, is_dataclass
from typing import Dict, Any, List, Union, Optional

from pathlib import Path


'''
This script will combine multiple run results together, evaluate them and find the best union parameter sets for the 
overall runs. In order to achive this the code does the following steps:

1) Load parameters and make compare check
    - Load the corresponding parameter sets for example "1779363559_test_params.json" and "1779363559_test_2_params.json"
    - Check if the same params has been used
    - If not mention this with a warning and asks user if he want to coninue
    - This means the user can self design if he wanne go on or not (true, false) 

2) Load the run result
    - We define a list of files and load the data 
    - Load the summary rank score data for example "1779363559_testsummary_rank_scored.csv" and "1779363559_test_2_summary_rank_scored.csv"    
    - Store inside dfs. 

3) Combine csv data (dfs)
    Clean way:
        - Define column mapping to current columns defined in "to_dataframe" except the score (will be computed in next step)
        - We need one map for each file we loaded. But this will result into a massive map cause we have over 100 columns
          in the score csv files. 
        - Concatenate the dfs based on the given mappings

    Dirty way:
        - Test of all columns from "to_dataframe" exists except the score (will be computed in next step) in the loaded csv files.
          if not then raise error.
        - The columns are the same so we are able to concate the dfs by just concatenating them.
      

4) Extract info and run scorer
    - Because it can be that the scorer changed between the two runs we need to ensure we score all the data with 
      the same scorer. 
    - SO we extract the columns of the dfs needed for the scorer. If not existing raise error
    - We define a dict at the beginning. Each key will be the data the scorer expects and each value will be a list
      containing the corresponding column names in the csv data of the files we loaded (2 files loaded -> 2 vals in list). 
      Here we need to check if we have as many list elements as we have files loaded. This we we can map the columns to the expected
      dat of the scorer
    - Since our scorer expects a summary dict we can easily call the scorer for each dataset once with the key and corresponding
      key value pair. 
      This way we got the score values from the scorer
    
'''

@dataclass
class OptmResultData:
    dir: str
    param_filename: str
    rank_scored_summary_filename: str

@dataclass
class LoadedOptmResultData:
    param_path: Path
    rank_scored_summary_path: Path
    params: Dict[str, Any]
    rank_scored_summary: pd.DataFrame
    


OPTM_RESULT_DATA_LIST = [
    OptmResultData(
        dir='/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results',
        param_filename='1779363559_test_params.json',
        rank_scored_summary_filename='1779363559_test_summary_rank_scored.csv'
    ),
    OptmResultData(
        dir='/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results',
        param_filename='1779363559_test_2_params.json',
        rank_scored_summary_filename='1779363559_test_2_summary_rank_scored.csv'
    ),
]


# Mapping from RunScorer summary keys to columns in:
# 1779363559_test_2_summary_rank_scored.csv
# Value format: [column_name, is_angle]
SCORER_SUMMARY_DF_MAPPINGS: Dict[str, List[Union[str, bool]]] = {
    "n_steps": ["n_steps", False],
    "proposal_n_samples": ["n_samples_dir", False],
    "rmse_translation_error": ["rmse_trans_error", False],
    "rmse_rotation_error": ["rmse_rot_error_deg", True],
    "drift_trans_err": ["drift_trans_err", False],
    "drift_rot_err": ["drift_rot_err_deg", True],
    "scan_match_failed_count": ["scan_match_failed_count", False],
    "mean_best_weighted_xj_pose_err_true": ["mean_best_weighted_xj_pose_err_true", False],
    "mean_mu_true_err_improves_over_sm_true": ["mean_mu_true_err_improves_over_sm_true", False],
    "mean_best_xj_true_err_improves_over_sm_true": ["mean_best_xj_true_err_improves_over_sm_true", False],
    "mean_min_xj_is_best_xj": ["mean_min_xj_is_best_xj", False],
    "median_log_meas_range": ["median_log_meas_range", False],
    "median_log_motion_range": ["median_log_motion_range", False],
    "mean_xj_eff_meas": ["mean_xj_eff_meas", False],
}


def sim_col_switch(loaded_optm_data_list: List[LoadedOptmResultData]):
    df = loaded_optm_data_list[1].rank_scored_summary

    cols = list(df.columns)

    col_a = "n_particles"
    col_b = "parameter_hash"

    idx_a = cols.index(col_a)
    idx_b = cols.index(col_b)

    cols[idx_a], cols[idx_b] = cols[idx_b], cols[idx_a]

    loaded_optm_data_list[1].rank_scored_summary = df[cols]



class LoadOptmResultData:
    '''
    Class to load the optimization result data. This includes the parameter file and the rank scored summary csv file.
    '''
    def load(self, optm_res_data: OptmResultData) -> LoadedOptmResultData:
        '''
        Loads the given optimization result data. This includes the parameter file and the rank scored summary csv file.
        Returns the loaded data and path information.
        
        Parameters
        ----------
        optm_res_data: OptmResultData
            The optimization result data to load, including directory and file names for parameters and rank scored summary
        
        Returns
        -------
        LoadedOptmResultData
            The loaded optimization result data, including the loaded parameters, rank scored summary dataframe and the paths
        '''
        # Validate base dir
        base_dir = self._validate_dir(dir=optm_res_data.dir)

        # Create file paths
        param_path = base_dir / optm_res_data.param_filename
        rank_summary_path = base_dir / optm_res_data.rank_scored_summary_filename

        # Load data
        params = self._load_params(param_path)
        rank_scored_summary = self._load_rank_score_summary(rank_summary_path)

        return LoadedOptmResultData(
            params=params,
            rank_scored_summary=rank_scored_summary,
            param_path=param_path,
            rank_scored_summary_path=rank_summary_path
        )


    @staticmethod
    def _load_params(path: Union[str, Path]) -> Dict[str, Any]:
        file_path = Path(path)

        LoadOptmResultData._validate_path(file_path)

        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
        

    @staticmethod
    def _load_rank_score_summary(path: Union[str, Path]) -> pd.DataFrame:
        file_path = Path(path)

        LoadOptmResultData._validate_path(file_path)

        return pd.read_csv(file_path)


    @staticmethod
    def _validate_dir(dir: Union[str, Path]) -> Path:
        '''
        Checks if the given directory exists and is a directory. 
        '''
        dir = Path(dir)

        if not dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {dir}")

        if not dir.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {dir}")
        
        return dir
    

    @staticmethod
    def _validate_path(path: Union[str, Path], exp_file_type: Union[str, None] = None) -> Path:
        '''
        Checks if the given path exists and is a file. Optionally checks if the file type matches the expected file type if
        provided.
        '''
        path = Path(path)
        # Validate path
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Path exists but is not a file: {path}")

        if exp_file_type is not None and path.suffix != exp_file_type:
            raise ValueError(f"Expected file type {exp_file_type}, but got {path.suffix}")

        return path


class ParamsDiffEstimator:
    '''
    Class to compare multiple parameter json files and check if they are equal. 
    '''
    def check_equal_params(self, loaded_results: List[LoadedOptmResultData]) -> bool:
        if len(loaded_results) < 2:
            raise ValueError("Need at least two loaded results to compare params.")

        reference_result = loaded_results[0]
        reference_params = reference_result.params
        reference_path = reference_result.param_path

        all_differences = {}

        for loaded_result in loaded_results[1:]:
            differences = self.compare_nested_params(
                reference_params,
                loaded_result.params,
            )

            if differences:
                all_differences[str(loaded_result.param_path)] = differences

        if all_differences:
            warning_msg = (
                "Not all parameter files are equal.\n"
                f"Reference file: {reference_path}\n\n"
            )

            for file_path, differences in all_differences.items():
                warning_msg += f"Different file: {file_path}\n"

                for difference in differences:
                    warning_msg += f"  - {difference}\n"

                warning_msg += "\n"

            print(warning_msg)
            return False

        return True
    

    @staticmethod
    def compare_nested_params(reference: Any, other: Any, path: str = "") -> List[str]:
        '''
        Checks recursively if the given parameters are equal. If not it returns a list of differences with
        the path to the different parameter and the values that differ. 
        '''
        differences = []

        if isinstance(reference, dict) and isinstance(other, dict):
            reference_keys = set(reference.keys())
            other_keys = set(other.keys())

            only_in_reference = reference_keys - other_keys
            only_in_other = other_keys - reference_keys
            common_keys = reference_keys & other_keys

            for key in sorted(only_in_reference):
                full_path = f"{path}.{key}" if path else key
                differences.append(f"Missing in other: {full_path}")

            for key in sorted(only_in_other):
                full_path = f"{path}.{key}" if path else key
                differences.append(f"Only in other: {full_path}")

            for key in sorted(common_keys):
                full_path = f"{path}.{key}" if path else key
                differences.extend(
                    ParamsDiffEstimator.compare_nested_params(
                        reference[key],
                        other[key],
                        full_path,
                    )
                )

        elif isinstance(reference, list) and isinstance(other, list):
            if len(reference) != len(other):
                differences.append(
                    f"Different list length at {path}: "
                    f"reference={len(reference)}, other={len(other)}"
                )

            min_len = min(len(reference), len(other))

            for i in range(min_len):
                full_path = f"{path}[{i}]"
                differences.extend(
                    ParamsDiffEstimator.compare_nested_params(
                        reference[i],
                        other[i],
                        full_path,
                    )
                )

        else:
            if reference != other:
                differences.append(
                    f"Different value at {path}: "
                    f"reference={reference}, other={other}"
                )

        return differences



class RankeScoredSummaryCombiner:
    @staticmethod
    def combine(loaded_results: List[LoadedOptmResultData]) -> Optional[pd.DataFrame]:
        # Check if results exist
        if len(loaded_results) == 0:
            raise ValueError("No loaded results given.")

        # Storage for cleaned dfs
        cleaned_dfs: List[pd.DataFrame] = []

        # Clean dfs
        for loaded_result in loaded_results:
            df = loaded_result.rank_scored_summary.copy()

            if "score" in df.columns:
                df = df.drop(columns=["score"])

            cleaned_dfs.append(df)

        # Extract reference df
        reference_col = cleaned_dfs[0].columns
        aligned_dfs: List[pd.DataFrame] = [cleaned_dfs[0]]

        # Check if all dfs have the same columns and print differences
        for i, df in enumerate(cleaned_dfs[1:], start=1):
            current_col = df.columns

            # Find missing and additional columns
            missing_col = reference_col.difference(current_col)
            additional_col = current_col.difference(reference_col)

            # Define indicators
            same_names = len(missing_col) == 0 and len(additional_col) == 0
            same_order = list(current_col) == list(reference_col)

            # Handle different or not equal column names
            if not same_names:
                print(f"\nColumn name mismatch in dataframe index {i}")

                if len(missing_col) > 0:
                    print("\nMissing columns compared to reference:")
                    for col in missing_col:
                        print(f"  - {col}")

                if len(additional_col) > 0:
                    print("\nAdditional columns compared to reference:")
                    for col in additional_col:
                        print(f"  - {col}")

                return None

            # handle column order missmatch but same column names
            if not same_order:
                print(f"\nColumn order mismatch in dataframe index {i}")
                print("Same column names found. Reordering dataframe to match reference order.")

                different_positions = [
                    (pos, ref_col, cur_col)
                    for pos, (ref_col, cur_col) in enumerate(zip(reference_col, current_col))
                    if ref_col != cur_col
                ]

                print("\nDifferent column positions:")
                for pos, ref_col, cur_col in different_positions:
                    print(f"  - position {pos}: reference='{ref_col}', current='{cur_col}'")

                df = df.loc[:, reference_col]

            aligned_dfs.append(df)

        return pd.concat(aligned_dfs, axis=0, ignore_index=True)


def load_data() -> List[LoadedOptmResultData]:
    load_optm_res_data = LoadOptmResultData()
    loaded_optm_data_list = []

    for optm_res_data in OPTM_RESULT_DATA_LIST:
        loaded_data = load_optm_res_data.load(optm_res_data)
        loaded_optm_data_list.append(loaded_data)

    return loaded_optm_data_list



def combine_summary_runs(
        loaded_results: List[LoadedOptmResultData],
        ranked_scpred_combiner: RankeScoredSummaryCombiner,
) -> Optional[pd.DataFrame]:
    summary_run = ranked_scpred_combiner.combine(loaded_results=loaded_results)

    return summary_run


def extract_and_convert_scorer_cols(
        summary_run: pd.DataFrame,
        col_mappings: Dict,
) -> Dict[str, pd.Series]:
    '''
    Extracts the columns from the given summary run dataframe based on the given column mappings. Convert all 
    columns makred as angle from rad to deg. Functions assumes that indicators are correct. Wrong column names will
    raise key error. 

    Parameters
    ----------
    summary_run: pd.DataFrame
        The summary run dataframe to extract the columns from. This is the combined dataframe of all runs.
    col_mappings: Dict
        The column mappings to use for the extraction. The keys are the expected scorer summary keys and a list 
        of [column_name, is_angle] where column_name is the name of the column in the dataframe and is_angle is
        a boolean indicating if the column represents an angle.

    Returns
    -------
    Dict[str, pd.Series]
        A dictionary containing the extracted columns as pandas Series, where the keys are the expected scorer summary
        keys and the values are the corresponding columns from the dataframe. 
    '''
    # get all pairs
    summarys_dict = {}

    for key, data in col_mappings.items():
        col_name, is_angle_deg = data
        try:
            # Convert angle from rad to deg if it is angle
            if is_angle_deg:
                summary_run[col_name] = np.rad2deg(summary_run[col_name])
            
            # Extract column and add to dict
            summarys_dict[key] = summary_run.loc[:, col_name]
        
        # Handle key errors
        except KeyError:
            raise KeyError(
                f"Required column '{col_name}' for scorer key '{key}' "
                f"does not exist in given dataframe."
            )
    
    return summarys_dict



def main():
    # Init 
    params_diff_estimator = ParamsDiffEstimator()
    ranked_scpred_combiner = RankeScoredSummaryCombiner()
    
    # Load data
    loaded_optm_data_list = load_data()
    for loaded_data in loaded_optm_data_list:
        print(f"Loaded data from {loaded_data.param_path} and {loaded_data.rank_scored_summary_path}")
        # print(f"Params: {loaded_data.params}")
        # print(f"Rank scored summary head:\n{loaded_data.rank_scored_summary.head()}\n\n")

    # Check if params are equal
    if params_diff_estimator.check_equal_params(loaded_optm_data_list):
        print("\nAll loaded parameter files are equal.")
    else:
        # Manually decide if want to go on with difference in params or not. 
        print("\nWarning: Loaded parameter files are not equal!")

        user_input = input("\nDo you want to continue with the analysis? (y/n): ")
        if user_input.lower() != "y":
            print("Exiting.")
            return
        

    # Combine summary run dfs
    summary_run = combine_summary_runs(
        loaded_results=loaded_optm_data_list,
        ranked_scpred_combiner=ranked_scpred_combiner
    )

    if summary_run is None:
        raise ValueError("Could not combine summary dataframes due to column mismatches.")
    else:
        print("\nSuccessfully combined summary run dataframes:\n")
    
    # Extract and convert scorer columns
    summarys_dict = extract_and_convert_scorer_cols(
        summary_run=summary_run,
        col_mappings=SCORER_SUMMARY_DF_MAPPINGS,
    )

    print("\nSuccessfully extracted and converted scorer columns.")


if __name__ == "__main__":
    main()