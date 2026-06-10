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

from .scorer import RunScorer
from .aggregator import ResultAggregator
from .result_writer import ResultWriter


'''
This script combines multiple run results together, evaluate them and find the best union parameter sets for the 
overall runs. It does so by doing the following steps:

    1. Load the summary ranked score csv files and the parameter json files from multiple runs.
    2. Validate that the loaded data is consistent and has the same parameter sets.
    3. Combine the summary ranked score csv files into one dataframe.
    4. Extract the columns needed for the scorer from the combined summary ranked score dataframe and convert them to the correct format if needed.
    5. Score the combined summary ranked score dataframe with the scorer and add the score as a new column to the dataframe.
    6. Build a ranked summary dataframe based on the score column and rank the dataframe by score.
    7. Aggregate the ranked summary dataframe by dataset and parameter set and compute the mean score for each parameter set and dataset.
    8. Aggregate the ranked summary dataframe by parameter set and compute the mean score for each parameter set.
    9. Build a ranked parameter overview dataframe by merging the aggregated parameter scores with the parameter json files and rank the dataframe by score.    
    10. Write results to csv and json files

'''

'''
TODO

1) Ensure unique rows in combine_summary_runs

    - Currently it can happen that we got duplicate rows in the df
    - If 2 loaded summary runs have the same playback id (map) and param grid was equal than they occur two times 
    - This must be fixed


2) Add possibility to exclude parameter ids (maps)
    - maybe we wanne exclude a bad map (cafe) from the summary
    - We shold add the possibility to do this at wish
    - Define list of playback ids that should be excluded
    - i list empty skip this
    - Else delete those data rows

3) Add excluder to exclude params from param overview 
    - Would be nice to exclude some columns in the params comparsion. 
    - This way we could combine results made from different measurement models
    - Also when metrics have been added or removed in the code and we make a grid run, than this run is not 
      comparable to the old runs with other params
       

'''

# Define data storage path
COMB_OPTM_SUMMARY_PATH= '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/comb_optimization_results/proposal_optm_30_summary'
PARAMETER_OVERVIEW_PATH = '/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/comb_optimization_results/proposal_optm_30_params.json'



@dataclass
class OptmResultData:
    dir: str
    param_filename: str
    rank_scored_summary_filename: str
    ranked_param_overview_filename: str

@dataclass
class LoadedOptmResultData:
    param_path: Path
    rank_scored_summary_path: Path
    ranked_param_overview_path: Path
    params: Dict[str, Any]
    rank_scored_summary: pd.DataFrame
    ranked_param_overview: pd.DataFrame
    

# Define data to load
OPTM_RESULT_DATA_LIST = [
    OptmResultData(
        dir='/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results',
        param_filename='proposal_optm_30_1_params.json',
        rank_scored_summary_filename='proposal_optm_30_1_summary_rank_scored.csv',
        ranked_param_overview_filename='proposal_optm_30_1_summary_ranked_param_overview.csv'
    ),
    OptmResultData(
        dir='/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results',
        param_filename='proposal_optm_30_2_params.json',
        rank_scored_summary_filename='proposal_optm_30_2_summary_rank_scored.csv',
        ranked_param_overview_filename='proposal_optm_30_2_summary_ranked_param_overview.csv'
    ),
]

CSV_FLOAT_DECIMALS = 6
# Decide if existing results should be overwritten or not. If true existing files with the same name will be overwritten 
# by the current results 
OVERRIDE_EXISTING_RESULTS = False

# Columns to delete
COLS_TO_DEL_SUMMARY_RANKED_SCORED = ["score"]
COLS_TO_DEL_PARAM_SUMMARY = ["rank", "global_score"]

# Data to filter (exclude)
FILTER_DUPL_BY_COL = ["dataset_id", "parameter_hash", "seed"]
FILTER_COL = "dataset_id"
FILTER_PLAYBACK_IDS = ["1779375646"]



# Mapping from RunScorer summary keys to columns
# Value format: [column_name, is_angle]
# We transform the angles from deg to rad in this script. For simpicity we keep the "_deg" in the column names!
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
    '''
    Test function to switch two columns in the second loaded dataframe. 
    '''
    df = loaded_optm_data_list[1].rank_scored_summary

    cols = list(df.columns)

    col_a = "n_particles"
    col_b = "parameter_hash"

    idx_a = cols.index(col_a)
    idx_b = cols.index(col_b)

    cols[idx_a], cols[idx_b] = cols[idx_b], cols[idx_a]

    loaded_optm_data_list[1].rank_scored_summary = df[cols]



class FileLoader:
    @staticmethod
    def load_json(path: Union[str, Path]):
        path = FileLoader._validate_path(path, exp_file_type=".json")

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_csv(path: Union[str, Path]) -> pd.DataFrame:
        path = FileLoader._validate_path(path, exp_file_type=".csv")
        return pd.read_csv(path)

    
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
    


class LoadOptmResultData:
    '''
    Class to load the optimization result data. This includes the parameter file and the rank scored summary csv file.
    '''
    def  __init__(self, file_loader: FileLoader):
        self.file_loader = file_loader

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
        base_dir = self.file_loader._validate_dir(dir=optm_res_data.dir)

        # Create file paths
        param_path = base_dir / optm_res_data.param_filename
        rank_summary_path = base_dir / optm_res_data.rank_scored_summary_filename
        ranked_param_overview_path = base_dir/ optm_res_data.ranked_param_overview_filename

        # Load data
        params = self.file_loader.load_json(param_path)
        rank_scored_summary = self.file_loader.load_csv(rank_summary_path)
        ranked_param_overview = self.file_loader.load_csv(ranked_param_overview_path)

        return LoadedOptmResultData(
            param_path=param_path,
            rank_scored_summary_path=rank_summary_path,
            ranked_param_overview_path=ranked_param_overview_path,
            params=params,
            rank_scored_summary=rank_scored_summary,
            ranked_param_overview=ranked_param_overview,            
        )



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
    def combine(
        df_list: List[pd.DataFrame],
        cols_to_del: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        '''
        Combines all ranked summary dfs into one df. Ensures that resulting df has unique rows. 
        '''
        # Validate data
        if len(df_list) == 0:
            raise ValueError("No loaded results given.")
        
        if len(df_list) == 1:
            if isinstance(df_list[0], pd.DataFrame):
                return df_list[0]
            else:
                raise ValueError("Expected a list of pandas DataFrames, but got a list with one element of type "
                                 f"{type(df_list[0])}.")

        # Storage for cleaned dfs
        cleaned_dfs: List[pd.DataFrame] = []

        # Preprocess dfs
        for df in df_list:
            df_new = df.copy()

            # Delete columns 
            if cols_to_del is not None:
                for col in cols_to_del:
                    if col in df_new.columns:
                        df_new = df_new.drop(columns=[col])
                
            cleaned_dfs.append(df_new)
        
        aligned_dfs = RankeScoredSummaryCombiner.check_and_corr_differences(cleaned_dfs)

        return aligned_dfs
        

    @staticmethod
    def check_and_corr_differences(df_list: List[pd.DataFrame]):
        '''
        Checks if all dfs have the same columns. Also checks for column order. If columns are the same but order differ
        the code automatically reorders all dfs to the reference dfs (first df) column order.
        '''
        # Extract reference df
        reference_col = df_list[0].columns

        # Define storage for dfs
        aligned_dfs: List[pd.DataFrame] = [df_list[0]]

        # Check if all dfs have the same columns and print differences
        for i, df in enumerate(df_list[1:], start=1):
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


# class RankeScoredSummaryCombinerCopy:
#     @staticmethod
#     def combine(loaded_results: List[LoadedOptmResultData]) -> Optional[pd.DataFrame]:
#         # Check if results exist
#         if len(loaded_results) == 0:
#             raise ValueError("No loaded results given.")

#         # Storage for cleaned dfs
#         cleaned_dfs: List[pd.DataFrame] = []

#         # Preprocess dfs
#         for loaded_result in loaded_results:
#             df = loaded_result.rank_scored_summary.copy()

#             if "score" in df.columns:
#                 df = df.drop(columns=["score"])

#             cleaned_dfs.append(df)
        
#         aligned_dfs = RankeScoredSummaryCombiner.check_and_corr_differences(cleaned_dfs)

#         return aligned_dfs
        

#     @staticmethod
#     def check_and_corr_differences(df_list: List[pd.DataFrame]):
#         # Extract reference df
#         reference_col = df_list[0].columns

#         # Define storage for dfs
#         aligned_dfs: List[pd.DataFrame] = [df_list[0]]

#         # Check if all dfs have the same columns and print differences
#         for i, df in enumerate(df_list[1:], start=1):
#             current_col = df.columns

#             # Find missing and additional columns
#             missing_col = reference_col.difference(current_col)
#             additional_col = current_col.difference(reference_col)

#             # Define indicators
#             same_names = len(missing_col) == 0 and len(additional_col) == 0
#             same_order = list(current_col) == list(reference_col)

#             # Handle different or not equal column names
#             if not same_names:
#                 print(f"\nColumn name mismatch in dataframe index {i}")

#                 if len(missing_col) > 0:
#                     print("\nMissing columns compared to reference:")
#                     for col in missing_col:
#                         print(f"  - {col}")

#                 if len(additional_col) > 0:
#                     print("\nAdditional columns compared to reference:")
#                     for col in additional_col:
#                         print(f"  - {col}")

#                 return None

#             # handle column order missmatch but same column names
#             if not same_order:
#                 print(f"\nColumn order mismatch in dataframe index {i}")
#                 print("Same column names found. Reordering dataframe to match reference order.")

#                 different_positions = [
#                     (pos, ref_col, cur_col)
#                     for pos, (ref_col, cur_col) in enumerate(zip(reference_col, current_col))
#                     if ref_col != cur_col
#                 ]

#                 print("\nDifferent column positions:")
#                 for pos, ref_col, cur_col in different_positions:
#                     print(f"  - position {pos}: reference='{ref_col}', current='{cur_col}'")

#                 df = df.loc[:, reference_col]

#             aligned_dfs.append(df)

#         return pd.concat(aligned_dfs, axis=0, ignore_index=True)


def load_data(load_optm_res_data: LoadOptmResultData) -> List[LoadedOptmResultData]:
    # Load optimization result data from previous runs 
    loaded_optm_data_list = []

    for optm_res_data in OPTM_RESULT_DATA_LIST:
        loaded_data = load_optm_res_data.load(optm_res_data)
        loaded_optm_data_list.append(loaded_data)

    return loaded_optm_data_list


def validate_data(loaded_optm_data_list: List[LoadedOptmResultData]):
    '''
    Validate if the loaded data has more than 1 element and if all data lists have same length. If not raise error. 
    '''
    params = [optm_data.params for optm_data in loaded_optm_data_list]
    rank_scored_summarys = [optm_data.rank_scored_summary for optm_data in loaded_optm_data_list]
    ranked_param_overviews = [optm_data.ranked_param_overview for optm_data in loaded_optm_data_list]
    
    len_params = len(params)
    len_rank_scored_summarys = len(rank_scored_summarys)
    len_ranked_param_overviews = len(ranked_param_overviews)

    # Check if all data lists have the same length
    if len_params != len_rank_scored_summarys or len_params != len_ranked_param_overviews:
        raise ValueError(
            f"Loaded data lists have different lengths: "
            f"params={len_params}, rank_scored_summarys={len_rank_scored_summarys}, "
            f"ranked_param_overviews={len_ranked_param_overviews}"
        )

    # check if the overall len is > 1
    if len_params < 2:
        raise ValueError(
            f"Need at least two loaded results to compare and combine data, but got {len_params}."
        )
    


def combine_summary_runs(
        loaded_results: List[LoadedOptmResultData],
        ranked_scored_combiner: RankeScoredSummaryCombiner,
) -> Optional[pd.DataFrame]:
    '''
    Combines all ranked summary dfs into one df. Ensures that resulting df has unique rows. 
    '''
    # Combine summary run dataframes from loaded results
    summary_run = ranked_scored_combiner.combine(
        df_list=[loaded_result.rank_scored_summary for loaded_result in loaded_results],
        cols_to_del=COLS_TO_DEL_SUMMARY_RANKED_SCORED,
    )

    return summary_run



def filter_playback(
    df: pd.DataFrame,
    filter_dupl_by_cols: Optional[List[str]] = None,
    filter_col: Optional[str] = None,
    filter_values: Optional[List[str]] = None,
):
    '''
    Erase duplicate rows and filter the given df by the given playback ids in the given column. If filter_col or filter_values is None, 
    the original df is returned. All rows with a value in filter_col that is in filter_values are deleted.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to filter.
    filter_dupl_by_cols: List[str]
        The column names to filter duplicates by. If None, no duplicate filtering is applied.
    filter_col: str
        The column name to filter by. If None, no filtering is applied.
    filter_values: List[str]
        The values to filter by. All rows with a value in filter_col that is in this list are deleted. 
        If None, no filtering is applied.

    Returns
    -------
    pd.DataFrame        
        The filtered dataframe.
    ''' 
    # Erase duplicate rows
    filtered_df: pd.DataFrame = df.copy()

    if filter_dupl_by_cols is not None:
        filtered_df = filtered_df.drop_duplicates(
            subset=filter_dupl_by_cols
        ).reset_index(drop=True)
    else:
        filtered_df = filtered_df.drop_duplicates().reset_index(drop=True)

    # Filter all columns that have a value in filter_col 
    if filter_col is not None and filter_values is not None:        
        filtered_df = filtered_df[~filtered_df[filter_col].isin(filter_values)].reset_index(drop=True)

    return filtered_df



def extract_and_convert_scorer_cols(
        summary_run: pd.DataFrame,
        col_mappings: Dict,
) -> Dict[str, pd.Series]:
    '''
    Extracts the columns from the given summary run dataframe based on the given column mappings. Convert all 
    columns marked as angle from deg to rad. Stores extracted columns into new df. Functions assumes that indicators
    are correct. Wrong column names will raise key error.

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
    pd.DataFrame
        A dataframe containing the extracted columns, where the columns are the expected scorer summary
        keys and the values are the corresponding columns from the original dataframe. 
    '''
    # Define df to store data
    scorer_df = pd.DataFrame(index=summary_run.index)

    for key, data in col_mappings.items():
        col_name, is_angle_deg = data
        try:            
            # Extract column and add to new df
            scorer_df[key] = summary_run.loc[:, col_name]

            # Transform angle from deg to rad if it is angle
            if is_angle_deg:
                scorer_df[key] = np.deg2rad(scorer_df[key])
            
        # Handle key errors
        except KeyError:
            raise KeyError(
                f"Required column '{col_name}' for scorer key '{key}' "
                f"does not exist in given dataframe."
            )
    
    return scorer_df



# def prepare_scorer(
#     summary_run_df: pd.DataFrame,
#     col_mappings: Dict,
# ):
#     summary_list = []

#     valid_fields = {field.name for field in fields(RunS)}
    
#     for i in range(summary_run_df.shape[0]):
#         # Init summary with none vals
#         summary = RunSummaryScanMatching.init_with_none()

#         for key, data in col_mappings.items():
#             col_name, is_angle_deg = data

#             # Check if data exists
#             if key not in valid_fields:
#                 raise ValueError(
#                     f"Mapping key '{key}' is not a valid RunSummaryScanMatching field."
#                 )
            
#             if col_name not in summary_run_df.columns:
#                 raise ValueError(
#                     f"Column '{col_name}' not found in summary_run_df."
#                 )
        
#             value = summary_run_df.loc[i, col_name]

#             # Convert tio rad if its an angle
#             if is_angle_deg:
#                 value = np.deg2rad(value)
            
#             # Set attribute
#             setattr(summary, key, value)

#         summary_list.append(summary)
    
#     return summary_list


def score_summary_runs(
        scorer_df: pd.DataFrame,
        run_scorer: RunScorer,
) -> pd.Series:
    score_series = scorer_df.apply(
        lambda row: run_scorer.score(row.to_dict()),
        axis=1,
    )
        
    return score_series



def build_summary_rank_scored(
        summary_run_df: pd.DataFrame,
        score_series: pd.Series,
        result_aggregator: ResultAggregator,
) -> pd.DataFrame:
    '''
    Adds the given score series as a new column to the given summary run dataframe. Sorts the df by score.
    '''
    # Add score series to summary run df
    summary_rank_scored_df = summary_run_df.copy()
    summary_rank_scored_df["score"] = score_series

    # Move score column to begining of df
    result_aggregator._place_col_after_col(
        df=summary_rank_scored_df,
        col="score",
        col_after=summary_rank_scored_df.columns[0],
    )

    # rank df by score
    summary_rank_scored_df = result_aggregator.rank_by_score(
        ranked_run_df=summary_rank_scored_df,
        score_col="score",
    )

    return summary_rank_scored_df



def build_summary_ranked_param_overview(
        loaded_optm_data_list: List[LoadedOptmResultData],
        ranked_scored_combiner: RankeScoredSummaryCombiner,
        agg_param_df: pd.DataFrame,
        result_aggregator: ResultAggregator,
):
    # Combine summary ranked runs
    summray_ranked_param_dfs = [data.ranked_param_overview for data in loaded_optm_data_list]
    comb_summary_ranked_params_df = ranked_scored_combiner.combine(
        df_list=summray_ranked_param_dfs,
        cols_to_del=COLS_TO_DEL_PARAM_SUMMARY,
    )

    print("\nCombined summary ranked param overview head:\n", comb_summary_ranked_params_df.head())

    # Keep only unique parameter sets
    unique_comb_sum_ranked_params_df: pd.DataFrame = comb_summary_ranked_params_df.drop_duplicates(subset=["parameter_hash"])
    print("\nUnique combined summary ranked param overview head:\n", unique_comb_sum_ranked_params_df.head())

    # Build ranked summary param run
    final_param_summary_ranked_df = unique_comb_sum_ranked_params_df.merge(
        agg_param_df[["parameter_hash", "global_score"]],
        on="parameter_hash",
        how="inner"
    )

    # Rank by score 
    final_param_summary_ranked_df = result_aggregator.rank_by_score(
        ranked_run_df=final_param_summary_ranked_df,
        score_col="global_score",
    )

    # Place score column after directly rank
    final_param_summary_ranked_df = ResultAggregator._place_col_after_col(
        df=final_param_summary_ranked_df,
        col="global_score",
        col_after=final_param_summary_ranked_df.columns[0],
    )
    print("\n\nFinal param summary:\n", final_param_summary_ranked_df.head())

    return final_param_summary_ranked_df
    


def write_results(
    
    summary_rank_scored_df: pd.DataFrame,
    agg_dataset_param_df: pd.DataFrame,
    agg_param_df: pd.DataFrame,
    param_summary_ranked_df: pd.DataFrame,
    param_json,
    result_writer: ResultWriter,
):
    # Define storage paths
    ranked_scored_path = COMB_OPTM_SUMMARY_PATH + "_" + "rank_scored.csv"
    agg_dataset_param_path = COMB_OPTM_SUMMARY_PATH + "_" + "agg_dataset_id_param.csv"
    agg_param_path = COMB_OPTM_SUMMARY_PATH + "_" + "agg_param.csv"
    ranked_param_overview_path = COMB_OPTM_SUMMARY_PATH + "_" + "ranked_param_overview.csv"
    param_json_path = PARAMETER_OVERVIEW_PATH

    # Write results
    result_writer.write_dataframe_csv(
        path=ranked_scored_path,
        df=summary_rank_scored_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,        
    )

    result_writer.write_dataframe_csv(
        path=agg_dataset_param_path,
        df=agg_dataset_param_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,        
    )

    result_writer.write_dataframe_csv(
        path=agg_param_path,
        df=agg_param_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,        
    )

    result_writer.write_dataframe_csv(
        path=ranked_param_overview_path,
        df=param_summary_ranked_df,
        override=OVERRIDE_EXISTING_RESULTS,
        float_decimals=CSV_FLOAT_DECIMALS,        
    )

    # Write json params
    with open(param_json_path, "w") as json_file:
        json.dump(param_json, json_file, indent=4)
    

def main():
    # Init 
    file_loader = FileLoader()
    optm_result_loader = LoadOptmResultData(file_loader=file_loader)
    params_diff_estimator = ParamsDiffEstimator()
    ranked_scored_combiner = RankeScoredSummaryCombiner()
    run_scorer = RunScorer()
    result_aggregator = ResultAggregator()
    result_writer = ResultWriter()
    
    # Load data
    loaded_optm_data_list = load_data(optm_result_loader)
    
    # Validate data
    validate_data(loaded_optm_data_list)

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
        ranked_scored_combiner=ranked_scored_combiner
    )

    if summary_run is None:
        raise ValueError("Could not combine summary dataframes due to column mismatches.")
    else:
        print("\nSuccessfully combined summary run dataframes:\n")
    
    # Filter certain columns and erase rows
    filtered_summary_df = filter_playback(
        df=summary_run,
        filter_dupl_by_cols=FILTER_DUPL_BY_COL,
        filter_col=FILTER_COL,
        filter_values=FILTER_PLAYBACK_IDS,
    )

    # Prepare scorer
    # summary_list = prepare_scorer(
    #     summary_run_df=filtered_summary_df,
    #     col_mappings=SCORER_SUMMARY_DF_MAPPINGS
    # )


    # Extract and convert scorer columns
    scorer_df = extract_and_convert_scorer_cols(
        summary_run=filtered_summary_df,
        col_mappings=SCORER_SUMMARY_DF_MAPPINGS,
    )

    print("\nSuccessfully extracted and converted scorer columns.")

    # score summary runs with scorer
    scored_series = score_summary_runs(
        scorer_df=scorer_df,
        run_scorer=run_scorer,
    )

    print("\nSuccessfully scored summary runs.")
    print("\nScored summary head:\n", scored_series.head())

    # Build summary rank scored df -> sorted by score
    summary_rank_scored_df = build_summary_rank_scored(
        summary_run_df=filtered_summary_df,
        score_series=scored_series,
        result_aggregator=result_aggregator,
    )
    print("\nSuccessfully built summary rank scored dataframe.")
    # print("\nSummary rank scored head:\n", summary_rank_scored_df.head())
    print(f"Shape of unioned summary ranked score df: {summary_rank_scored_df.shape}")

    # Aggregate results
    agg_dataset_param_df = result_aggregator.aggregate_by_dataset_and_param(summary_rank_scored_df)

    # Aggregate and rank by parameters 
    agg_param_df = result_aggregator.aggregate_by_params(agg_dataset_param_df)

    print(f"\nfinal summary ranked result:\n{agg_param_df.head()}")

    # Build ranked param overview
    param_summary_ranked_df = build_summary_ranked_param_overview(
        loaded_optm_data_list=loaded_optm_data_list,
        ranked_scored_combiner=ranked_scored_combiner,
        agg_param_df=agg_param_df,
        result_aggregator=result_aggregator,
    )

    print(f"\nSuccessfully created ranked parameter overview.")

    # Write results to csv
    write_results(
        result_writer=result_writer,
        summary_rank_scored_df=summary_rank_scored_df,
        agg_dataset_param_df=agg_dataset_param_df,
        agg_param_df=agg_param_df,
        param_summary_ranked_df=param_summary_ranked_df,
        param_json=loaded_optm_data_list[0].params, 
    )


if __name__ == "__main__":
    main()