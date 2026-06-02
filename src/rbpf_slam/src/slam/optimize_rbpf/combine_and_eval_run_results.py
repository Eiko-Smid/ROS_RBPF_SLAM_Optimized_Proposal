#!/usr/bin/env python3

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

import numpy as np
import pandas as pd
import json

from dataclasses import asdict, dataclass, is_dataclass
from typing import Dict, Any, List, Union

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
    params: Dict[str, Any]
    rank_scored_summary: pd.DataFrame
    param_path: Path
    rank_scored_summary_path: Path


OPTM_RESULT_DATA_LIST = [
    OptmResultData(
        dir='/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results',
        param_filename='1779363559_test_params.json',
        rank_scored_summary_filename='1779363559_testsummary_rank_scored.csv'
    ),
    OptmResultData(
        dir='/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optimization_results',
        param_filename='1779363559_test_2_params.json',
        rank_scored_summary_filename='1779363559_test_2_summary_rank_scored.csv'
    ),
]


class LoadOptmResultData:
    def load(self, optm_res_data: OptmResultData) -> LoadedOptmResultData:
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
        dir = Path(dir)

        if not dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {dir}")

        if not dir.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {dir}")
        
        return dir
    

    @staticmethod
    def _validate_path(path: Union[str, Path], exp_file_type: Union[str, None] = None) -> Path:
        path = Path(path)
        # Validate path
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Path exists but is not a file: {path}")

        if exp_file_type is not None and path.suffix != exp_file_type:
            raise ValueError(f"Expected file type {exp_file_type}, but got {path.suffix}")

        return path


    # @staticmethod
    # def check_equal_params(loaded_results: List[LoadedOptmResultData]):
    #     '''
    #     Checks if the loaded parameter files are equal. Returns True if they are equal, otherwise false.
    #     '''
    #     if len(loaded_results) < 2:
    #         raise ValueError("Need at least two loaded results to compare params.")

    #     reference_params = loaded_results[0].params
    #     reference_path = loaded_results[0].param_path

    #     different_param_files = []

    #     for loaded_result in loaded_results[1:]:
    #         if loaded_result.params != reference_params:
    #             different_param_files.append(loaded_result.param_path)

    #     if different_param_files:
    #         return False

    #     return True


class ParamsDiffEstimator:
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


def load_data():
    load_optm_res_data = LoadOptmResultData()
    loaded_optm_data_list = []

    for optm_res_data in OPTM_RESULT_DATA_LIST:
        loaded_data = load_optm_res_data.load(optm_res_data)
        loaded_optm_data_list.append(loaded_data)

    return loaded_optm_data_list



def combine_summary_runs():
    pass    


def main():
    # Init 
    params_diff_estimator = ParamsDiffEstimator()
    
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
        
    # Combine csv data
    combine_summary_runs()

    

if __name__ == "__main__":
    main()