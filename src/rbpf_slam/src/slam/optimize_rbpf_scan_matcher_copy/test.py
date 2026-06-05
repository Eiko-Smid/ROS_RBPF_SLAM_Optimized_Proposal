#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .summary_combinator_scnamatching import FileLoader
from .result_writer_scanmatching import ResultWriterScanMatching 



def load_file(dir: str, filename: str) ->pd.DataFrame:
    dir = FileLoader._validate_dir(dir)
    path = dir / filename

    df = FileLoader.load_csv(path=path)

    return df



def adapt_df(df: pd.DataFrame) -> pd.DataFrame:
    df["max_corr_trans_err"] = 1.0
    df["max_corr_rot_err_deg"] = 1.0
    # df["scan_match_success_rate"] = 1.0
    return df


def store_df(df: pd.DataFrame, result_writer: ResultWriterScanMatching, stor_path: str):
    result_writer.write_dataframe_csv(
        path=stor_path,
        df=df, 
        override=False,
        float_decimals=6
    )


def main():    
    dir='/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/scan_matching/optimization_results/'
    param_filename='sm_4_1_params.json'
    rank_scored_summary_filename='sm_4_1_summary_rank_scored.csv'

    stor_path = dir + "_" + "sm_4_1_summary_rank_scored_update_del_later.csv"
    
    # Init
    result_writer = ResultWriterScanMatching()

    # Load data 
    df = load_file(dir=dir, filename=rank_scored_summary_filename)
    
    # adapt 
    df = adapt_df(df)

    # Store data
    store_df(
        df=df,
        result_writer=result_writer,
        stor_path=stor_path,
    )


if __name__ == "__main__":
    main()