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



def downsample_pointcloud_spatial(pointcloud: np.ndarray, grid_size: float) -> np.ndarray:
    pointcloud = np.asarray(pointcloud, dtype=float)

    if pointcloud.ndim != 2 or pointcloud.shape[0] != 2:
        return np.empty((0, 2), dtype=float)
    
    if grid_size is None or grid_size <= 0.0:
        return pointcloud
    
    n_points = pointcloud.shape[0]

    if n_points == 0:
        return pointcloud
    
    # Compute indices of grid for each point. Same logic as point to cell transformation in ogm
    grid_indices = np.floor(pointcloud / grid_size).astype(np.int64)

    # Find unique indices -> all other points will be erased
    _, unique_indices = np.unique(grid_indices, axis=0, return_index=True)

    # Extract all unique indices from pointcloud. Sorting unique_indices restores deterministic original pointcloud order,
    # not geometric order.
    sorted_subsampled_pointcloud = pointcloud[np.sort(unique_indices)]

    return sorted_subsampled_pointcloud


def test_downsample_pointcloud_spatial():
    x_vals = np.arange(-0.4, 0.4, 0.05)
    y_vals = x_vals.copy()

    print(x_vals)

    x, y = np.meshgrid(x_vals, y_vals)
    pointcloud = np.stack((x.flatten(), y.flatten()), axis=1)

    print(f"\npointcloud: {pointcloud}")



def test():
    x_vals = np.arange(-0.4, 0.4, 0.05)
    grid_len = 0.1
    print(f"x vals: {x_vals}")

    indices = np.floor(x_vals / grid_len).astype(np.int64)

    print(f"\nIndices: {indices}")


def main():    
    test()


if __name__ == "__main__":
    main()