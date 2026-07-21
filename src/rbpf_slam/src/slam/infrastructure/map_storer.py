import json
from pathlib import Path

import numpy as np


def save_final_map(
    output_dir: str,
    log_odds_map: np.ndarray,
    resolution: float,
    shift_x: float,
    shift_y: float,
    occupied_threshold: float,
    free_threshold: float,
    min_log_odds: float,
    max_log_odds: float,
) -> None:
    '''
    Save the final log-odds map and its metadata to the given output directory.

    Parameters
    ----------
    output_dir : str
        The directory where the map and metadata will be saved.
    log_odds_map : np.ndarray
        The final log-odds map as a 2D NumPy array.
    resolution : float
        The grid resolution of the map in meters per cell.
    shift_x : float
        The x-coordinate of the map's origin in meters.
    shift_y : float
        The y-coordinate of the map's origin in meters.
    occupied_threshold : float
        The log-odds threshold for occupied cells.
    free_threshold : float
        The log-odds threshold for free cells.
    min_log_odds : float
        The minimum log-odds value for the map.
    max_log_odds : float
        The maximum log-odds value for the map.
    '''
    # Define and create the output directory 
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    map_array = np.asarray(log_odds_map)

    if map_array.ndim != 2:
        raise ValueError("log_odds_map must be a 2D NumPy array.")

    # Store the original 2D log-odds map.
    np.save(
        output_path / "final_log_odds_map.npy",
        map_array,
        allow_pickle=False,
    )
    
    # Get map dimensions
    height, width = map_array.shape

    # Store metadata in a JSON file
    metadata = {
        "format_version": 1,
        "representation": "log_odds",
        "frame_id": "map",

        "resolution": float(resolution),
        "width": int(width),
        "height": int(height),

        # ROS-like origin of grid cell [0, 0].
        "origin": {
            "x": float(-shift_x),
            "y": float(-shift_y),
            "theta": 0.0,
        },

        "thresholds": {
            "occupied_log_odds": float(occupied_threshold),
            "free_log_odds": float(free_threshold),
        },

        "log_odds_limits": {
            "min": float(min_log_odds),
            "max": float(max_log_odds),
        },

        "dtype": str(map_array.dtype),
    }

    with open(
        output_path / "final_log_odds_map_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)