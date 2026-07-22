import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


class MapDataHandler:
    """
    Stores and loads the final log-odds map of one tuning run.

    Two files are used:

        final_log_odds_map.npy
            The original two-dimensional NumPy log-odds map.

        final_log_odds_map_metadata.json
            Metadata required to correctly place and display the map,
            including resolution, origin, dimensions and thresholds.

    The map is deliberately stored as a 2D array and is not raveled.
    Raveling is only required when transferring the map into a ROS message.
    """
    @staticmethod
    def _prepare_map(log_odds_map: np.ndarray) -> np.ndarray:
        """
        Convert the supplied map into a NumPy array and ensure that it is
        a non-empty numeric 2D array.
        """
        map_array = np.asarray(log_odds_map)

        if map_array.ndim != 2:
            raise ValueError(
                "log_odds_map must be 2D, got shape {}."
                .format(map_array.shape)
            )

        if map_array.size == 0:
            raise ValueError("log_odds_map must not be empty.")

        if not np.issubdtype(map_array.dtype, np.number):
            raise ValueError(
                "log_odds_map must contain numeric values."
            )

        return map_array


    @staticmethod
    def _validate_loaded_data(
        log_odds_map: np.ndarray,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Check whether the loaded map dimensions agree with the dimensions
        stored in the metadata.
        """
        if not isinstance(metadata, dict):
            raise ValueError("The loaded metadata must be a dictionary.")

        required_keys = {
            "resolution",
            "width",
            "height",
            "origin",
            "thresholds",
            "log_odds_limits",
        }

        missing_keys = required_keys.difference(metadata.keys())

        if missing_keys:
            raise ValueError(
                "Metadata is missing required keys: {}."
                .format(sorted(missing_keys))
            )

        expected_shape = (
            int(metadata["height"]),
            int(metadata["width"]),
        )

        if log_odds_map.shape != expected_shape:
            raise ValueError(
                "Loaded map shape {} does not match metadata shape {}."
                .format(log_odds_map.shape, expected_shape)
            )

        resolution = float(metadata["resolution"])

        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError(
                "The stored map resolution must be finite and positive."
            )


    @classmethod
    def save(
        cls,
        output_dir: str,
        log_odds_map: np.ndarray,
        resolution: float,
        shift_x: float,
        shift_y: float,
        occupied_threshold: float,
        free_threshold: float,
        min_log_odds: float,
        max_log_odds: float,
        map_filename: str = "log_odds_map.npy",
        metadata_filename: str = "log_odds_map_metadata.json",
    ) -> None:
        """
        Save a final 2D log-odds map and its metadata.

        Parameters
        ----------
        output_dir:
            Directory in which the map and metadata files are stored.

        log_odds_map:
            Final two-dimensional log-odds map.

        resolution:
            Size of one grid cell in metres.

        shift_x, shift_y:
            OGM shifts used for converting world coordinates into grid
            indices. The ROS-compatible map origin is stored as
            (-shift_x, -shift_y).

        occupied_threshold:
            Log-odds threshold above which a cell is considered occupied.

        free_threshold:
            Log-odds threshold below which a cell is considered free.

        min_log_odds, max_log_odds:
            Minimum and maximum allowed log-odds values.

        map_filename:
            Filename to use for the NumPy map file.

        metadata_filename:
            Filename to use for the JSON metadata file.
        """
        map_array = cls._prepare_map(log_odds_map)

        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("resolution must be finite and positive.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        map_path = output_path / map_filename
        metadata_path = output_path / metadata_filename

        height, width = map_array.shape

        metadata = {
            "format_version": 1,
            "representation": "log_odds",
            "frame_id": "map",
            "resolution": float(resolution),
            "width": int(width),
            "height": int(height),
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

        np.save(
            str(map_path),
            map_array,
            allow_pickle=False,
        )

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

    @classmethod
    def load(
        cls,
        input_dir: str,
        map_filename: str = "final_log_odds_map.npy",
        metadata_filename: str = "final_log_odds_map_metadata.json",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a final 2D log-odds map and its metadata.

        Parameters
        ----------
        input_dir:
            Directory containing the map and metadata files created by
            FinalMapStorage.save().

        map_filename:
            Filename of the NumPy map file to load.

        metadata_filename:
            Filename of the JSON metadata file to load.

        Returns
        -------
        log_odds_map:
            Loaded two-dimensional NumPy log-odds map.

        metadata:
            Dictionary containing resolution, origin, dimensions,
            thresholds and log-odds limits.
        """
        input_path = Path(input_dir)

        map_path = input_path / map_filename
        metadata_path = input_path / metadata_filename

        if not map_path.is_file():
            raise FileNotFoundError(
                "Map file not found: {}".format(map_path)
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                "Metadata file not found: {}".format(metadata_path)
            )

        log_odds_map = np.load(
            str(map_path),
            allow_pickle=False,
        )

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        log_odds_map = cls._prepare_map(log_odds_map)
        cls._validate_loaded_data(log_odds_map, metadata)

        return log_odds_map, metadata


