#!/usr/bin/env python3
"""
Organize proposal optimization result files into one folder per experiment.

Example source file:
    proposal_optm_1_1_summary_agg_param.csv

Result:
    proposal_optm_1_1/
        agg_param.csv

The original source files are copied, not moved.
"""
from typing import Dict
from pathlib import Path
import re
import shutil


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directory containing the proposal_optm_* files.
# Change this path before running the script.
FILE_DIR = Path(
    "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optm_results_mult_part"
)

# False: Skip destination files that already exist.
# True:  Replace destination files that already exist.
OVERWRITE = False


# Maps the original filename ending to the new filename inside the folder.
FILE_NAME_MAPPING = {
    "params.json": "params.json",
    "summary_agg_dataset_id_param.csv": "agg_dataset_id_param.csv",
    "summary_agg_param.csv": "agg_param.csv",
    "summary_ranked_param_overview.csv": "ranked_param_overview.csv",
    "summary_rank_scored.csv": "rank_scored.csv",
}

# Matches names such as:
# proposal_optm_1_1_params.json
# proposal_optm_12_3_summary_rank_scored.csv
FILE_PATTERN = re.compile(
    r"^proposal_optm_(?P<experiment>\d+)_(?P<sub_experiment>\d+)_(?P<ending>.+)$"
)


def organize_proposal_files(
    file_dir: Path,
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    Copy recognized proposal optimization files into experiment folders.

    Each folder is named:
        proposal_optm_<experiment>_<sub_experiment>

    Existing destination files are skipped unless overwrite=True.

    Parameters
    ----------
    file_dir:
        Directory containing the source files.
    overwrite:
        Whether existing destination files may be replaced.

    Returns
    -------
    Dict[str, int]
        Processing statistics.
    """
    file_dir = Path(file_dir).expanduser().resolve()

    if not file_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {file_dir}")

    if not file_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {file_dir}")

    statistics = {
        "folders_created": 0,
        "files_copied": 0,
        "files_overwritten": 0,
        "files_skipped": 0,
        "files_ignored": 0,
    }

    # Only inspect files directly inside file_dir. Created subfolders are not
    # searched recursively.
    for source_path in sorted(file_dir.iterdir()):
        if not source_path.is_file():
            continue

        match = FILE_PATTERN.fullmatch(source_path.name)
        if match is None:
            statistics["files_ignored"] += 1
            print(f"IGNORED: {source_path.name}")
            continue

        ending = match.group("ending")
        destination_name = FILE_NAME_MAPPING.get(ending)

        if destination_name is None:
            statistics["files_ignored"] += 1
            print(f"IGNORED: Unknown file type: {source_path.name}")
            continue

        experiment = match.group("experiment")
        sub_experiment = match.group("sub_experiment")
        folder_name = f"proposal_optm_{experiment}_{sub_experiment}"
        destination_dir = file_dir / folder_name

        if not destination_dir.exists():
            destination_dir.mkdir(parents=True)
            statistics["folders_created"] += 1
            print(f"CREATED FOLDER: {destination_dir.name}")

        destination_path = destination_dir / destination_name
        destination_existed = destination_path.exists()

        if destination_existed and not overwrite:
            statistics["files_skipped"] += 1
            print(
                f"SKIPPED: {source_path.name} -> "
                f"{folder_name}/{destination_name} already exists"
            )
            continue

        shutil.copy2(source_path, destination_path)

        if destination_existed:
            statistics["files_overwritten"] += 1
            print(
                f"OVERWRITTEN: {source_path.name} -> "
                f"{folder_name}/{destination_name}"
            )
        else:
            statistics["files_copied"] += 1
            print(
                f"COPIED: {source_path.name} -> "
                f"{folder_name}/{destination_name}"
            )

    return statistics


def print_summary(statistics: Dict[str, int]) -> None:
    """Print a compact processing summary."""
    print("\nFinished.")
    print(f"Folders created:   {statistics['folders_created']}")
    print(f"Files copied:      {statistics['files_copied']}")
    print(f"Files overwritten: {statistics['files_overwritten']}")
    print(f"Files skipped:     {statistics['files_skipped']}")
    print(f"Files ignored:     {statistics['files_ignored']}")


def main() -> None:
    """Run the organizer using the global configuration."""
    statistics = organize_proposal_files(
        file_dir=FILE_DIR,
        overwrite=OVERWRITE,
    )
    print_summary(statistics)


if __name__ == "__main__":
    main()
