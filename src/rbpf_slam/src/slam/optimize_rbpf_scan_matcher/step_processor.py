from dataclasses import fields
from numbers import Real
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from .evaluator_scanmatching import StepResultScanMatching
from .optimizer_scanmatching import RankedRunScanMatching


# Constants for pose component indexing
X = 0
Y = 1
THETA = 2


class StepProcessor:
    """
    Convert the stored step results of ranked scan-matching runs into one flat
    DataFrame row per step.

    Coordinate-pose fields are split into scalar columns. All other step fields
    are preserved under their native names, with additional degree and
    millisecond aliases for the step-trace CSV.
    """

    _RADIAN_TO_DEGREE_ALIASES = {
        "rot_err": "rot_err_deg",
        "pred_rot_err": "pred_rot_err_deg",
        "raw_odom_rot_err": "raw_odom_rot_err_deg",
        "corr_rot_err": "corr_rot_err_deg",
        "icp_best_rot_abs_rad": "icp_best_rot_abs_deg",
        "pred_to_corr_rot_err": "pred_to_corr_rot_err_deg",
    }

    _SECOND_TO_MILLISECOND_ALIASES = {
        "t_ogm": "t_ogm_ms",
        "t_scan_matching": "t_scan_matching_ms",
        "t_prediction": "t_prediction_ms",
        "t_map_extraction": "t_map_extraction_ms",
        "t_correct_pose": "t_correct_pose_ms",
        "step_duration": "step_duration_ms",
        "timing_update_particle": "timing_update_particle_ms",
    }


    @staticmethod
    def _is_pose(name: str) -> bool:
        """Return True only for coordinate-pose fields, not pose timing fields."""
        name_lowered = name.lower()
        return (
            name_lowered.endswith("_pose")
            and not name_lowered.startswith("t_")
            and not name_lowered.startswith("timing_")
        )


    @staticmethod
    def _split_pose(
        name: str,
        value,
        pose_appendix: Sequence[str],
    ) -> Dict[str, object]:
        """
        Split a pose tuple/list/array and convert its heading to degrees.

        Missing, malformed, or non-finite poses produce three None values.
        """
        if len(pose_appendix) != 3:
            raise ValueError(
                f"pose_appendix must have exactly 3 elements, got {len(pose_appendix)}."
            )
        if not pose_appendix[THETA].endswith("_deg"):
            raise ValueError(
                "The pose angle column appendix must end with '_deg', "
                f"got {pose_appendix[THETA]!r}."
            )

        pose_columns = {
            name + "_" + pose_appendix[X]: None,
            name + "_" + pose_appendix[Y]: None,
            name + "_" + pose_appendix[THETA]: None,
        }

        if value is None:
            return pose_columns

        try:
            pose = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return pose_columns

        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            return pose_columns

        pose_columns.update(
            {
                name + "_" + pose_appendix[X]: pose[X],
                name + "_" + pose_appendix[Y]: pose[Y],
                name + "_" + pose_appendix[THETA]: float(np.rad2deg(pose[THETA])),
            }
        )
        return pose_columns


    @staticmethod
    def _optional_deg(value_rad):
        if value_rad is None or not isinstance(value_rad, Real):
            return None
        return float(np.rad2deg(value_rad))


    @staticmethod
    def _optional_ms(value_s):
        if value_s is None or not isinstance(value_s, Real):
            return None
        return float(value_s) * 1000.0


    @staticmethod
    def process_ranked_runs(
        ranked_runs: Iterable[RankedRunScanMatching],
        pose_appendix: Sequence[str] = ("x", "y", "theta_deg"),
    ) -> pd.DataFrame:
        """Return all stored scan-matching run steps as a flat DataFrame."""
        rows = []

        for rank, run in enumerate(ranked_runs, start=1):
            for step_idx, step in enumerate(run.step_results):
                if not isinstance(step, StepResultScanMatching):
                    raise TypeError(
                        f"Step {step_idx} of run with rank {rank} "
                        "is not a valid StepResultScanMatching instance."
                    )

                row = {
                    "rank": rank,
                    "score": run.score,
                    "dataset_id": run.dataset_id,
                    "map_name": run.map_name,
                    "seed": run.seed,
                    "parameter_tag": run.parameter_tag,
                    "parameter_hash": run.parameter_hash,
                }

                for field in fields(step):
                    name = field.name
                    value = getattr(step, name)

                    if StepProcessor._is_pose(name):
                        row.update(
                            StepProcessor._split_pose(
                                name=name,
                                value=value,
                                pose_appendix=pose_appendix,
                            )
                        )
                        continue

                    row[name] = value

                    degree_alias = StepProcessor._RADIAN_TO_DEGREE_ALIASES.get(name)
                    if degree_alias is not None:
                        row[degree_alias] = StepProcessor._optional_deg(value)

                    millisecond_alias = StepProcessor._SECOND_TO_MILLISECOND_ALIASES.get(name)
                    if millisecond_alias is not None:
                        row[millisecond_alias] = StepProcessor._optional_ms(value)

                rows.append(row)

        return pd.DataFrame(rows)
