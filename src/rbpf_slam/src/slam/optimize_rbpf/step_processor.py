from dataclasses import fields
from numbers import Real
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from .evaluator import StepResult
from .optimizer import RankedRun

# Constants for pose component indexing
X = 0
Y = 1
THETA = 2


class StepProcessor:
    """
    Convert the stored step results of ranked runs into one flat DataFrame row
    per step.

    Pose fields are split into scalar columns. All other StepResult fields are
    copied to the row, with additional aliases for the existing step-trace CSV
    column names and units.
    """

    _FIELD_ALIASES = {
        "step_idx": "step_id",
        "trans_err_raw_odom": "trans_error_raw_odom",
        "translation_error": "trans_error",
    }

    _RADIAN_TO_DEGREE_ALIASES = {
        "rotation_error": "rot_error_deg",
        "rot_err_raw_odom": "rot_error_raw_odom_deg",
        "rotation_error_best_p": "rot_error_best_p_deg",
        "rot_err_sm_true": "rot_err_sm_true_deg",
        "rot_err_mu_true": "rot_err_mu_true_deg",
        "rot_err_mu_sm": "rot_err_mu_sm_deg",
        "rot_err_mu_pred": "rot_err_mu_pred_deg",
        "prop_std_theta": "prop_std_theta_deg",
    }

    @staticmethod
    def _is_pose(name: str) -> bool:
        """
        Return True for coordinate-pose fields, excluding pose-error metrics.
        """
        return name.lower().endswith("_pose")

    @staticmethod
    def _split_pose(
        name: str,
        value,
        pose_appendix: Sequence[str],
    ) -> Dict[str, object]:
        """
        Split a pose tuple/list/array into three scalar columns.
        """
        if len(pose_appendix) != 3:
            raise ValueError(
                f"pose_appendix must have exactly 3 elements, got {len(pose_appendix)}."
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
                name + "_" + pose_appendix[THETA]: pose[THETA],
            }
        )
        return pose_columns

    @staticmethod
    def _optional_ms(value_s):
        return value_s * 1000.0 if value_s is not None else None

    @staticmethod
    def _optional_deg(value_rad):
        if value_rad is None:
            return None
        if not isinstance(value_rad, Real):
            return None
        return float(np.rad2deg(value_rad))

    @staticmethod
    def process_ranked_runs(
        ranked_runs: Iterable[RankedRun],
        pose_appendix: Sequence[str] = ("x", "y", "theta_rad"),
    ) -> pd.DataFrame:
        """
        Return all stored run steps as a flat DataFrame.
        """
        rows = []

        for rank, run in enumerate(ranked_runs, start=1):
            for step_idx, step in enumerate(run.step_results):
                if not isinstance(step, StepResult):
                    raise TypeError(
                        f"Step {step_idx} of run with rank {rank} "
                        "is not a valid StepResult instance."
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

                    # Preserve every non-pose StepResult field under its native name.
                    row[name] = value

                    alias = StepProcessor._FIELD_ALIASES.get(name)
                    if alias is not None:
                        row[alias] = value

                    degree_alias = StepProcessor._RADIAN_TO_DEGREE_ALIASES.get(name)
                    if degree_alias is not None:
                        row[degree_alias] = StepProcessor._optional_deg(value)

                    if name == "step_duration":
                        row["step_duration_ms"] = StepProcessor._optional_ms(value)

                rows.append(row)

        return pd.DataFrame(rows)
