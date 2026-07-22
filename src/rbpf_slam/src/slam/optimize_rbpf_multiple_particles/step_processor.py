from dataclasses import fields, is_dataclass
from numbers import Real
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .evaluator_mult_part import StepResult
from .optimizer import RankedRun

# Constants for pose components indexing
X = 0
Y = 1
THETA = 2


class StepProcessor:

    @staticmethod
    def _read_from_summary(run: RankedRun, key: str, default=None):
        '''
        Safely read a value from run.summary. Raises error if run.summary is not a dict or if the key is not present. 
        '''
        if not isinstance(run.summary, dict):
            raise ValueError(f"\nExpected run.summary to be a dict, got {type(run.summary)}")

        if key not in run.summary.keys():
            raise KeyError(f"\nKey '{key}' not found in run.summary. Available keys: {list(run.summary.keys())}")

        value = run.summary.get(key, default)
            
        return value


    @staticmethod
    def _is_pose(name: str):
        '''
        Returns true if the name indicates a pose field, else false. 
        '''
        name_lowered = name.lower()
        return (
            "pose" in name_lowered
            and "poses" not in name_lowered
        )


    @staticmethod
    def _is_angle(name: str):
        '''
        Returns True if the name indicates an angle field, else False.
        '''
        name_lowered = name.lower()
        return (
            "theta" in name_lowered
            or name_lowered.startswith("rot_")
            or "_rot_" in name_lowered
            or "angle" in name_lowered
        )
        

    @staticmethod
    def _split_pose(name: str, value: np.ndarray, pose_appendix: Iterable[str]):
        '''
        Get's the pose name, value and pose_appendix and returns a dict with the splitted pose values.
        The pose_appendix should be a list of 3 strings, e.g. ["x", "y", "theta"].
        
        Parameters
        ----------
        name : str
            The name of the pose field.
        value : np.ndarray
            The pose value as a numpy array of shape (3,).
        pose_appendix : Iterable[str]
            The appendix for the pose values, e.g. ["x", "y", "theta"].
        
        Returns
        -------
        dict
            A dictionary with the splitted pose values, e.g. {"pose_x": value[0], "pose_y": value[1], "pose_theta": value[2]}.
        '''
        if len(pose_appendix) != 3:
            raise ValueError(
                f"pose_appendix must have exactly 3 elements, got {len(pose_appendix)}."
            )

        x_col = name + "_" + pose_appendix[X]
        y_col = name + "_" + pose_appendix[Y]
        theta_col = name + "_" + pose_appendix[THETA]

        if value is None or not isinstance(value, np.ndarray) or value.shape != (3,):
            splitted_pose = {
                x_col: None,
                y_col: None,
                theta_col: None,
            }
        else:
            splitted_pose = {
                x_col: value[X],
                y_col: value[Y],
                theta_col: np.rad2deg(value[THETA]),
            }

        return splitted_pose


    @staticmethod
    def _is_valid_traj(
        map_traj: Union[np.ndarray, Sequence[Sequence[float]], None],
        step_results: Sequence[StepResult]
    ):
        map_traj = np.asarray(map_traj) if map_traj is not None else None
        
        map_traj_is_valid = (
                map_traj is not None
                and map_traj.ndim == 2
                and map_traj.shape[1] == 3
                and len(step_results) == map_traj.shape[0]
            )
        return map_traj_is_valid
    

    @staticmethod
    def _is_valid_err(
        errs: Union[np.ndarray, Sequence[float], None],
        step_results: Sequence[StepResult]
    ):        
        errs = np.asarray(errs) if errs is not None else None

        err_is_valid = (
                errs is not None
                and errs.ndim == 1                                              
                and len(step_results) == errs.shape[0]
            )
        return err_is_valid

    
    @staticmethod
    def process_ranked_runs(ranked_runs: Iterable[RankedRun], pose_appendix=("x", "y", "theta")):
        '''
        Get's the ranked runs and the pose appendix structure and returns a pandas DataFrame including all step results
        ordered from best to worst run.  
        '''
        # Store each step in one row
        rows = []

        for rank, run in enumerate(ranked_runs, start=1):        
            # Extract and validate map traj
            map_traj = StepProcessor._read_from_summary(run, "map_traj", default=None)
            map_traj_arr = np.asarray(map_traj)
            map_traj_is_valid = StepProcessor._is_valid_traj(map_traj, run.step_results)

            # Extract and validate map traj errors
            # Trans
            trans_errs_map_traj = StepProcessor._read_from_summary(run, "trans_errs_map_traj", default=None)
            trans_errs_map_traj_arr = np.asarray(trans_errs_map_traj)
            trans_errs_map_traj_is_valid = StepProcessor._is_valid_err(trans_errs_map_traj, run.step_results)
            # rot
            rot_errs_map_traj = StepProcessor._read_from_summary(run, "rot_errs_map_traj", default=None)
            rot_errs_map_traj_arr = np.asarray(rot_errs_map_traj)
            rot_errs_map_traj_is_valid = StepProcessor._is_valid_err(rot_errs_map_traj, run.step_results)

            # Loop through all steps            
            for step_idx, step in enumerate(run.step_results):
                # Validate step type
                if not isinstance(step, StepResult):
                    raise TypeError(
                        f"Step {step_idx} of run with rank {rank} "
                        f"is not a valid StepResult instance."
                    )

                # Add general run info to row
                row = {
                    "rank": rank,
                    "score": run.score,
                    "dataset_id": run.dataset_id,
                    "map_name": run.map_name,
                    "seed": run.seed,
                    "parameter_tag": run.parameter_tag,
                    "parameter_hash": run.parameter_hash
                }

                # Add map trajectory to row if valid
                map_pose = map_traj_arr[step_idx] if map_traj_is_valid else None
                row.update(
                    StepProcessor._split_pose(
                        name="map_traj",
                        value=map_pose,
                        pose_appendix=pose_appendix,
                    )
                )

                # Add map trajectory errors to row if valid
                trans_err_map_pose = trans_errs_map_traj_arr[step_idx] if trans_errs_map_traj_is_valid else None
                rot_err_map_pose = rot_errs_map_traj_arr[step_idx] if rot_errs_map_traj_is_valid else None
                row.update({
                    "trans_err_map_traj": trans_err_map_pose,
                    "rot_err_map_traj": np.rad2deg(rot_err_map_pose),
                })

                # Process step fields 
                for field in fields(step):
                    name = field.name
                    value = getattr(step, name)

                    # Transform all pose obj
                    if StepProcessor._is_pose(name):                        
                        row.update(
                            StepProcessor._split_pose(
                                name=name,
                                value=value,
                                pose_appendix=pose_appendix,
                            )
                        ) 
                    else:
                        # Add all others
                        row.update({name: value})
            
                rows.append(row)

        return pd.DataFrame(rows)

