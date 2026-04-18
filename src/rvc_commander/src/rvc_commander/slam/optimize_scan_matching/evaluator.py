import math
import numpy as np

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from ..scan_match_playback_def import ExperimentParams

Pose2D = Tuple[float, float, float]


@dataclass
class StepResult:
    '''
    Storage class that stores the results of one scan matcher run.
    '''
    step_idx: int
    t: float
    true_pose: Pose2D
    pred_pose: Optional[Pose2D]
    corr_pose: Optional[Pose2D]
    used_fallback_prediction: bool
    translation_error_pred: Optional[float]
    rotation_error_pred: Optional[float]
    translation_error_corr: Optional[float]
    rotation_error_corr: Optional[float]
    icp_info: dict = field(default_factory=dict)
    step_duration: Optional[float] = None


@dataclass
class RunResult:
    '''
    Storage class for the results of a single run of the scan matcher.
    '''
    params: ExperimentParams
    step_results: List[StepResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)



class ScanMatcherEvaluator:
    '''
    Evaluation class for evaluating the results for one parameter set run of the scan matcher.
    '''
    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        '''
        Computes the difference between two angles a and b. Ensures that the result is always between -pi and pi.
        '''
        return math.atan2(math.sin(a - b), math.cos(a - b))


    @staticmethod
    def translation_error(p1, p2) -> float:
        '''
        Computes the translation error between two poses p1 and p2. Only considers the x and y components of the poses.
        '''
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


    def evaluate_step(
        self,
        step_idx: int,
        t: float,
        true_pose: Pose2D,
        pred_pose: Optional[Pose2D],
        corr_pose: Optional[Pose2D],
        icp_info: dict,

        used_fallback_prediction: bool,
        step_duration: Optional[float],
    ) -> StepResult:
        '''
        Computes the translational and rotational error for the predicted and corrected poses, given the true pose. Returns a
        StepResult object that contains the results of this evaluation step, including the errors and the ICP info. 

        Parameters
        ----------
        step_idx: int
            index of the current step in the run
        t: float
            timestamp of the current step
        true_pose: Pose2D
            the true pose at the current step
        pred_pose: Optional[Pose2D]
            the predicted pose at the current step
        corr_pose: Optional[Pose2D]
            the corrected pose at the current step
        icp_info: dict
            information from the ICP algorithm for the current step
        used_fallback_prediction: bool
            whether a fallback prediction was used instead of a corrected pose

        Returns
        -------
        StepResult
            an object containing the results of the evaluation for this step, including the errors and ICP info
        '''
        # Init error values  
        pred_trans_err = None
        pred_rot_err = None
        corr_trans_err = None
        corr_rot_err = None

        # Compute translation and rotational error for predicted pose
        if pred_pose is not None:
            pred_trans_err = self.translation_error(pred_pose, true_pose)
            pred_rot_err = abs(self.angle_diff(pred_pose[2], true_pose[2]))

        # Compute translation and rotational error for corrected pose
        if corr_pose is not None:
            corr_trans_err = self.translation_error(corr_pose, true_pose)
            corr_rot_err = abs(self.angle_diff(corr_pose[2], true_pose[2]))

        return StepResult(
            step_idx=step_idx,
            t=t,
            true_pose=true_pose,
            pred_pose=pred_pose,
            corr_pose=corr_pose,
            used_fallback_prediction=used_fallback_prediction,
            translation_error_pred=pred_trans_err,
            rotation_error_pred=pred_rot_err,
            translation_error_corr=corr_trans_err,
            rotation_error_corr=corr_rot_err,
            icp_info=icp_info.copy() if icp_info else {},
            step_duration=step_duration
        )


    def summarize_run(self, step_results: List[StepResult]) -> dict:
        '''
        Get's teh results from all steps that have been processed by the scan matcher. Computes summary statistics for
        the entire run, such as mean translation and rotation error, number of fallback predictions used, and mean ICP
        iterations. Returns a dictionary containing these summary statistics. 

        Parameters
        ----------
        step_results: List[StepResult]
            a list of StepResult objects containing the evaluation results for each step

        Returns
        -------
        dict
            a dictionary containing the summary statistics for the entire run consisting of:
            - n_steps: total number of steps in the run
            - mean_translation_error: mean translation error across all steps (only non None values, only considering corrected poses)
            - rmse_translation_error: root mean squared translation error across all steps (only non None values, only considering corrected poses)
            - mean_rotation_error: mean rotation error across all steps (only considering corrected poses)
            - fallback_count: total number of steps where a fallback prediction was used instead of a corrected pose
            - mean_icp_iterations: mean number of ICP iterations across all steps (only considering steps where ICP info is available)
            - mean_step_duration: mean duration of each step in the run
        '''
        # Clean the translation and rotation errors by filtering out None values
        cleaned_trans_err = [s.translation_error_corr for s in step_results if s.translation_error_corr is not None]
        cleaned_rot_err = [s.rotation_error_corr for s in step_results if s.rotation_error_corr is not None]
        fallback_count = sum(s.used_fallback_prediction for s in step_results)
        step_durations = [s.step_duration for s in step_results if s.step_duration is not None]

        # Extract the number of ICP iterations
        icp_iterations = [
            s.icp_info.get("iteration")
            for s in step_results
            if s.icp_info and s.icp_info.get("iteration") is not None
        ]

        return {
            "n_steps": len(step_results),
            "mean_translation_error": float(np.mean(cleaned_trans_err)) if cleaned_trans_err else float("inf"),
            "rmse_translation_error": float(np.sqrt(np.mean(np.square(cleaned_trans_err)))) if cleaned_trans_err else float("inf"),
            "mean_rotation_error": float(np.mean(cleaned_rot_err)) if cleaned_rot_err else float("inf"),
            "fallback_count": int(fallback_count),
            "mean_icp_iterations": float(np.mean(icp_iterations)) if icp_iterations else float("inf"),
            "mean_step_duration": float(np.mean(step_durations)) if step_durations else 0.0
        }