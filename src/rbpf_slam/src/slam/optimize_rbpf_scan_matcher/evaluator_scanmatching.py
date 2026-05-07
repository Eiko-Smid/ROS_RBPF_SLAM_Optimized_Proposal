from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import math
import numpy as np

from ..optimize_rbpf.playback_defs import ExperimentParams

Pose2D = Tuple[float, float, float]


@dataclass
class StepResultScanMatching:
    '''
    Dataclass for storing the results of one step of the rbpf variant with scan matching only. 
    '''
    step_idx: int
    t: float
    true_pose: Pose2D
    pred_pose: Optional[Pose2D]
    corr_pose: Optional[Pose2D]
    # est_pose: Optional[Pose2D]
    scan_match_failed: bool
    translation_error: Optional[float]
    rotation_error: Optional[float]
    pred_translation_error: Optional[float]
    corr_translation_error: Optional[float]
    pred_rotation_error: Optional[float]
    corr_rotation_error: Optional[float]
    icp_best_trans_param: Optional[float]
    icp_best_rot_abs_deg: Optional[float]
    pred_to_corr_trans_err: Optional[float]
    pred_to_corr_rot_err: Optional[float]
    icp_iterations: Optional[int]
    icp_mean_error: Optional[float]
    n_correspondences: Optional[int]
    use_transformation: Optional[bool]
    stop_reason: Optional[str]
    n_measurements_total: Optional[int]
    n_valid_measurements_filter: Optional[int]
    n_valid_measurements_map_update: Optional[int]
    n_map_points_extracted: Optional[int]
    t_ogm: Optional[float]
    t_scan_matching: Optional[float]
    t_prediction: Optional[float]
    t_map_extraction: Optional[float]
    t_correct_pose: Optional[float]
    step_duration: Optional[float]
    timing_update_particle: Optional[float]


@dataclass
class RunResultScanMatching:
    '''
    Dataclass for storing the results of an entire run of the rbpf variant with scan matching only.
    '''
    params: ExperimentParams
    step_results: List[StepResultScanMatching] = field(default_factory=list)
    summary: Optional["RunSummaryScanMatching"] = None


@dataclass
class RunSummaryScanMatching:
    n_steps: int
    n_particles: int
    scan_match_failed_count: int
    icp_success_rate: float
    scan_match_success_rate: float
    mean_pred_trans_error: float
    mean_pred_rot_error: float
    mean_corr_trans_error: float
    mean_corr_rot_error: float
    rmse_pred_trans_error: float
    rmse_pred_rot_error: float
    rmse_corr_trans_error: float
    rmse_corr_rot_error: float
    final_drift: float
    mean_icp_iterations: float
    mean_icp_error: float
    mean_best_trans_norm: float
    max_best_trans_norm: float
    mean_best_rot_abs: float
    max_best_rot_abs: float
    mean_translation_error: float
    mean_rotation_error: float
    rmse_translation_error: float
    rmse_rotation_error: float
    mean_step_duration: float
    mean_timing_sm_update_particle_s: float
    mean_timing_sm_scan_match_update_pose_s: float = 0.0
    mean_timing_sm_map_extension_s: float = 0.0
    mean_timing_sm_map_update_s: float = 0.0
    timing_sm_update_particle_count: int = 0
    timing_sm_scan_match_update_pose_count: int = 0
    timing_sm_map_extension_count: int = 0
    timing_sm_map_update_count: int = 0
    count_too_few_points: int = 0
    count_too_few_corresp: int = 0
    infinite_h_or_g: int = 0
    ill_cond_H: int = 0
    infinite_dtransform: int = 0
    infinite_mean_err: int = 0
    best_transf_too_large: int = 0
    best_mean_err_too_large: int = 0
    icp_total_runs: int = 0
    icp_success_count: int = 0
    icp_failed_count: int = 0


class ScanMatchingEvaluator:
    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        '''
        Computes the difference between the two given angles a and b and binds the result to the range [-pi, pi].
        '''
        return math.atan2(math.sin(a - b), math.cos(a - b))


    @staticmethod
    def translation_error(p1: Pose2D, p2: Pose2D) -> float:
        '''
        Computes the translational error between two given poses p1 and p2.
        '''
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


    @staticmethod
    def _to_pose_tuple(pose: Any) -> Optional[Pose2D]:
        '''
        Converts the given pose to a tuple of (x, y, theta) if possible. Supports various input formats.
        '''
        if pose is None:
            return None

        if isinstance(pose, (tuple, list, np.ndarray)) and len(pose) >= 3:
            return (float(pose[0]), float(pose[1]), float(pose[2]))

        if hasattr(pose, "x") and hasattr(pose, "y") and hasattr(pose, "theta"):
            return (float(pose.x), float(pose.y), float(pose.theta))

        raise TypeError(f"Unsupported pose format: {type(pose)}")


    def evaluate_step(
        self,
        step_idx: int,
        t: float,
        true_pose: Any,
        pred_pose: Any,
        corr_pose: Any,
        best_transformation: Any,
        icp_iterations: Optional[int],
        icp_mean_error: Optional[float],
        n_correspondences: Optional[int],
        use_transformation: Optional[bool],
        stop_reason: Optional[str],
        n_measurements_total: Optional[int],
        n_valid_measurements_filter: Optional[int],
        n_valid_measurements_map_update: Optional[int],
        n_map_points_extracted: Optional[int],
        t_ogm: Optional[float],
        t_scan_matching: Optional[float],
        t_prediction: Optional[float],
        t_map_extraction: Optional[float],
        t_correct_pose: Optional[float],
        scan_match_failed: bool,
        step_duration: Optional[float],
        timing_update_particle: Optional[float],
    ) -> StepResultScanMatching:
        # Converts the given poses to a Pose2D tuple format
        true_pose_t = self._to_pose_tuple(true_pose)
        pred_pose_t = self._to_pose_tuple(pred_pose)
        corr_pose_t = self._to_pose_tuple(corr_pose)
        # est_pose_t = corr_pose_t

        # Init metrics
        trans_err = None
        rot_err = None
        pred_trans_err = None
        corr_trans_err = None
        pred_rot_err = None
        corr_rot_err = None
        best_trans_norm = None
        best_rot_abs = None
        pred_to_corr_trans_err = None
        pred_to_corr_rot_err = None

        # Computes the translation and rotation errors between the predicted and corrected pose
        if pred_pose_t is not None and true_pose_t is not None:
            pred_trans_err = self.translation_error(pred_pose_t, true_pose_t)
            pred_rot_err = abs(self.angle_diff(pred_pose_t[2], true_pose_t[2]))

        # Computes the translation and rotation errors between the corrected and true pose
        if corr_pose_t is not None and true_pose_t is not None:
            corr_trans_err = self.translation_error(corr_pose_t, true_pose_t)
            corr_rot_err = abs(self.angle_diff(corr_pose_t[2], true_pose_t[2]))

            # Keep existing summary/scorer behavior based on corrected pose error.
            trans_err = corr_trans_err
            rot_err = corr_rot_err

        # Computes the translational and rotation errors between the predicted and corrected pose
        if pred_pose_t is not None and corr_pose_t is not None:
            pred_to_corr_trans_err = self.translation_error(corr_pose_t, pred_pose_t)
            pred_to_corr_rot_err = abs(self.angle_diff(corr_pose_t[2], pred_pose_t[2]))

        if best_transformation is not None:
            tf = np.asarray(best_transformation, dtype=float).reshape(-1)
            if tf.size >= 3 and np.all(np.isfinite(tf[:3])):
                best_trans_norm = float(np.linalg.norm(tf[:2]))
                best_rot_abs = float(abs(tf[2]))

        # If scan matching failed before ICP started, these diagnostics are not meaningful.
        pre_icp_failure = bool(scan_match_failed) and (str(stop_reason) == "scan matcher failed before icp")
        if pre_icp_failure:
            n_map_points_extracted = None
            best_trans_norm = None
            best_rot_abs = None

        return StepResultScanMatching(
            step_idx=int(step_idx),
            t=float(t),
            true_pose=true_pose_t,
            pred_pose=pred_pose_t,
            corr_pose=corr_pose_t,
            # est_pose=est_pose_t,
            scan_match_failed=bool(scan_match_failed),
            translation_error=trans_err,
            rotation_error=rot_err,
            pred_translation_error=pred_trans_err,
            corr_translation_error=corr_trans_err,
            pred_rotation_error=pred_rot_err,
            corr_rotation_error=corr_rot_err,
            icp_best_trans_param=best_trans_norm,
            icp_best_rot_abs_deg=best_rot_abs,
            pred_to_corr_trans_err=pred_to_corr_trans_err,
            pred_to_corr_rot_err=pred_to_corr_rot_err,
            icp_iterations=int(icp_iterations) if icp_iterations is not None else None,
            icp_mean_error=float(icp_mean_error) if icp_mean_error is not None else None,
            n_correspondences=int(n_correspondences) if n_correspondences is not None else None,
            use_transformation=bool(use_transformation) if use_transformation is not None else None,
            stop_reason=str(stop_reason) if stop_reason is not None else None,
            n_measurements_total=int(n_measurements_total) if n_measurements_total is not None else None,
            n_valid_measurements_filter=(
                int(n_valid_measurements_filter) if n_valid_measurements_filter is not None else None
            ),
            n_valid_measurements_map_update=(
                int(n_valid_measurements_map_update) if n_valid_measurements_map_update is not None else None
            ),
            n_map_points_extracted=int(n_map_points_extracted) if n_map_points_extracted is not None else None,
            t_ogm=float(t_ogm) if t_ogm is not None else None,
            t_scan_matching=float(t_scan_matching) if t_scan_matching is not None else None,
            t_prediction=float(t_prediction) if t_prediction is not None else None,
            t_map_extraction=float(t_map_extraction) if t_map_extraction is not None else None,
            t_correct_pose=float(t_correct_pose) if t_correct_pose is not None else None,
            step_duration=float(step_duration) if step_duration is not None else None,
            timing_update_particle=float(timing_update_particle) if timing_update_particle is not None else None,
        )

    def summarize_run(self, step_results: List[StepResultScanMatching], params: ExperimentParams) -> RunSummaryScanMatching:
        trans_err = [s.translation_error for s in step_results if s.translation_error is not None]
        rot_err = [s.rotation_error for s in step_results if s.rotation_error is not None]
        pred_trans_err = [s.pred_translation_error for s in step_results if s.pred_translation_error is not None]
        pred_rot_err = [s.pred_rotation_error for s in step_results if s.pred_rotation_error is not None]
        corr_trans_err = [s.corr_translation_error for s in step_results if s.corr_translation_error is not None]
        corr_rot_err = [s.corr_rotation_error for s in step_results if s.corr_rotation_error is not None]
        step_durations = [s.step_duration for s in step_results if s.step_duration is not None]
        update_particle_timings = [
            s.timing_update_particle for s in step_results if s.timing_update_particle is not None
        ]

        final_drift = float("inf")
        if step_results:
            last_step = step_results[-1]
            if last_step.corr_pose is not None and last_step.true_pose is not None:
                final_drift = self.translation_error(last_step.corr_pose, last_step.true_pose)

        use_transformation_vals = [
            s.use_transformation for s in step_results if s.use_transformation is not None
        ]
        icp_success_rate = (
            float(sum(1 for v in use_transformation_vals if v)) / float(len(use_transformation_vals))
            if use_transformation_vals
            else 0.0
        )

        scan_match_failed_count = int(sum(1 for s in step_results if s.scan_match_failed))
        
        n_steps = len(step_results)

        if scan_match_failed_count:
            scan_match_success_rate =  1.0 - (scan_match_failed_count / n_steps)
        else:
            scan_match_success_rate = 1.0

        
        icp_iterations_per_step = [
            float(s.icp_iterations) if s.icp_iterations is not None else 0.0
            for s in step_results
        ]
        mean_icp_iterations = float(np.mean(icp_iterations_per_step)) if n_steps > 0 else 0.0

        successful_steps = [s for s in step_results if s.use_transformation is True]
        successful_icp_errors = [s.icp_mean_error for s in successful_steps if s.icp_mean_error is not None]
        successful_best_trans_norm = [s.icp_best_trans_param for s in successful_steps if s.icp_best_trans_param is not None]
        successful_best_rot_abs = [s.icp_best_rot_abs_deg for s in successful_steps if s.icp_best_rot_abs_deg is not None]

        mean_icp_error = float(np.mean(successful_icp_errors)) if successful_icp_errors else float("inf")
        mean_best_trans_norm = (
            float(np.mean(successful_best_trans_norm)) if successful_best_trans_norm else float("inf")
        )
        max_best_trans_norm = (
            float(np.max(successful_best_trans_norm)) if successful_best_trans_norm else float("inf")
        )
        mean_best_rot_abs = (
            float(np.mean(successful_best_rot_abs)) if successful_best_rot_abs else float("inf")
        )
        max_best_rot_abs = (
            float(np.max(successful_best_rot_abs)) if successful_best_rot_abs else float("inf")
        )

        return RunSummaryScanMatching(
            n_steps=n_steps,
            n_particles=int(params.particle_params.n_particles),
            scan_match_failed_count=scan_match_failed_count,
            icp_success_rate=icp_success_rate,
            scan_match_success_rate=scan_match_success_rate,
            mean_pred_trans_error=float(np.mean(pred_trans_err)) if pred_trans_err else float("inf"),
            mean_pred_rot_error=float(np.mean(pred_rot_err)) if pred_rot_err else float("inf"),
            mean_corr_trans_error=float(np.mean(corr_trans_err)) if corr_trans_err else float("inf"),
            mean_corr_rot_error=float(np.mean(corr_rot_err)) if corr_rot_err else float("inf"),
            rmse_pred_trans_error=float(np.sqrt(np.mean(np.square(pred_trans_err)))) if pred_trans_err else float("inf"),
            rmse_pred_rot_error=float(np.sqrt(np.mean(np.square(pred_rot_err)))) if pred_rot_err else float("inf"),
            rmse_corr_trans_error=float(np.sqrt(np.mean(np.square(corr_trans_err)))) if corr_trans_err else float("inf"),
            rmse_corr_rot_error=float(np.sqrt(np.mean(np.square(corr_rot_err)))) if corr_rot_err else float("inf"),
            final_drift=final_drift,
            mean_icp_iterations=mean_icp_iterations,
            mean_icp_error=mean_icp_error,
            mean_best_trans_norm=mean_best_trans_norm,
            max_best_trans_norm=max_best_trans_norm,
            mean_best_rot_abs=mean_best_rot_abs,
            max_best_rot_abs=max_best_rot_abs,
            mean_translation_error=float(np.mean(trans_err)) if trans_err else float("inf"),
            mean_rotation_error=float(np.mean(rot_err)) if rot_err else float("inf"),
            rmse_translation_error=float(np.sqrt(np.mean(np.square(trans_err)))) if trans_err else float("inf"),
            rmse_rotation_error=float(np.sqrt(np.mean(np.square(rot_err)))) if rot_err else float("inf"),
            mean_step_duration=float(np.mean(step_durations)) if step_durations else 0.0,
            mean_timing_sm_update_particle_s=float(np.mean(update_particle_timings)) if update_particle_timings else 0.0,
        )
