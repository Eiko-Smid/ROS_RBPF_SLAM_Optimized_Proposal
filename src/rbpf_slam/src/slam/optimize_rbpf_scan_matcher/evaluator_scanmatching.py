from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

from ..optimize_rbpf.playback_defs import ExperimentParams

Pose2D = Tuple[float, float, float]


@dataclass
class StepResultScanMatching:
    step_idx: int
    t: float
    true_pose: Pose2D
    pred_pose: Optional[Pose2D]
    corr_pose: Optional[Pose2D]
    est_pose: Optional[Pose2D]
    scan_match_failed: bool
    translation_error: Optional[float]
    rotation_error: Optional[float]
    pred_translation_error: Optional[float]
    corr_translation_error: Optional[float]
    pred_rotation_error: Optional[float]
    corr_rotation_error: Optional[float]
    best_trans_norm: Optional[float]
    best_rot_abs: Optional[float]
    pred_to_corr_dist: Optional[float]
    pred_to_corr_rot: Optional[float]
    icp_iterations: Optional[int]
    icp_mean_error: Optional[float]
    n_correspondences: Optional[int]
    use_transformation: Optional[bool]
    stop_reason: Optional[str]
    step_duration: Optional[float]
    timing_update_particle: Optional[float]


@dataclass
class RunResultScanMatching:
    params: ExperimentParams
    step_results: List[StepResultScanMatching] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class ScanMatchingEvaluator:
    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    @staticmethod
    def translation_error(p1: Pose2D, p2: Pose2D) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def _to_pose_tuple(pose: Any) -> Optional[Pose2D]:
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
        scan_match_failed: bool,
        step_duration: Optional[float],
        timing_update_particle: Optional[float],
    ) -> StepResultScanMatching:
        true_pose_t = self._to_pose_tuple(true_pose)
        pred_pose_t = self._to_pose_tuple(pred_pose)
        corr_pose_t = self._to_pose_tuple(corr_pose)
        est_pose_t = corr_pose_t

        trans_err = None
        rot_err = None
        pred_trans_err = None
        corr_trans_err = None
        pred_rot_err = None
        corr_rot_err = None
        best_trans_norm = None
        best_rot_abs = None
        pred_to_corr_dist = None
        pred_to_corr_rot = None

        if pred_pose_t is not None and true_pose_t is not None:
            pred_trans_err = self.translation_error(pred_pose_t, true_pose_t)
            pred_rot_err = abs(self.angle_diff(pred_pose_t[2], true_pose_t[2]))

        if corr_pose_t is not None and true_pose_t is not None:
            corr_trans_err = self.translation_error(corr_pose_t, true_pose_t)
            corr_rot_err = abs(self.angle_diff(corr_pose_t[2], true_pose_t[2]))

            # Keep existing summary/scorer behavior based on corrected pose error.
            trans_err = corr_trans_err
            rot_err = corr_rot_err

        if pred_pose_t is not None and corr_pose_t is not None:
            pred_to_corr_dist = self.translation_error(corr_pose_t, pred_pose_t)
            pred_to_corr_rot = abs(self.angle_diff(corr_pose_t[2], pred_pose_t[2]))

        if best_transformation is not None:
            tf = np.asarray(best_transformation, dtype=float).reshape(-1)
            if tf.size >= 3 and np.all(np.isfinite(tf[:3])):
                best_trans_norm = float(np.linalg.norm(tf[:2]))
                best_rot_abs = float(abs(tf[2]))

        return StepResultScanMatching(
            step_idx=int(step_idx),
            t=float(t),
            true_pose=true_pose_t,
            pred_pose=pred_pose_t,
            corr_pose=corr_pose_t,
            est_pose=est_pose_t,
            scan_match_failed=bool(scan_match_failed),
            translation_error=trans_err,
            rotation_error=rot_err,
            pred_translation_error=pred_trans_err,
            corr_translation_error=corr_trans_err,
            pred_rotation_error=pred_rot_err,
            corr_rotation_error=corr_rot_err,
            best_trans_norm=best_trans_norm,
            best_rot_abs=best_rot_abs,
            pred_to_corr_dist=pred_to_corr_dist,
            pred_to_corr_rot=pred_to_corr_rot,
            icp_iterations=int(icp_iterations) if icp_iterations is not None else None,
            icp_mean_error=float(icp_mean_error) if icp_mean_error is not None else None,
            n_correspondences=int(n_correspondences) if n_correspondences is not None else None,
            use_transformation=bool(use_transformation) if use_transformation is not None else None,
            stop_reason=str(stop_reason) if stop_reason is not None else None,
            step_duration=float(step_duration) if step_duration is not None else None,
            timing_update_particle=float(timing_update_particle) if timing_update_particle is not None else None,
        )

    def summarize_run(self, step_results: List[StepResultScanMatching], params: ExperimentParams) -> Dict[str, Any]:
        trans_err = [s.translation_error for s in step_results if s.translation_error is not None]
        rot_err = [s.rotation_error for s in step_results if s.rotation_error is not None]
        step_durations = [s.step_duration for s in step_results if s.step_duration is not None]
        update_particle_timings = [
            s.timing_update_particle for s in step_results if s.timing_update_particle is not None
        ]

        return {
            "n_steps": len(step_results),
            "scan_match_failed_count": int(sum(1 for s in step_results if s.scan_match_failed)),
            "mean_translation_error": float(np.mean(trans_err)) if trans_err else float("inf"),
            "mean_rotation_error": float(np.mean(rot_err)) if rot_err else float("inf"),
            "rmse_translation_error": float(np.sqrt(np.mean(np.square(trans_err)))) if trans_err else float("inf"),
            "rmse_rotation_error": float(np.sqrt(np.mean(np.square(rot_err)))) if rot_err else float("inf"),
            "mean_step_duration": float(np.mean(step_durations)) if step_durations else 0.0,
            "mean_timing_sm_update_particle_s": float(np.mean(update_particle_timings)) if update_particle_timings else 0.0,
            "n_particles": int(params.particle_params.n_particles),
        }
