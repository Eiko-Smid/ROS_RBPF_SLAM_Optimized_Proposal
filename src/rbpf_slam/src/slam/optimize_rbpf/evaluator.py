
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .playback_defs import ExperimentParams

Pose2D = Tuple[float, float, float]


@dataclass
class StepResult:
    """
    Stores evaluation data for one RBPF step.
    """
    step_idx: int
    t: float
    true_pose: Pose2D
    est_pose: Optional[Pose2D]
    best_particle_pose: Optional[Pose2D]
    neff: Optional[float]
    scan_match_failed: Optional[bool] = None
    scan_match_fallback_failed: Optional[bool] = None
    translation_error: Optional[float] = None
    rotation_error: Optional[float] = None
    translation_error_best_p: Optional[float] = None
    rotation_error_best_p: Optional[float] = None
    particle_weight_min: Optional[float] = None
    particle_weight_max: Optional[float] = None
    particle_weight_mean: Optional[float] = None
    step_duration: Optional[float] = None


@dataclass
class RunResult:
    """
    Stores all RBPF evaluation data for one parameter-set run.
    """
    params: ExperimentParams
    step_results: List[StepResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class RBPFEvaluator:
    """
    Computes per-step errors and run-level metrics for one RBPF playback run.
    """
    @staticmethod
    def _to_pose_tuple(pose) -> Optional[Pose2D]:
        """
        Converts a pose object to (x, y, theta).

        Supports tuples/lists/ndarrays and objects exposing x/y/theta attributes.
        """
        if pose is None:
            return None

        if hasattr(pose, "x") and hasattr(pose, "y") and hasattr(pose, "theta"):
            return (float(pose.x), float(pose.y), float(pose.theta))

        if isinstance(pose, (tuple, list, np.ndarray)) and len(pose) >= 3:
            return (float(pose[0]), float(pose[1]), float(pose[2]))

        raise TypeError(f"Unsupported pose format: {type(pose)}")

    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        """
        Returns wrapped angular difference in [-pi, pi].
        """
        return math.atan2(math.sin(a - b), math.cos(a - b))

    @staticmethod
    def translation_error(p1: Pose2D, p2: Pose2D) -> float:
        """
        Euclidean translation error in the x-y plane.
        """
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    def evaluate_step(
        self,
        step_idx: int,
        t: float,
        true_pose,
        est_pose,
        best_particle_pose,
        scan_match_failed: Optional[bool],
        scan_match_fallback_failed: Optional[bool],
        neff: Optional[float],
        particle_weight_min: Optional[float],
        particle_weight_max: Optional[float],
        particle_weight_mean: Optional[float],
        step_duration: Optional[float],
    ) -> StepResult:
        """
        Evaluates one RBPF step and returns per-step errors.
        """
        true_pose_t = self._to_pose_tuple(true_pose)
        est_pose_t = self._to_pose_tuple(est_pose)
        best_particle_pose_t = self._to_pose_tuple(best_particle_pose)

        trans_err = None
        rot_err = None
        trans_err_best_p = None
        rot_err_best_p = None

        if est_pose_t is not None:
            trans_err = self.translation_error(est_pose_t, true_pose_t)
            rot_err = abs(self.angle_diff(est_pose_t[2], true_pose_t[2]))

        if best_particle_pose_t is not None:
            trans_err_best_p = self.translation_error(best_particle_pose_t, true_pose_t)
            rot_err_best_p = abs(self.angle_diff(best_particle_pose_t[2], true_pose_t[2]))

        return StepResult(
            step_idx=step_idx,
            t=float(t),
            true_pose=true_pose_t,
            est_pose=est_pose_t,
            best_particle_pose=best_particle_pose_t,
            neff=float(neff) if neff is not None else None,
            scan_match_failed=scan_match_failed,
            scan_match_fallback_failed=scan_match_fallback_failed,
            translation_error=trans_err,
            rotation_error=rot_err,
            translation_error_best_p=trans_err_best_p,
            rotation_error_best_p=rot_err_best_p,
            particle_weight_min=float(particle_weight_min) if particle_weight_min is not None else None,
            particle_weight_max=float(particle_weight_max) if particle_weight_max is not None else None,
            particle_weight_mean=float(particle_weight_mean) if particle_weight_mean is not None else None,
            step_duration=float(step_duration) if step_duration is not None else None,
        )

    def summarize_run(self, step_results: List[StepResult], params: Optional[ExperimentParams] = None) -> dict:
        """
        Computes run-level metrics for optimization and reporting.
        """
        trans_err = [s.translation_error for s in step_results if s.translation_error is not None]
        rot_err = [s.rotation_error for s in step_results if s.rotation_error is not None]
        trans_err_best_p = [s.translation_error_best_p for s in step_results if s.translation_error_best_p is not None]
        rot_err_best_p = [s.rotation_error_best_p for s in step_results if s.rotation_error_best_p is not None]
        scan_match_failed_count = sum(1 for s in step_results if s.scan_match_failed)
        scan_match_fallback_failed_count = sum(1 for s in step_results if s.scan_match_fallback_failed)
        neff_values = [s.neff for s in step_results if s.neff is not None]
        particle_weight_min_values = [s.particle_weight_min for s in step_results if s.particle_weight_min is not None]
        particle_weight_max_values = [s.particle_weight_max for s in step_results if s.particle_weight_max is not None]
        particle_weight_mean_values = [s.particle_weight_mean for s in step_results if s.particle_weight_mean is not None]
        step_durations = [s.step_duration for s in step_results if s.step_duration is not None]

        drift = float("inf")
        drift_rotation_error = float("inf")

        for s in reversed(step_results):
            if s.est_pose is not None:
                drift = self.translation_error(s.est_pose, s.true_pose)
                drift_rotation_error = abs(self.angle_diff(s.est_pose[2], s.true_pose[2]))
                break

        summary = {
            "n_steps": len(step_results),
            "scan_match_failed_count": int(scan_match_failed_count),
            "scan_match_fallback_failed_count": int(scan_match_fallback_failed_count),
            "mean_translation_error": float(np.mean(trans_err)) if trans_err else float("inf"),
            "mean_rotation_error": float(np.mean(rot_err)) if rot_err else float("inf"),
            "rmse_translation_error": float(np.sqrt(np.mean(np.square(trans_err)))) if trans_err else float("inf"),
            "rmse_rotation_error": float(np.sqrt(np.mean(np.square(rot_err)))) if rot_err else float("inf"),
            "mean_trans_err_best_p": float(np.mean(trans_err_best_p)) if trans_err_best_p else float("inf"),
            "mean_rot_err_best_p": float(np.mean(rot_err_best_p)) if rot_err_best_p else float("inf"),
            "rmse_trans_error_best_p": float(np.sqrt(np.mean(np.square(trans_err_best_p)))) if trans_err_best_p else float("inf"),
            "rmse_rot_error_best_p": float(np.sqrt(np.mean(np.square(rot_err_best_p)))) if rot_err_best_p else float("inf"),
            "drift": drift,
            "drift_rotation_error": drift_rotation_error,
            "mean_neff": float(np.mean(neff_values)) if neff_values else 0.0,
            "mean_particle_weight_min": float(np.mean(particle_weight_min_values)) if particle_weight_min_values else 0.0,
            "mean_particle_weight_max": float(np.mean(particle_weight_max_values)) if particle_weight_max_values else 0.0,
            "mean_particle_weight_mean": float(np.mean(particle_weight_mean_values)) if particle_weight_mean_values else 0.0,
            "mean_step_duration": float(np.mean(step_durations)) if step_durations else 0.0,
        }

        if params is not None:
            summary.update(
                {
                    "n_particles": self._extract_n_particles(params),
                    "sigma_measurement": self._extract_sigma_measurement(params),
                    "neff_threshold": self._extract_neff_threshold(params),
                }
            )

        return summary

    @staticmethod
    def _extract_n_particles(params: ExperimentParams) -> Optional[int]:
        if hasattr(params, "particle_params") and hasattr(params.particle_params, "n_particles"):
            return int(params.particle_params.n_particles)
        if hasattr(params, "n_particles"):
            return int(params.n_particles)
        return None

    @staticmethod
    def _extract_sigma_measurement(params: ExperimentParams) -> Optional[float]:
        if hasattr(params, "measurement_model_params") and hasattr(params.measurement_model_params, "sigma_measurement"):
            return float(params.measurement_model_params.sigma_measurement)
        if hasattr(params, "sigma_measurement"):
            return float(params.sigma_measurement)
        return None

    @staticmethod
    def _extract_neff_threshold(params: ExperimentParams) -> Optional[float]:
        if hasattr(params, "neff_threshold"):
            value = params.neff_threshold
            return float(value) if value is not None else None

        if hasattr(params, "particle_params") and hasattr(params.particle_params, "n_particles"):
            return float(params.particle_params.n_particles) / 2.0

        return None