
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
    neff: Optional[float]
    translation_error: Optional[float]
    rotation_error: Optional[float]
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
        neff: Optional[float],
        step_duration: Optional[float],
    ) -> StepResult:
        """
        Evaluates one RBPF step and returns per-step errors.
        """
        true_pose_t = self._to_pose_tuple(true_pose)
        est_pose_t = self._to_pose_tuple(est_pose)

        trans_err = None
        rot_err = None

        if est_pose_t is not None:
            trans_err = self.translation_error(est_pose_t, true_pose_t)
            rot_err = abs(self.angle_diff(est_pose_t[2], true_pose_t[2]))

        return StepResult(
            step_idx=step_idx,
            t=float(t),
            true_pose=true_pose_t,
            est_pose=est_pose_t,
            neff=float(neff) if neff is not None else None,
            translation_error=trans_err,
            rotation_error=rot_err,
            step_duration=float(step_duration) if step_duration is not None else None,
        )

    def summarize_run(self, step_results: List[StepResult], params: Optional[ExperimentParams] = None) -> dict:
        """
        Computes run-level metrics for optimization and reporting.
        """
        trans_err = [s.translation_error for s in step_results if s.translation_error is not None]
        rot_err = [s.rotation_error for s in step_results if s.rotation_error is not None]
        neff_values = [s.neff for s in step_results if s.neff is not None]
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
            "rmse_translation_error": float(np.sqrt(np.mean(np.square(trans_err)))) if trans_err else float("inf"),
            "rmse_rotation_error": float(np.sqrt(np.mean(np.square(rot_err)))) if rot_err else float("inf"),
            "drift": drift,
            "drift_rotation_error": drift_rotation_error,
            "mean_neff": float(np.mean(neff_values)) if neff_values else 0.0,
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