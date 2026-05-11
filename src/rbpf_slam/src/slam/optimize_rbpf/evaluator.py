
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
    trans_err_mu_sm: Optional[float] = None
    rot_err_mu_sm: Optional[float] = None
    trans_err_sm_true: Optional[float] = None
    rot_err_sm_true: Optional[float] = None
    
    trans_err_best_xj_true: Optional[float] = None
    rot_err_best_xj_true: Optional[float] = None
    trans_err_worst_xj_true: Optional[float] = None
    rot_err_worst_xj_true: Optional[float] = None
    best_xj_improves_over_sm_trans : Optional[bool] = None
    best_xj_improves_over_sm_rot : Optional[bool] = None
    best_xj_better_than_worst_trans : Optional[bool] = None
    best_xj_better_than_worst_rot : Optional[bool] = None
    
    trans_err_mu_pred: Optional[float] = None
    rot_err_mu_pred: Optional[float] = None
    prop_std_x: Optional[float] = None
    prop_std_y: Optional[float] = None
    prop_std_theta: Optional[float] = None
    corr_xy: Optional[float] = None
    corr_x_theta: Optional[float] = None
    corr_y_theta: Optional[float] = None
    xj_eff: Optional[float] = None


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
        proposal_metrics: Optional[dict] = None,
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
        trans_err_mu_sm = None
        rot_err_mu_sm = None
        trans_err_sm_true = None
        rot_err_sm_true = None
        trans_err_best_xj_true = None
        rot_err_best_xj_true = None
        trans_err_worst_xj_true = None
        rot_err_worst_xj_true = None
        trans_err_mu_pred = None
        rot_err_mu_pred = None
        prop_std_x = None
        prop_std_y = None
        prop_std_theta = None
        corr_xy = None
        corr_x_theta = None
        corr_y_theta = None
        xj_eff = None
        best_xj_improves_over_sm_trans = None
        best_xj_better_than_worst_trans = None
        best_xj_improves_over_sm_rot = None
        best_xj_better_than_worst_rot = None


        if est_pose_t is not None:
            trans_err = self.translation_error(est_pose_t, true_pose_t)
            rot_err = abs(self.angle_diff(est_pose_t[2], true_pose_t[2]))

        if best_particle_pose_t is not None:
            trans_err_best_p = self.translation_error(best_particle_pose_t, true_pose_t)
            rot_err_best_p = abs(self.angle_diff(best_particle_pose_t[2], true_pose_t[2]))

        # Compute proposal metrics
        if proposal_metrics is not None:
            mu = proposal_metrics.get("prop_mu")
            scan_match_pose = proposal_metrics.get("scan_match_pose")
            pred_pose = proposal_metrics.get("pred_pose")
            cov = proposal_metrics.get("prop_cov_matrix")
            xjs = proposal_metrics.get("xjs")
            xj_weights = proposal_metrics.get("xj_weights")

            if xjs is not None and xj_weights is not None and true_pose_t is not None:
                xjs_arr = np.asarray(xjs, dtype=float)
                weights = np.asarray(xj_weights, dtype=float).reshape(-1)

                if (
                    xjs_arr.ndim == 2
                    and xjs_arr.shape[0] > 0
                    and xjs_arr.shape[1] >= 3
                    and weights.shape[0] == xjs_arr.shape[0]
                ):
                    finite_idx = np.where(np.isfinite(weights))[0]
                    if finite_idx.size > 0:
                        local_best = int(np.argmax(weights[finite_idx]))
                        local_worst = int(np.argmin(weights[finite_idx]))
                        best_idx = int(finite_idx[local_best])
                        worst_idx = int(finite_idx[local_worst])

                        best_xj_t = self._to_pose_tuple(xjs_arr[best_idx, :3])
                        worst_xj_t = self._to_pose_tuple(xjs_arr[worst_idx, :3])

                        trans_err_best_xj_true = self.translation_error(best_xj_t, true_pose_t)
                        rot_err_best_xj_true = abs(self.angle_diff(best_xj_t[2], true_pose_t[2]))
                        trans_err_worst_xj_true = self.translation_error(worst_xj_t, true_pose_t)
                        rot_err_worst_xj_true = abs(self.angle_diff(worst_xj_t[2], true_pose_t[2]))

            if mu is not None and scan_match_pose is not None:
                mu_t = self._to_pose_tuple(mu)
                sm_t = self._to_pose_tuple(scan_match_pose)
                trans_err_mu_sm = self.translation_error(mu_t, sm_t)
                rot_err_mu_sm = abs(self.angle_diff(mu_t[2], sm_t[2]))

                if true_pose_t is not None:
                    trans_err_sm_true = self.translation_error(sm_t, true_pose_t)
                    rot_err_sm_true = abs(self.angle_diff(sm_t[2], true_pose_t[2]))

            if mu is not None and pred_pose is not None:
                mu_t = self._to_pose_tuple(mu)
                pred_t = self._to_pose_tuple(pred_pose)
                trans_err_mu_pred = self.translation_error(mu_t, pred_t)
                rot_err_mu_pred = abs(self.angle_diff(mu_t[2], pred_t[2]))

            if cov is not None:
                cov_arr = np.asarray(cov, dtype=float)
                if cov_arr.shape == (3, 3):
                    diag = np.clip(np.diag(cov_arr), a_min=0.0, a_max=None)
                    std = np.sqrt(diag)
                    prop_std_x = float(std[0])
                    prop_std_y = float(std[1])
                    prop_std_theta = float(std[2])

                    std_x = std[0]
                    std_y = std[1]
                    std_theta = std[2]

                    if std_x > 0.0 and std_y > 0.0:
                        corr_xy = float(cov_arr[0, 1] / (std_x * std_y))
                    if std_x > 0.0 and std_theta > 0.0:
                        corr_x_theta = float(cov_arr[0, 2] / (std_x * std_theta))
                    if std_y > 0.0 and std_theta > 0.0:
                        corr_y_theta = float(cov_arr[1, 2] / (std_y * std_theta))

            if xj_weights is not None:
                weights = np.asarray(xj_weights, dtype=float)
                w_sum = float(np.sum(weights))
                if weights.size > 0 and np.isfinite(w_sum) and w_sum > 0.0:
                    norm_weights = weights / w_sum
                    denom = float(np.sum(norm_weights ** 2))
                    if np.isfinite(denom) and denom > 0.0:
                        xj_eff = float(1.0 / denom)


            # Check if proposal improves scan match pose
            if trans_err_best_xj_true is not None and trans_err_sm_true is not None:
                best_xj_improves_over_sm_trans = trans_err_best_xj_true < trans_err_sm_true

            if rot_err_best_xj_true is not None and rot_err_sm_true is not None:
                best_xj_improves_over_sm_rot = rot_err_best_xj_true < rot_err_sm_true
            
            # Check if worst xj pose is worse than best xj pose
            if trans_err_best_xj_true is not None and trans_err_worst_xj_true is not None:
                best_xj_better_than_worst_trans = trans_err_best_xj_true < trans_err_worst_xj_true

            if rot_err_best_xj_true is not None and rot_err_worst_xj_true is not None:
                best_xj_better_than_worst_rot = rot_err_best_xj_true < rot_err_worst_xj_true


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
            trans_err_mu_sm=trans_err_mu_sm,
            rot_err_mu_sm=rot_err_mu_sm,
            trans_err_sm_true=trans_err_sm_true,
            rot_err_sm_true=rot_err_sm_true,

            trans_err_best_xj_true=trans_err_best_xj_true,
            rot_err_best_xj_true=rot_err_best_xj_true,
            trans_err_worst_xj_true=trans_err_worst_xj_true,
            rot_err_worst_xj_true=rot_err_worst_xj_true,
            best_xj_improves_over_sm_trans=best_xj_improves_over_sm_trans,
            best_xj_improves_over_sm_rot=best_xj_improves_over_sm_rot,
            best_xj_better_than_worst_trans=best_xj_better_than_worst_trans,
            best_xj_better_than_worst_rot=best_xj_better_than_worst_rot,

            trans_err_mu_pred=trans_err_mu_pred,
            rot_err_mu_pred=rot_err_mu_pred,
            prop_std_x=prop_std_x,
            prop_std_y=prop_std_y,
            prop_std_theta=prop_std_theta,
            corr_xy=corr_xy,
            corr_x_theta=corr_x_theta,
            corr_y_theta=corr_y_theta,
            xj_eff=xj_eff,
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
        
        # Compute proposal metrics
        trans_err_mu_sm_values = [s.trans_err_mu_sm for s in step_results if s.trans_err_mu_sm is not None]
        rot_err_mu_sm_values = [s.rot_err_mu_sm for s in step_results if s.rot_err_mu_sm is not None]
        trans_err_mu_pred_values = [s.trans_err_mu_pred for s in step_results if s.trans_err_mu_pred is not None]
        rot_err_mu_pred_values = [s.rot_err_mu_pred for s in step_results if s.rot_err_mu_pred is not None]
        best_xj_improves_over_sm_trans_values = [
            float(s.best_xj_improves_over_sm_trans) for s in step_results if s.best_xj_improves_over_sm_trans is not None
        ]
        best_xj_improves_over_sm_rot_values = [
            float(s.best_xj_improves_over_sm_rot) for s in step_results if s.best_xj_improves_over_sm_rot is not None
        ]
        best_xj_better_than_worst_trans_values = [
            float(s.best_xj_better_than_worst_trans) for s in step_results if s.best_xj_better_than_worst_trans is not None
        ]
        best_xj_better_than_worst_rot_values = [
            float(s.best_xj_better_than_worst_rot) for s in step_results if s.best_xj_better_than_worst_rot is not None
        ]
        prop_std_x_values = [s.prop_std_x for s in step_results if s.prop_std_x is not None]
        prop_std_y_values = [s.prop_std_y for s in step_results if s.prop_std_y is not None]
        prop_std_theta_values = [s.prop_std_theta for s in step_results if s.prop_std_theta is not None]
        
        mean_prop_std_xy = np.mean(
            [(sx + sy) / 2.0 for sx, sy in zip(prop_std_x_values, prop_std_y_values)] 
        )
        mean_std_theta = np.mean(prop_std_theta_values) if prop_std_theta_values else float("nan")
        
        prop_corr_xy_values = [s.corr_xy for s in step_results if s.corr_xy is not None]
        prop_corr_x_theta_values = [s.corr_x_theta for s in step_results if s.corr_x_theta is not None]
        prop_corr_y_theta_values = [s.corr_y_theta for s in step_results if s.corr_y_theta is not None]
        xj_eff_values = [s.xj_eff for s in step_results if s.xj_eff is not None]

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
            
            "mean_trans_err_mu_sm": float(np.mean(trans_err_mu_sm_values)) if trans_err_mu_sm_values else float("nan"),
            "mean_rot_err_mu_sm": float(np.mean(rot_err_mu_sm_values)) if rot_err_mu_sm_values else float("nan"),
            "rmse_trans_err_mu_sm": float(np.sqrt(np.mean(np.square(trans_err_mu_sm_values)))) if trans_err_mu_sm_values else float("nan"),
            "rmse_rot_err_mu_sm": float(np.sqrt(np.mean(np.square(rot_err_mu_sm_values)))) if rot_err_mu_sm_values else float("nan"),
            "mean_trans_err_mu_pred": float(np.mean(trans_err_mu_pred_values)) if trans_err_mu_pred_values else float("nan"),
            "mean_rot_err_mu_pred": float(np.mean(rot_err_mu_pred_values)) if rot_err_mu_pred_values else float("nan"),
            "rmse_trans_err_mu_pred": float(np.sqrt(np.mean(np.square(trans_err_mu_pred_values)))) if trans_err_mu_pred_values else float("nan"),
            "rmse_rot_err_mu_pred": float(np.sqrt(np.mean(np.square(rot_err_mu_pred_values)))) if rot_err_mu_pred_values else float("nan"),
            "mean_best_xj_improves_over_sm_trans": float(np.mean(best_xj_improves_over_sm_trans_values)) if best_xj_improves_over_sm_trans_values else float("nan"),
            "mean_best_xj_improves_over_sm_rot": float(np.mean(best_xj_improves_over_sm_rot_values)) if best_xj_improves_over_sm_rot_values else float("nan"),
            "mean_best_xj_better_than_worst_trans": float(np.mean(best_xj_better_than_worst_trans_values)) if best_xj_better_than_worst_trans_values else float("nan"),
            "mean_best_xj_better_than_worst_rot": float(np.mean(best_xj_better_than_worst_rot_values)) if best_xj_better_than_worst_rot_values else float("nan"),
            
            "mean_prop_std_xy": mean_prop_std_xy,
            "mean_prop_std_theta": mean_std_theta,
            
            "mean_prop_corr_xy": float(np.mean(prop_corr_xy_values)) if prop_corr_xy_values else float("nan"),
            "mean_prop_corr_x_theta": float(np.mean(prop_corr_x_theta_values)) if prop_corr_x_theta_values else float("nan"),
            "mean_prop_corr_y_theta": float(np.mean(prop_corr_y_theta_values)) if prop_corr_y_theta_values else float("nan"),
            "mean_xj_eff": float(np.mean(xj_eff_values)) if xj_eff_values else float("nan"),
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