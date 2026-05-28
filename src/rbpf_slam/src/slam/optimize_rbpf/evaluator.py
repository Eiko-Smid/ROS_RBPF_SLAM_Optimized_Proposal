
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from .playback_defs import ExperimentParams


# Transfers the angle in rad into meters to combine translational and rotational errors 
ROT_SCALE = 2.0     # trans_err + ROT_SCALE * angle (rad) -> m

Pose2D = Tuple[float, float, float]


@dataclass
class StepResult:
    """
    Stores evaluation data for one RBPF step.
    """
    step_idx: int
    t: float
    true_pose: Pose2D
    raw_odom_pose: Optional[Pose2D]
    est_pose: Optional[Pose2D]
    best_particle_pose: Optional[Pose2D]
    neff: Optional[float]
    scan_match_pose: Optional[Pose2D] = None
    scan_match_failed: Optional[bool] = None
    scan_match_fallback_failed: Optional[bool] = None
    translation_error: Optional[float] = None
    rotation_error: Optional[float] = None
    trans_err_raw_odom: Optional[float] = None
    rot_err_raw_odom: Optional[float] = None
    translation_error_best_p: Optional[float] = None
    rotation_error_best_p: Optional[float] = None
    particle_weight_min: Optional[float] = None
    particle_weight_max: Optional[float] = None
    particle_weight_mean: Optional[float] = None
    step_duration: Optional[float] = None
    
    trans_err_mu_true: Optional[float] = None
    rot_err_mu_true: Optional[float] = None
    pose_err_mu_true: Optional[float] = None
    trans_err_mu_sm: Optional[float] = None
    rot_err_mu_sm: Optional[float] = None
    trans_err_sm_true: Optional[float] = None
    rot_err_sm_true: Optional[float] = None
    pose_err_sm_true: Optional[float] = None
    mu_true_err_improves_over_sm_true: Optional[float] = None
    
    min_xj_pose_err_true: Optional[float] = None
    min_xj_is_best_xj: Optional[bool] = None
    weight_min_xj_err: Optional[float] = None
    best_weighted_xj_pose_err_true: Optional[float] = None
    weight_best_xj: Optional[float] = None
    rot_err_best_xj_true: Optional[float] = None
    trans_err_worst_xj_true: Optional[float] = None
    rot_err_worst_xj_true: Optional[float] = None
    min_xj_true_err_improves_over_sm_true: Optional[float] = None
    best_xj_true_err_improves_over_sm_true : Optional[float] = None
    best_xj_improves_over_sm_rot : Optional[bool] = None
    best_xj_better_than_worst_trans : Optional[bool] = None
    best_xj_better_than_worst_rot : Optional[bool] = None
    min_xj_true_err_weight_score: Optional[float] = None
    corr_xjs_weights: Optional[float] = None
    corr_xjs_motion: Optional[float] = None
    corr_xjs_meas: Optional[float] = None
    corr_weights_motion: Optional[float] = None
    corr_weights_meas: Optional[float] = None
    best_xj_score: Optional[float] = None
    motion_rank_score: Optional[float] = None
    meas_rank_score: Optional[float] = None
    weight_ratio_min_best_weight: Optional[float] = None
    log_motion_range: Optional[float] = None
    log_meas_range: Optional[float] = None
    log_weight_range: Optional[float] = None
    xj_indices: Optional[List[int]] = None
    xj_pose_err: Optional[List[float]] = None
    xj_weight: Optional[List[float]] = None
    xj_motion: Optional[List[float]] = None
    xj_meas: Optional[List[float]] = None
    
    trans_err_mu_pred: Optional[float] = None
    rot_err_mu_pred: Optional[float] = None
    prop_std_x: Optional[float] = None
    prop_std_y: Optional[float] = None
    prop_std_theta: Optional[float] = None
    corr_xy: Optional[float] = None
    corr_x_theta: Optional[float] = None
    corr_y_theta: Optional[float] = None
    xj_eff: Optional[float] = None
    xj_eff_motion: Optional[float] = None
    xj_eff_meas: Optional[float] = None


@dataclass
class RunResult:
    """
    Stores all RBPF evaluation data for one parameter-set run.
    """
    params: ExperimentParams
    step_results: List[StepResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class RBPFEvaluator:
    # non_finite_count = 0
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


    @staticmethod
    def pose_err(trans_err: float, rot_err: float, rot_scale: float) -> float:
        '''
        Get's the translational and rotational errors between two poses and computes a combined error metric
        that allows to compare the overall error of two poses.
        '''
        return float(np.sqrt(trans_err**2 + (rot_scale * rot_err)**2))


    @staticmethod
    def xj_weight_score(xj_pose_errors_true: np.ndarray, xj_weights: np.ndarray) -> float:
        '''
        Computes the normalized rank score of the xj closest to the true pose based on its weight.
        The bigger the weight of the weights list, the xj corresponds to, the higher the score.

        Parameters
        ----------
        xj_pose_errors_true: np.ndarray
            Array of pose errors of each xj pose to the true pose.
        xj_weights: np.ndarray
            Array of weights for each xj pose.

        Returns         
        ---------
        rank_score: float
            Normalized rank score of the xj closest to the true pose based on its weight. 
                    
                rank_score = 1.0  -> closest-to-true xj has highest weight
                rank_score = 0.0  -> closest-to-true xj has lowest weight
                rank_score ≈ 0.5  -> closest-to-true xj is around the middle of the weight ranking

        '''
        # Find index of lowest error
        idx_closest_true = np.argmin(xj_pose_errors_true)

        # Order the negated weights from low to high
        order = np.argsort(-xj_weights)

        # Compute the rank at which the closest-to-true xj appears in the weight ranking 
        rank_of_closest = int(np.where(order == idx_closest_true)[0][0]) + 1
        N = len(xj_weights)
        
        # Compute score
        if N == 1:
            rank_score = 1.0
        else:
            rank_score = 1.0 - (rank_of_closest - 1) / (N - 1)
        
        return rank_score
    

    @staticmethod
    def rank_model_probs(pose_errors, weights):
        # get idx of max weights
        max_weight_idx = np.argmax(weights)

        # Pseudo sort pose err from low to high
        order = np.argsort(pose_errors)

        # Compute rank
        rank = int(np.where(order == max_weight_idx)[0][0]) + 1

        # Compute score
        N = len(pose_errors)
        if N == 1:
            rank_score = 1.0
        else:
            rank_score = 1.0 - (rank - 1) / (N - 1)
        
        return rank_score


    @staticmethod
    def _finite_values(values: List[Optional[float]]) -> List[float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return []
        return arr[np.isfinite(arr)].astype(float).tolist()
    

    def evaluate_step(
        self,
        step_idx: int,
        t: float,
        true_pose,
        raw_odom_pose,
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
        raw_odom_pose_t = self._to_pose_tuple(raw_odom_pose)
        scan_match_pose_t = None
        est_pose_t = self._to_pose_tuple(est_pose)
        best_particle_pose_t = self._to_pose_tuple(best_particle_pose)

        trans_err = None
        rot_err = None
        trans_err_raw_odom = None
        rot_err_raw_odom = None
        trans_err_best_p = None
        rot_err_best_p = None
        trans_err_mu_true = None
        rot_err_mu_true = None
        pose_err_mu_true = None
        trans_err_mu_sm = None
        rot_err_mu_sm = None

        trans_err_sm_true = None
        rot_err_sm_true = None
        pose_err_sm_true = None
        mu_true_err_improves_over_sm_true = None
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
        xj_eff_motion = None
        xj_eff_meas = None

        min_xj_pose_err_true = None
        min_xj_is_best_xj = None
        weight_min_xj_err = None
        best_weighted_xj_pose_err_true = None
        weight_best_xj = None
        best_xj_true_err_improves_over_sm_true = None
        best_xj_better_than_worst_trans = None
        best_xj_improves_over_sm_rot = None
        best_xj_better_than_worst_rot = None
        
        trans_errors_xj_true = []
        rot_errors_xj_true = []
        pose_errors_xj_true = []

        min_xj_true_err_improves_over_sm_true = None
        xj_improves_over_sm_rot_ratio = None
        min_xj_true_err_weight_score = None
        corr_xjs_weights = None
        corr_xjs_motion = None
        corr_xjs_meas = None
        corr_weights_motion = None
        corr_weights_meas = None
        best_xj_score = None
        motion_rank_score = None
        meas_rank_score = None
        weight_ratio_min_best_weight = None
        log_motion_range = None
        log_meas_range = None
        log_weight_range = None
        xj_indices = None
        xj_pose_err = None
        xj_weight = None
        xj_motion = None
        xj_meas = None

        if raw_odom_pose_t is not None:
            trans_err_raw_odom = self.translation_error(raw_odom_pose_t, true_pose_t)
            rot_err_raw_odom = abs(self.angle_diff(raw_odom_pose_t[2], true_pose_t[2]))

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
            scan_match_pose_t = self._to_pose_tuple(scan_match_pose)
            pred_pose = proposal_metrics.get("pred_pose")
            cov = proposal_metrics.get("prop_cov_matrix")
            xjs = proposal_metrics.get("xjs")
            xj_weights = proposal_metrics.get("xj_weights")
            motion_probs = proposal_metrics.get("motion_probs")
            meas_probs = proposal_metrics.get("meas_probs")


            # Compute mu, sm error metrics
            if mu is not None and scan_match_pose is not None:
                mu_t = self._to_pose_tuple(mu)
                sm_t = self._to_pose_tuple(scan_match_pose)
                trans_err_mu_sm = self.translation_error(mu_t, sm_t)
                rot_err_mu_sm = abs(self.angle_diff(mu_t[2], sm_t[2]))

                if true_pose_t is not None:
                    trans_err_sm_true = self.translation_error(sm_t, true_pose_t)
                    rot_err_sm_true = abs(self.angle_diff(sm_t[2], true_pose_t[2]))
                    pose_err_sm_true = self.pose_err(trans_err_sm_true, rot_err_sm_true, ROT_SCALE)

                    trans_err_mu_true = self.translation_error(mu_t, true_pose_t)
                    rot_err_mu_true = abs(self.angle_diff(mu_t[2], true_pose_t[2]))
                    pose_err_mu_true = self.pose_err(trans_err_mu_true, rot_err_mu_true, ROT_SCALE)

                    # >0: mu is better than sm, =0: mu is equal to sm, <0: mu is worse than sm,
                    mu_true_err_improves_over_sm_true = (pose_err_sm_true - pose_err_mu_true) / (pose_err_sm_true + 1e-12)

            # Analyze proposal xjs and weights
            if xjs is not None and xj_weights is not None and true_pose_t is not None:
                xjs_arr = np.asarray(xjs, dtype=float)
                weights = np.asarray(xj_weights, dtype=float).reshape(-1)
                motion_probs_arr = np.asarray(motion_probs, dtype=float).reshape(-1)
                meas_probs_arr = np.asarray(meas_probs, dtype=float).reshape(-1)

                if (
                    xjs_arr.ndim == 2
                    and xjs_arr.shape[0] > 0
                    and xjs_arr.shape[1] >= 3
                    and weights.shape[0] == xjs_arr.shape[0]
                ):
                    valid_idx = np.where(
                        np.isfinite(weights) & np.all(np.isfinite(xjs_arr[:, :3]), axis=1)
                    )[0]

                    if valid_idx.size == 0:
                        print(
                            f"[RBPFEvaluator] Step {step_idx}: skipping Proposal metrics computations, no shared finite xjs/weights samples."
                        )
                    else:
                        xj_indices = valid_idx.astype(int).tolist()

                        # Filter all proposal vectors on the same valid indices.
                        xjs_arr = xjs_arr[valid_idx]
                        weights = weights[valid_idx]
                        motion_probs_arr = motion_probs_arr[valid_idx]
                        meas_probs_arr = meas_probs_arr[valid_idx]

                        weight_sum = float(np.sum(weights))
                       
                        norm_weights = weights / weight_sum

                        # Find index of best and worst weight
                        best_idx = int(np.argmax(weights))
                        worst_idx = int(np.argmin(weights))
                        n_samples = int(weights.shape[0])

                        # Store normalized best-weighted xj weight for CSV/reporting.
                        weight_best_xj = float(norm_weights[best_idx])

                        # Compute errors between true pose and xjs
                        for xj in xjs_arr:
                            xj_t = self._to_pose_tuple(xj)
                            t_err = self.translation_error(xj_t, true_pose_t)
                            r_err = abs(self.angle_diff(xj_t[2], true_pose_t[2]))
                            p_err = self.pose_err(t_err, r_err, ROT_SCALE)
                            trans_errors_xj_true.append(t_err)
                            rot_errors_xj_true.append(r_err)
                            pose_errors_xj_true.append(p_err)

                        # Ensure numpy arrays for easier computations
                        trans_errors_xj_true = np.asarray(trans_errors_xj_true, dtype=float)
                        rot_errors_xj_true = np.asarray(rot_errors_xj_true, dtype=float)
                        pose_errors_xj_true = np.asarray(pose_errors_xj_true, dtype=float)

                        # Store per-sample proposal diagnostics for optional CSV export.
                        # xj_weight/xj_motion/xj_meas are exported as normalized arrays.
                        xj_pose_err = pose_errors_xj_true.astype(float).tolist()

                        # Find xj which has min error to true pose
                        min_xj_pose_err_true = np.min(pose_errors_xj_true)

                        # This ratio tells us how much closer the xj pose with lowest pose error is to the true pose compared to the scan match pose.
                        min_xj_true_err_improves_over_sm_true = (pose_err_sm_true - min_xj_pose_err_true) / (pose_err_sm_true + 1e-12)

                        # Check if best xj pose is better than scan match pose
                        best_weighted_xj_pose_err_true = pose_errors_xj_true[best_idx]
                        best_xj_true_err_improves_over_sm_true = (pose_err_sm_true - best_weighted_xj_pose_err_true) / (pose_err_sm_true + 1e-12)

                        # Compute rank score of xj closest to the true pose based on the given weights.
                        min_xj_true_err_weight_score = self.xj_weight_score(
                            xj_pose_errors_true=pose_errors_xj_true,
                            xj_weights=weights
                        )

                        # Measure if the min xj beats the bes xj
                        min_xj_is_best_xj = bool(best_idx == np.argmin(pose_errors_xj_true))

                        # Compute correlation between all xj errors to true pose and corresponding weights/probs
                        corr_xjs_weights, _ = spearmanr(-pose_errors_xj_true, weights)
                        corr_xjs_motion, _ = spearmanr(-pose_errors_xj_true, motion_probs_arr)
                        corr_xjs_meas, _ = spearmanr(-pose_errors_xj_true, meas_probs_arr)
                        # COmpute correlation between weights and probs
                        corr_weights_motion, _ = spearmanr(weights, motion_probs_arr)
                        corr_weights_meas, _ = spearmanr(weights, meas_probs_arr)

                        # Compute importance score of best xj's weight.
                        min_xj_err_idx = np.argmin(pose_errors_xj_true)
                        weight_min_xj_err = float(norm_weights[min_xj_err_idx])
                        uniform_weight = 1 / n_samples
                        best_weight_importance = weight_min_xj_err / uniform_weight

                        # Compute union score for best xj and its weight.
                        best_xj_score = best_weight_importance * min_xj_true_err_improves_over_sm_true

                        # Analyse motion and measurement model weights
                        motion_rank_score = self.rank_model_probs(
                            pose_errors=pose_errors_xj_true,
                            weights=motion_probs_arr,
                        )

                        meas_rank_score = self.rank_model_probs(
                            pose_errors=pose_errors_xj_true,
                            weights=meas_probs_arr,
                        )

                        # Effective sample size based on the same filtered normalized weights.
                        denom = float(np.sum(norm_weights ** 2))
                        if np.isfinite(denom) and denom > 0.0:
                            xj_eff = float(1.0 / denom)

                        # Effective sample size for motion-model weights.
                        motion_sum = float(np.sum(motion_probs_arr))
                        motion_norm = motion_probs_arr / motion_sum
                        denom_motion = float(np.sum(motion_norm ** 2))                    
                        xj_eff_motion = float(1.0 / denom_motion)

                        # Effective sample size for measurement-model weights.
                        meas_sum = float(np.sum(meas_probs_arr))                        
                        meas_norm = meas_probs_arr / meas_sum
                        denom_meas = float(np.sum(meas_norm ** 2))
                        xj_eff_meas = float(1.0 / denom_meas)

                        # Store normalized per-sample proposal diagnostics for CSV export.
                        xj_weight = norm_weights.astype(float).tolist()
                        xj_motion = motion_norm.astype(float).tolist()
                        xj_meas = meas_norm.astype(float).tolist()

                        if norm_weights.shape[0] < 27:
                            print("Weights lower than 27")

                        # Compute weight ratio between best xj and min xj weight
                        weight_ratio_min_best_weight = weight_min_xj_err / weight_best_xj 

                        # Motion model weight range (normalized motion probabilities)
                        log_motion_probs = np.log(motion_norm + 1e-12)
                        log_motion_range = np.max(log_motion_probs) - np.min(log_motion_probs)#

                        # measurement model weight range (normalized measurement probabilities)
                        log_meas_probs = np.log(meas_norm + 1e-12)
                        log_meas_range = np.max(log_meas_probs) - np.min(log_meas_probs)

                        # weight range
                        log_weights = np.log(norm_weights + 1e-12)
                        log_weight_range = np.max(log_weights) - np.min(log_weights)

                else:
                    print(
                        f"[RBPFEvaluator] Step {step_idx}: skipping Proposal metrics computations, invalid xjs/weights base shapes."
                    )
                        

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

            # Check if proposal improves scan match pose
            # if trans_err_best_xj_true is not None and trans_err_sm_true is not None:
            #     best_xj_improves_over_sm_trans = trans_err_best_xj_true < trans_err_sm_true

            # if rot_err_best_xj_true is not None and rot_err_sm_true is not None:
            #     best_xj_improves_over_sm_rot = rot_err_best_xj_true < rot_err_sm_true
            
            # # Check if worst xj pose is worse than best xj pose
            # if trans_err_best_xj_true is not None and trans_err_worst_xj_true is not None:
            #     best_xj_better_than_worst_trans = trans_err_best_xj_true < trans_err_worst_xj_true

            # if rot_err_best_xj_true is not None and rot_err_worst_xj_true is not None:
            #     best_xj_better_than_worst_rot = rot_err_best_xj_true < rot_err_worst_xj_true

        return StepResult(
            step_idx=step_idx,
            t=float(t),
            true_pose=true_pose_t,
            raw_odom_pose=raw_odom_pose_t,
            scan_match_pose=scan_match_pose_t,
            est_pose=est_pose_t,
            best_particle_pose=best_particle_pose_t,
            neff=float(neff) if neff is not None else None,
            scan_match_failed=scan_match_failed,
            scan_match_fallback_failed=scan_match_fallback_failed,
            translation_error=trans_err,
            rotation_error=rot_err,
            trans_err_raw_odom=trans_err_raw_odom,
            rot_err_raw_odom=rot_err_raw_odom,
            translation_error_best_p=trans_err_best_p,
            rotation_error_best_p=rot_err_best_p,
            particle_weight_min=float(particle_weight_min) if particle_weight_min is not None else None,
            particle_weight_max=float(particle_weight_max) if particle_weight_max is not None else None,
            particle_weight_mean=float(particle_weight_mean) if particle_weight_mean is not None else None,
            step_duration=float(step_duration) if step_duration is not None else None,
            
            trans_err_mu_true=trans_err_mu_true,
            rot_err_mu_true=rot_err_mu_true,
            pose_err_mu_true=pose_err_mu_true,
            trans_err_mu_sm=trans_err_mu_sm,
            rot_err_mu_sm=rot_err_mu_sm,
            trans_err_mu_pred=trans_err_mu_pred,
            rot_err_mu_pred=rot_err_mu_pred,
            
            trans_err_sm_true=trans_err_sm_true,
            rot_err_sm_true=rot_err_sm_true,
            pose_err_sm_true=pose_err_sm_true,
            
            min_xj_pose_err_true=min_xj_pose_err_true,
            min_xj_is_best_xj=min_xj_is_best_xj,
            weight_min_xj_err=weight_min_xj_err,
            best_weighted_xj_pose_err_true=best_weighted_xj_pose_err_true,
            weight_best_xj=weight_best_xj,
            # rot_err_best_xj_true=rot_err_best_xj_true,
            # trans_err_worst_xj_true=trans_err_worst_xj_true,
            # rot_err_worst_xj_true=rot_err_worst_xj_true,
            # best_xj_improves_over_sm_trans=best_xj_improves_over_sm_trans,
            # best_xj_improves_over_sm_rot=best_xj_improves_over_sm_rot,
            # best_xj_better_than_worst_trans=best_xj_better_than_worst_trans,
            # best_xj_better_than_worst_rot=best_xj_better_than_worst_rot,
            min_xj_true_err_improves_over_sm_true=min_xj_true_err_improves_over_sm_true,    
            best_xj_true_err_improves_over_sm_true=best_xj_true_err_improves_over_sm_true,  
            min_xj_true_err_weight_score=min_xj_true_err_weight_score,                      
            corr_xjs_weights=corr_xjs_weights,                                                         
            corr_xjs_motion=corr_xjs_motion,                                                  
            corr_xjs_meas=corr_xjs_meas,                                                       
            corr_weights_motion=corr_weights_motion,
            corr_weights_meas=corr_weights_meas,
            best_xj_score=best_xj_score,
            motion_rank_score=motion_rank_score,
            meas_rank_score=meas_rank_score,
            weight_ratio_min_best_weight=weight_ratio_min_best_weight,
            log_motion_range=log_motion_range,
            log_meas_range=log_meas_range,
            log_weight_range=log_weight_range,
            xj_indices=xj_indices,
            xj_pose_err=xj_pose_err,
            xj_weight=xj_weight,
            xj_motion=xj_motion,
            xj_meas=xj_meas,
            mu_true_err_improves_over_sm_true=mu_true_err_improves_over_sm_true,

            prop_std_x=prop_std_x,
            prop_std_y=prop_std_y,
            prop_std_theta=prop_std_theta,
            corr_xy=corr_xy,
            corr_x_theta=corr_x_theta,
            corr_y_theta=corr_y_theta,
            xj_eff=xj_eff,
            xj_eff_motion=xj_eff_motion,
            xj_eff_meas=xj_eff_meas,
        )


    def summarize_run(self, step_results: List[StepResult], params: Optional[ExperimentParams] = None) -> dict:
        """
        Computes run-level metrics for optimization and reporting.
        """
        # Filter out infinite values form data before comptuaing summary metrics
        trans_err = self._finite_values([s.translation_error for s in step_results if s.translation_error is not None])
        rot_err = self._finite_values([s.rotation_error for s in step_results if s.rotation_error is not None])
        trans_err_raw_odom = self._finite_values([s.trans_err_raw_odom for s in step_results if s.trans_err_raw_odom is not None])
        rot_err_raw_odom = self._finite_values([s.rot_err_raw_odom for s in step_results if s.rot_err_raw_odom is not None])
        trans_err_best_p = self._finite_values([s.translation_error_best_p for s in step_results if s.translation_error_best_p is not None])
        rot_err_best_p = self._finite_values([s.rotation_error_best_p for s in step_results if s.rotation_error_best_p is not None])
        scan_match_failed_count = sum(1 for s in step_results if s.scan_match_failed)
        scan_match_fallback_failed_count = sum(1 for s in step_results if s.scan_match_fallback_failed)
        neff_values = self._finite_values([s.neff for s in step_results if s.neff is not None])
        particle_weight_min_values = self._finite_values([s.particle_weight_min for s in step_results if s.particle_weight_min is not None])
        particle_weight_max_values = self._finite_values([s.particle_weight_max for s in step_results if s.particle_weight_max is not None])
        particle_weight_mean_values = self._finite_values([s.particle_weight_mean for s in step_results if s.particle_weight_mean is not None])
        step_durations = self._finite_values([s.step_duration for s in step_results if s.step_duration is not None])
        
        # Compute proposal metrics
        trans_err_mu_true_values = self._finite_values([s.trans_err_mu_true for s in step_results if s.trans_err_mu_true is not None])
        rot_err_mu_true_values = self._finite_values([s.rot_err_mu_true for s in step_results if s.rot_err_mu_true is not None])
        pose_err_mu_true_values = self._finite_values([s.pose_err_mu_true for s in step_results if s.pose_err_mu_true is not None])
        trans_err_mu_sm_values = self._finite_values([s.trans_err_mu_sm for s in step_results if s.trans_err_mu_sm is not None])
        rot_err_mu_sm_values = self._finite_values([s.rot_err_mu_sm for s in step_results if s.rot_err_mu_sm is not None])
        pose_err_sm_true_values = self._finite_values([s.pose_err_sm_true for s in step_results if s.pose_err_sm_true is not None])
        trans_err_mu_pred_values = self._finite_values([s.trans_err_mu_pred for s in step_results if s.trans_err_mu_pred is not None])
        rot_err_mu_pred_values = self._finite_values([s.rot_err_mu_pred for s in step_results if s.rot_err_mu_pred is not None])
        prop_std_x_values = self._finite_values([s.prop_std_x for s in step_results if s.prop_std_x is not None])
        prop_std_y_values = self._finite_values([s.prop_std_y for s in step_results if s.prop_std_y is not None])
        prop_std_theta_values = self._finite_values([s.prop_std_theta for s in step_results if s.prop_std_theta is not None])
        
        prop_std_xy_values = self._finite_values(
            [
                (s.prop_std_x + s.prop_std_y) / 2.0
                for s in step_results
                if s.prop_std_x is not None
                and s.prop_std_y is not None
                and np.isfinite(s.prop_std_x)
                and np.isfinite(s.prop_std_y)
            ]
        )
        mean_prop_std_xy = float(np.mean(prop_std_xy_values)) if prop_std_xy_values else float("nan")
        mean_std_theta = np.mean(prop_std_theta_values) if prop_std_theta_values else float("nan")
        
        prop_corr_xy_values = self._finite_values([s.corr_xy for s in step_results if s.corr_xy is not None])
        prop_corr_x_theta_values = self._finite_values([s.corr_x_theta for s in step_results if s.corr_x_theta is not None])
        prop_corr_y_theta_values = self._finite_values([s.corr_y_theta for s in step_results if s.corr_y_theta is not None])
        xj_eff_values = self._finite_values([s.xj_eff for s in step_results if s.xj_eff is not None])
        xj_eff_motion_values = self._finite_values([s.xj_eff_motion for s in step_results if s.xj_eff_motion is not None])
        xj_eff_meas_values = self._finite_values([s.xj_eff_meas for s in step_results if s.xj_eff_meas is not None])
        min_xj_pose_err_true_values = self._finite_values([s.min_xj_pose_err_true for s in step_results if s.min_xj_pose_err_true is not None])
        min_xj_is_best_xj_values = [s.min_xj_is_best_xj for s in step_results if s.min_xj_is_best_xj is not None]
        min_xj_true_err_improves_over_sm_true_values = self._finite_values([s.min_xj_true_err_improves_over_sm_true for s in step_results if s.min_xj_true_err_improves_over_sm_true is not None])
        best_xj_true_err_improves_over_sm_true_values = self._finite_values([s.best_xj_true_err_improves_over_sm_true for s in step_results if s.best_xj_true_err_improves_over_sm_true is not None])
        min_xj_true_err_weight_score_values = self._finite_values([s.min_xj_true_err_weight_score for s in step_results if s.min_xj_true_err_weight_score is not None])
        corr_xjs_weights_values = self._finite_values([s.corr_xjs_weights for s in step_results if s.corr_xjs_weights is not None])
        corr_xjs_motion_values = self._finite_values([s.corr_xjs_motion for s in step_results if s.corr_xjs_motion is not None])
        corr_xjs_meas_values = self._finite_values([s.corr_xjs_meas for s in step_results if s.corr_xjs_meas is not None])
        corr_weights_motion_values = self._finite_values([s.corr_weights_motion for s in step_results if s.corr_weights_motion is not None])
        corr_weights_meas_values = self._finite_values([s.corr_weights_meas for s in step_results if s.corr_weights_meas is not None])
        best_xj_score_values = self._finite_values([s.best_xj_score for s in step_results if s.best_xj_score is not None])
        motion_rank_score_values = self._finite_values([s.motion_rank_score for s in step_results if s.motion_rank_score is not None])
        meas_rank_score_values = self._finite_values([s.meas_rank_score for s in step_results if s.meas_rank_score is not None])
        mu_true_err_improves_over_sm_true_values = self._finite_values([s.mu_true_err_improves_over_sm_true for s in step_results if s.mu_true_err_improves_over_sm_true is not None])
        weight_min_xj_err_values = self._finite_values([s.weight_min_xj_err for s in step_results if s.weight_min_xj_err is not None])
        best_weighted_xj_pose_err_true_values = self._finite_values([s.best_weighted_xj_pose_err_true for s in step_results if s.best_weighted_xj_pose_err_true is not None])
        weight_best_xj_values = self._finite_values([s.weight_best_xj for s in step_results if s.weight_best_xj is not None])
        weight_ratio_min_best_weight_values = self._finite_values([s.weight_ratio_min_best_weight for s in step_results if s.weight_ratio_min_best_weight is not None])
        log_motion_range_values = self._finite_values([s.log_motion_range for s in step_results if s.log_motion_range is not None])
        log_meas_range_values = self._finite_values([s.log_meas_range for s in step_results if s.log_meas_range is not None])
        log_weight_range_values = self._finite_values([s.log_weight_range for s in step_results if s.log_weight_range is not None])

        drift_trans_err = float("inf")
        drift_rot_err = float("inf")
        drift_trans_err_raw_odom = float("inf")
        drift_rot_err_raw_odom = float("inf")

        for s in reversed(step_results):
            if s.est_pose is not None:
                drift_trans_err = self.translation_error(s.est_pose, s.true_pose)
                drift_rot_err = abs(self.angle_diff(s.est_pose[2], s.true_pose[2]))
                break

        for s in reversed(step_results):
            if s.raw_odom_pose is not None:
                drift_trans_err_raw_odom = self.translation_error(s.raw_odom_pose, s.true_pose)
                drift_rot_err_raw_odom = abs(self.angle_diff(s.raw_odom_pose[2], s.true_pose[2]))
                break

        summary = {
            "n_steps": len(step_results),
            "scan_match_failed_count": int(scan_match_failed_count),
            "scan_match_fallback_failed_count": int(scan_match_fallback_failed_count),
            "mean_translation_error": float(np.mean(trans_err)) if trans_err else float("inf"),
            "mean_rotation_error": float(np.mean(rot_err)) if rot_err else float("inf"),
            "rmse_translation_error": float(np.sqrt(np.mean(np.square(trans_err)))) if trans_err else float("inf"),
            "rmse_rotation_error": float(np.sqrt(np.mean(np.square(rot_err)))) if rot_err else float("inf"),
            "mean_translation_error_raw_odom": float(np.mean(trans_err_raw_odom)) if trans_err_raw_odom else float("inf"),
            "mean_rotation_error_raw_odom": float(np.mean(rot_err_raw_odom)) if rot_err_raw_odom else float("inf"),
            "rmse_translation_error_raw_odom": float(np.sqrt(np.mean(np.square(trans_err_raw_odom)))) if trans_err_raw_odom else float("inf"),
            "rmse_rotation_error_raw_odom": float(np.sqrt(np.mean(np.square(rot_err_raw_odom)))) if rot_err_raw_odom else float("inf"),
            "mean_trans_err_best_p": float(np.mean(trans_err_best_p)) if trans_err_best_p else float("inf"),
            "mean_rot_err_best_p": float(np.mean(rot_err_best_p)) if rot_err_best_p else float("inf"),
            "rmse_trans_error_best_p": float(np.sqrt(np.mean(np.square(trans_err_best_p)))) if trans_err_best_p else float("inf"),
            "rmse_rot_error_best_p": float(np.sqrt(np.mean(np.square(rot_err_best_p)))) if rot_err_best_p else float("inf"),
            "drift_trans_err": drift_trans_err,
            "drift_rot_err": drift_rot_err,
            "drift_trans_err_raw_odom": drift_trans_err_raw_odom,
            "drift_rot_err_raw_odom": drift_rot_err_raw_odom,
            "mean_neff": float(np.mean(neff_values)) if neff_values else 0.0,
            "mean_particle_weight_min": float(np.mean(particle_weight_min_values)) if particle_weight_min_values else 0.0,
            "mean_particle_weight_max": float(np.mean(particle_weight_max_values)) if particle_weight_max_values else 0.0,
            "mean_particle_weight_mean": float(np.mean(particle_weight_mean_values)) if particle_weight_mean_values else 0.0,
            "mean_step_duration": float(np.mean(step_durations)) if step_durations else float("nan"),
            
            "mean_pose_err_mu_true": float(np.mean(pose_err_mu_true_values)) if pose_err_mu_true_values else float("nan"),
            "mean_min_xj_pose_err_true": float(np.mean(min_xj_pose_err_true_values)) if min_xj_pose_err_true_values else float("nan"),
            "rmse_min_xj_pose_err_true": float(np.sqrt(np.mean(np.square(min_xj_pose_err_true_values)))) if min_xj_pose_err_true_values else float("nan"),
            "mean_min_xj_is_best_xj": float(np.mean(min_xj_is_best_xj_values)) if min_xj_is_best_xj_values else float("nan"),
            "mean_min_xj_true_err_improves_over_sm_true": float(np.mean(min_xj_true_err_improves_over_sm_true_values)) if min_xj_true_err_improves_over_sm_true_values else float("nan"),
            "rmse_min_xj_true_err_improves_over_sm_true": float(np.sqrt(np.mean(np.square(min_xj_true_err_improves_over_sm_true_values)))) if min_xj_true_err_improves_over_sm_true_values else float("nan"),
            "mean_best_xj_true_err_improves_over_sm_true": float(np.mean(best_xj_true_err_improves_over_sm_true_values)) if best_xj_true_err_improves_over_sm_true_values else float("nan"),
            "rmse_best_xj_true_err_improves_over_sm_true": float(np.sqrt(np.mean(np.square(best_xj_true_err_improves_over_sm_true_values)))) if best_xj_true_err_improves_over_sm_true_values else float("nan"),
            "mean_min_xj_true_err_weight_score": float(np.mean(min_xj_true_err_weight_score_values)) if min_xj_true_err_weight_score_values else float("nan"),
            "rmse_min_xj_true_err_weight_score": float(np.sqrt(np.mean(np.square(min_xj_true_err_weight_score_values)))) if min_xj_true_err_weight_score_values else float("nan"),
            "mean_corr_xjs_weights": float(np.mean(corr_xjs_weights_values)) if corr_xjs_weights_values else float("nan"),
            "median_corr_xjs_weights": float(np.median(corr_xjs_weights_values)) if corr_xjs_weights_values else float("nan"),
            "mean_corr_xjs_motion": float(np.mean(corr_xjs_motion_values)) if corr_xjs_motion_values else float("nan"),
            "median_corr_xjs_motion": float(np.median(corr_xjs_motion_values)) if corr_xjs_motion_values else float("nan"),
            "mean_corr_xjs_meas": float(np.mean(corr_xjs_meas_values)) if corr_xjs_meas_values else float("nan"),
            "median_corr_xjs_meas": float(np.median(corr_xjs_meas_values)) if corr_xjs_meas_values else float("nan"),
            "mean_corr_weights_motion": float(np.mean(corr_weights_motion_values)) if corr_weights_motion_values else float("nan"),
            "median_corr_weights_motion": float(np.median(corr_weights_motion_values)) if corr_weights_motion_values else float("nan"),
            "mean_corr_weights_meas": float(np.mean(corr_weights_meas_values)) if corr_weights_meas_values else float("nan"),
            "median_corr_weights_meas": float(np.median(corr_weights_meas_values)) if corr_weights_meas_values else float("nan"),
            "mean_best_xj_score": float(np.mean(best_xj_score_values)) if best_xj_score_values else float("nan"),
            "rmse_best_xj_score": float(np.sqrt(np.mean(np.square(best_xj_score_values)))) if best_xj_score_values else float("nan"),
            "mean_motion_rank_score": float(np.mean(motion_rank_score_values)) if motion_rank_score_values else float("nan"),
            "mean_meas_rank_score": float(np.mean(meas_rank_score_values)) if meas_rank_score_values else float("nan"),
            "mean_mu_true_err_improves_over_sm_true": float(np.mean(mu_true_err_improves_over_sm_true_values)) if mu_true_err_improves_over_sm_true_values else float("nan"),
            "rmse_mu_true_err_improves_over_sm_true": float(np.sqrt(np.mean(np.square(mu_true_err_improves_over_sm_true_values)))) if mu_true_err_improves_over_sm_true_values else float("nan"),

            "mean_weight_min_xj_err": float(np.mean(weight_min_xj_err_values)) if weight_min_xj_err_values else float("nan"),
            "mean_best_weighted_xj_pose_err_true": float(np.mean(best_weighted_xj_pose_err_true_values)) if best_weighted_xj_pose_err_true_values else float("nan"),
            "mean_weight_best_xj": float(np.mean(weight_best_xj_values)) if weight_best_xj_values else float("nan"),
            "mean_weight_ratio_min_best_weight": float(np.mean(weight_ratio_min_best_weight_values)) if weight_ratio_min_best_weight_values else float("nan"),
            "median_weight_ratio_min_best_weight": float(np.median(weight_ratio_min_best_weight_values)) if weight_ratio_min_best_weight_values else float("nan"),
            "mean_log_motion_range": float(np.mean(log_motion_range_values)) if log_motion_range_values else float("nan"),
            "median_log_motion_range": float(np.median(log_motion_range_values)) if log_motion_range_values else float("nan"),
            "mean_log_meas_range": float(np.mean(log_meas_range_values)) if log_meas_range_values else float("nan"),
            "median_log_meas_range": float(np.median(log_meas_range_values)) if log_meas_range_values else float("nan"),
            "mean_log_weight_range": float(np.mean(log_weight_range_values)) if log_weight_range_values else float("nan"),

            "mean_trans_err_mu_true": float(np.mean(trans_err_mu_true_values)) if trans_err_mu_true_values else float("nan"),
            "mean_rot_err_mu_true": float(np.mean(rot_err_mu_true_values)) if rot_err_mu_true_values else float("nan"), 
            "mean_pose_err_sm_true": float(np.mean(pose_err_sm_true_values)) if pose_err_sm_true_values else float("nan"),

            "mean_trans_err_mu_sm": float(np.mean(trans_err_mu_sm_values)) if trans_err_mu_sm_values else float("nan"),
            "mean_rot_err_mu_sm": float(np.mean(rot_err_mu_sm_values)) if rot_err_mu_sm_values else float("nan"),
            
            "rmse_trans_err_mu_sm": float(np.sqrt(np.mean(np.square(trans_err_mu_sm_values)))) if trans_err_mu_sm_values else float("nan"),
            "rmse_rot_err_mu_sm": float(np.sqrt(np.mean(np.square(rot_err_mu_sm_values)))) if rot_err_mu_sm_values else float("nan"),
            "mean_trans_err_mu_pred": float(np.mean(trans_err_mu_pred_values)) if trans_err_mu_pred_values else float("nan"),
            "mean_rot_err_mu_pred": float(np.mean(rot_err_mu_pred_values)) if rot_err_mu_pred_values else float("nan"),
            "rmse_trans_err_mu_pred": float(np.sqrt(np.mean(np.square(trans_err_mu_pred_values)))) if trans_err_mu_pred_values else float("nan"),
            "rmse_rot_err_mu_pred": float(np.sqrt(np.mean(np.square(rot_err_mu_pred_values)))) if rot_err_mu_pred_values else float("nan"),
            
            "mean_prop_std_xy": mean_prop_std_xy,
            "mean_prop_std_theta": mean_std_theta,
            
            "mean_prop_corr_xy": float(np.mean(prop_corr_xy_values)) if prop_corr_xy_values else float("nan"),
            "mean_prop_corr_x_theta": float(np.mean(prop_corr_x_theta_values)) if prop_corr_x_theta_values else float("nan"),
            "mean_prop_corr_y_theta": float(np.mean(prop_corr_y_theta_values)) if prop_corr_y_theta_values else float("nan"),
            "mean_xj_eff": float(np.mean(xj_eff_values)) if xj_eff_values else float("nan"),
            "mean_xj_eff_motion": float(np.mean(xj_eff_motion_values)) if xj_eff_motion_values else float("nan"),
            "mean_xj_eff_meas": float(np.mean(xj_eff_meas_values)) if xj_eff_meas_values else float("nan"),

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