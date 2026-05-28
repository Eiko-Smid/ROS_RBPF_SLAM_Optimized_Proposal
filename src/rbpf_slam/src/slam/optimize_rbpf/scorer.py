import numpy as np

# class RunScorer:
#     def score(self, summary: dict) -> float:
#         """
#         Computes a single optimization score for one RBPF run.

#         Lower score is better.
#         """
#         return (
#             1.5 * summary["rmse_translation_error"]
#             + 0.7 * summary["rmse_rotation_error"]
#             + 1.0 * summary["drift_trans_err"]
#             + 0.5 * summary["mean_step_duration"]   # TODO: Should not be part go optimization pipeline -> delete!
#             - 0.3 * summary["mean_neff"]            # Neff value depends on number of particles. Don't weight directly!
#         )



class RunScorer:
    def _get(self, summary, key, default=float("nan")):
        value = summary.get(key, default)
        if value is None:
            return default
        return float(value)


    def score(self, summary: dict) -> float:
        """
        Computes a single optimization score for one RBPF run.
        Lower score is better.
        """

        n_steps = max(1.0, self._get(summary, "n_steps", 1.0))

        # Extract needed metric from summary
        # Main trajectory quality        
        rmse_trans = self._get(summary, "rmse_translation_error")
        rmse_rot_deg = np.degrees(
            self._get(summary, "rmse_rotation_error")
        )

        # Drift
        drift_trans = self._get(summary, "drift_trans_err")
        drift_rot_deg = np.degrees(
            self._get(summary, "drift_rot_err")
        )

        # Reliability
        sm_fail_rate = self._get(summary, "scan_match_failed_count", 0.0) / n_steps
        sm_fallback_fail_rate = self._get(summary, "scan_match_fallback_failed_count", 0.0) / n_steps

        # Proposal correctness
        best_xj_pose_err = self._get(summary, "mean_best_weighted_xj_pose_err_true")
        mu_improve = self._get(summary, "mean_mu_true_err_improves_over_sm_true", -1.0)
        best_xj_improve = self._get(summary, "mean_best_xj_true_err_improves_over_sm_true", -1.0)
        min_xj_is_best_xj = self._get(summary, "mean_min_xj_is_best_xj", 0.0)
        min_xj_is_worse_best_xj = 1.0 - min_xj_is_best_xj

        # Only punish negative improvement.
        mu_worse_penalty = max(0.0, -mu_improve)
        best_xj_worse_penalty = max(0.0, -best_xj_improve)

        # Distribution behavior
        log_meas_range = self._get(summary, "median_log_meas_range", 0.0)
        log_motion_range = self._get(summary, "median_log_motion_range", 0.0)
        xj_eff_meas = self._get(summary, "mean_xj_eff_meas", 27.0)
        n_xj = 27.0
        meas_flatness = min(xj_eff_meas / n_xj, 1.0)

        # Reward useful measurement sharpness, but saturate it.
        meas_sharpness_reward = min(log_meas_range / 1.0, 1.0)

        return (
            # Trajectory quality
            1.4 * (rmse_trans / 0.20)
            + 0.9 * (rmse_rot_deg / 5.0)

            # Proposal correctness
            + 1.0 * (best_xj_pose_err / 0.20)
            + 1.2 * mu_worse_penalty
            + 1.0 * best_xj_worse_penalty
            + 1.2 * min_xj_is_worse_best_xj

            # Measurement/motion distribution behavior
            - 0.4 * meas_sharpness_reward
            + 0.3 * min(log_motion_range / 1.0, 2.0)
            + 0.4 * meas_flatness

            # Reliability
            + 2.0 * sm_fail_rate
            # + 4.0 * sm_fallback_fail_rate

            # Weak drift 
            + 0.15 * (drift_trans / 0.30)
            + 0.10 * (drift_rot_deg / 8.0)
        )