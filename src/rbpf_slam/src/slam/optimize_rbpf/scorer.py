import numpy as np


class RunScorer:
    @staticmethod
    def _require(summary, key):
        if key not in summary:
            available = ", ".join(sorted(summary.keys()))
            raise KeyError(f"Missing required summary key '{key}'. Available keys: {available}")

        value = summary[key]
        if value is None:
            raise ValueError(f"Summary key '{key}' is present but None.")

        return float(value)


    def score(self, summary: dict) -> float:
        """
        Computes a single optimization score for one RBPF run.
        Lower score is better.
        """
        # Get step count for normalization 
        n_steps = max(1.0, self._require(summary, "n_steps"))

        # Get number of samples
        n_samples_dir = self._require(summary, "proposal_n_samples")
        n_xj = n_samples_dir ** 3 

        # Extract needed metric from summary
        # Main trajectory quality        
        rmse_trans = self._require(summary, "rmse_translation_error")
        rmse_rot_deg = np.degrees(
            self._require(summary, "rmse_rotation_error")
        )

        # Drift
        drift_trans = self._require(summary, "drift_trans_err")
        drift_rot_deg = np.degrees(
            self._require(summary, "drift_rot_err")
        )

        # Reliability
        sm_fail_rate = self._require(summary, "scan_match_failed_count") / n_steps
        # sm_fallback_fail_rate = self._get(summary, "scan_match_fallback_failed_count", 0.0) / n_steps

        # Proposal correctness
        best_xj_pose_err = self._require(summary, "mean_best_weighted_xj_pose_err_true")
        mu_improve = self._require(summary, "mean_mu_true_err_improves_over_sm_true")
        best_xj_improve = self._require(summary, "mean_best_xj_true_err_improves_over_sm_true")
        min_xj_is_best_xj = self._require(summary, "mean_min_xj_is_best_xj")
        min_xj_is_worse_best_xj = 1.0 - min_xj_is_best_xj

        # Only punish negative improvement.
        mu_worse_penalty = max(0.0, -mu_improve)
        best_xj_worse_penalty = max(0.0, -best_xj_improve)

        # Distribution behavior
        log_meas_range = self._require(summary, "median_log_meas_range")
        log_motion_range = self._require(summary, "median_log_motion_range")
        xj_eff_meas = self._require(summary, "mean_xj_eff_meas")
        meas_flatness = min(xj_eff_meas / n_xj, 1.0)

        # Reward useful measurement sharpness, but saturate it.
        meas_sharpness_reward = min(log_meas_range / 0.5, 1.0)

        return (
            # Trajectory quality
            1.4 * (rmse_trans / 0.20)
            + 0.9 * (rmse_rot_deg / 5.0)

            # Proposal correctness
            + 1.0 * (best_xj_pose_err / 0.20)
            + 1.2 * mu_worse_penalty
            + 1.5 * best_xj_worse_penalty
            + 1.5 * min_xj_is_worse_best_xj

            # Measurement/motion distribution behavior
            - 0.25 * meas_sharpness_reward
            + 0.1 * min(log_motion_range / 1.0, 2.0)
            + 0.2 * meas_flatness

            # Reliability
            + 2.0 * sm_fail_rate
            # + 4.0 * sm_fallback_fail_rate

            # Weak drift 
            + 0.15 * (drift_trans / 0.30)
            + 0.10 * (drift_rot_deg / 8.0)
        )