import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RunScorer:
    """
    Scores one RBPF multi-particle summary.

    Low score  -> good parameter set
    High score -> bad parameter set

    This first scorer version intentionally uses only a compact core set
    of metrics. Other evaluator metrics should stay diagnostic for now.
    """
    # Scales to make metrics dimensionless and comparable
    # Main pose scales
    trans_scale: float = 0.10
    rot_scale: float = math.radians(3.0)

    # Smoothness / drift-growth scales
    trans_motion_scale: float = 0.02
    rot_motion_scale: float = math.radians(1.0)
    trans_slope_scale: float = 0.003
    rot_slope_scale: float = math.radians(0.15)

    # Runtime soft scales
    step_duration_scale: float = 0.10
    scan_matching_time_scale: float = 0.05

    # SLAM should improve over odom by at least this much (10%)
    improvement_scale: float = 0.10

    # Generic caps
    max_scaled_badness: float = 5.0
    missing_core_penalty: float = 5.0
    missing_optional_penalty: float = 0.0
    invalid_rate_penalty: float = 5.0

    # Count of metrics used in scoring
    n_scorer_metrics: int = 0  


    @staticmethod
    def _get(summary: Dict, key: str) -> Optional[float]:
        """
        Safely extract one finite float from summary.
        """
        value = summary.get(key, None)

        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(value):
            return None

        return value


    def _add_scaled(
        self,
        summary: Dict,
        key: str,
        scale: float,
        weight: float,
        use_std_penalty: bool = False,
    ) -> float:
        '''
        Extracts the given metric from the summary, scales the given value -> dimensionless, and applies the given weight to
        compute the score contribution.
        '''
        # Increment metric counter
        self.n_scorer_metrics += 1

        # Get metric
        metric = self._get(summary, key)

        # Define penalty for missing metrics
        penalty = self.missing_core_penalty if use_std_penalty else self.missing_optional_penalty
        
        # Penalize missing metrics
        if metric is None:
            return weight * penalty

        # Ensure valid scale
        if scale <= 0.0:
            raise ValueError("scale must be > 0")

        # Define 
        metric = max(0.0, metric)
        return weight * min(metric / scale, self.max_scaled_badness)


    def _add_rate(
        self,
        summary: Dict,
        key: str,
        weight: float,
        use_std_penalty: bool = False,
    ) -> float:
        # Increment metric counter
        self.n_scorer_metrics += 1        

        # Get metric
        metric = self._get(summary, key)

        # Define penalty for missing metrics
        penalty = self.missing_core_penalty if use_std_penalty else self.missing_optional_penalty
        
        # Penalize missing metrics
        if metric is None:
            return weight * penalty

        # Penalize invalid rates
        if metric < 0.0 or metric > 1.0:
            raise ValueError(f"[Scorer] Rate metric '{key}' must be in range [0.0, 1.0], but got {metric}")
        
        return weight * metric


    def _add_correlation(
        self,
        summary: Dict,
        key: str,
        weight: float,
        use_std_penalty: bool = False,
    ) -> float:
        # Increment metric counter
        self.n_scorer_metrics += 1        

        # Get metric
        metric = self._get(summary, key)

        # Define penalty for missing metrics
        penalty = self.missing_core_penalty if use_std_penalty else self.missing_optional_penalty

        # Penalize missing metrics
        if metric is None:
            return weight * penalty

        return weight * min(max(metric, 0.0), 2.0)


    def _add_improvement(
        self,
        summary: Dict,
        key: str,
        weight: float,
        use_std_penalty: bool = False,
    ) -> float:
        # Increment metric counter
        self.n_scorer_metrics += 1        

        # Get metric
        metric = self._get(summary, key)

        # Define penalty for missing metrics
        penalty = self.missing_core_penalty if use_std_penalty else self.missing_optional_penalty

        # Penalize missing metrics
        if metric is None:
            return weight * penalty

        badness = max(self.improvement_scale - metric, 0.0) / self.improvement_scale
        return weight * min(badness, self.max_scaled_badness)


    def score(self, summary: Dict) -> float:
        """
        Compute final score.

        Parameters
        ----------
        summary : dict
            Summary dictionary from evaluator.

        Returns
        -------
        float
            Lower is better.
        """
        # Check for valid summary
        if summary is None or not isinstance(summary, dict):
            return float("inf")

        # Init vars
        score = 0.0
        self.n_scorer_metrics = 0

        # ------------------------------------------------------------
        # 1. Final MAP trajectory quality
        # Most important part: final map quality.
        # ------------------------------------------------------------
        score += self._add_scaled(
            summary, "rmse_trans_err_map_traj",
            self.trans_scale, weight=8.0, use_std_penalty=True,
        )
        score += self._add_scaled(
            summary, "rmse_rot_err_map_traj",
            self.rot_scale, weight=8.0, use_std_penalty=True,
        )
        score += self._add_scaled(
            summary, "final_trans_drift_trans_err_map_traj",
            self.trans_scale, weight=5.0, use_std_penalty=True,
        )
        score += self._add_scaled(
            summary, "final_rot_drift_rot_err_map_traj",
            self.rot_scale, weight=5.0, use_std_penalty=True,
        )
        score += self._add_rate(
            summary, "rate_above_thres_trans_err_map_traj",
            weight=5.0, use_std_penalty=True,
        )
        score += self._add_rate(
            summary, "rate_above_thres_rot_err_map_traj",
            weight=5.0, use_std_penalty=True,
        )

        # ------------------------------------------------------------
        # 2. Improvement over raw odometry
        # SLAM must be better than odom.
        # ------------------------------------------------------------
        score += self._add_improvement(
            summary, "median_trans_err_map_traj_impr_over_raw_odom",
            weight=4.0, use_std_penalty=True,
        )
        score += self._add_improvement(
            summary, "median_rot_err_map_traj_impr_over_raw_odom",
            weight=4.0, use_std_penalty=True,
        )

        # ------------------------------------------------------------
        # 3. Drift growth and smoothness
        # Avoid trajectories that slowly diverge or jump.
        # ------------------------------------------------------------
        score += self._add_scaled(
            summary, "p90_pos_trans_err_slopes_map_traj",
            self.trans_slope_scale, weight=3.0,
        )
        score += self._add_scaled(
            summary, "p90_pos_rot_err_slopes_map_traj",
            self.rot_slope_scale, weight=3.0,
        )
        score += self._add_scaled(
            summary, "p90_trans_motion_err_map_traj",
            self.trans_motion_scale, weight=3.0,
        )
        score += self._add_scaled(
            summary, "p90_rot_motion_err_map_traj",
            self.rot_motion_scale, weight=3.0,
        )

        # ------------------------------------------------------------
        # 4. Online weighted mean estimate
        # Useful for robot control/localization during the run.
        # ------------------------------------------------------------
        score += self._add_scaled(
            summary, "rmse_trans_err_weighted_mean",
            self.trans_scale, weight=3.0,
        )
        score += self._add_scaled(
            summary, "rmse_rot_err_weighted_mean",
            self.rot_scale, weight=3.0,
        )

        # ------------------------------------------------------------
        # 5. Particle selection / measurement model quality
        # The weights should prefer good particles.
        # ------------------------------------------------------------
        score += self._add_correlation(
            summary, "median_corr_trans_weights_pos",
            weight=3.0,
        )
        score += self._add_correlation(
            summary, "median_corr_rot_weights_pos",
            weight=3.0,
        )
        score += self._add_rate(
            summary, "rate_gap_trans_best_to_min_before_resamp_above_max_trans_gap",
            weight=3.0,
        )
        score += self._add_rate(
            summary, "rate_gap_rot_best_to_min_before_resamp_above_max_rot_gap",
            weight=3.0,
        )

        # ------------------------------------------------------------
        # 6. Robustness
        # Scan-matching failures are dangerous.
        # ------------------------------------------------------------
        score += self._add_rate(
            summary, "scan_match_failed_rate",
            weight=8.0,
        )
        score += self._add_rate(
            summary, "scan_match_fallback_failed_rate",
            weight=12.0,
        )

        # ------------------------------------------------------------
        # 7. Runtime soft penalties
        # Low weight for now.
        # ------------------------------------------------------------
        # score += self._add_scale(
        #     summary, "mean_step_duration",
        #     self.step_duration_target, weight=1.0,
        # )
        # score += self._add_scale(
        #     summary, "mean_time_duration_scan_matching",
        #     self.scan_matching_time_target, weight=1.0,
        # )

        return float(score)



def test():
    metric = -0.5
    metric = max(0.0, metric)

    print(f"Metric: {metric}")



def main():
    test()


if __name__ == "__main__":
    main()