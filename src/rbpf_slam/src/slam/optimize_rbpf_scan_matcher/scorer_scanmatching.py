import math
from typing import Tuple

from .evaluator_scanmatching import RunSummaryScanMatching

'''
TODO:

1) Think about useful scoring params
    - Currently we use mean values
    - In rbpf we are using rmse values
    - rmse vals penalize outliers more. Weighting those might prefer more stable version of scan-matching

'''

# class ScanMatchingScorer:
#     def score(self, summary: RunSummaryScanMatching) -> float:
#         """
#         Computes a score for scan-matching-only runs.
#         Lower is better.
#         """
#         mean_corr_trans_error = float(summary.mean_corr_trans_err)
#         mean_corr_rot_error_rad = float(summary.mean_corr_rot_err)
#         mean_corr_rot_error_deg = math.degrees(mean_corr_rot_error_rad)
#         final_drift_trans = float(summary.final_drift_trans)
#         mean_icp_iterations = float(summary.mean_icp_iterations)
#         success_rate = float(
#             getattr(summary, "scan_match_success_rate", getattr(summary, "success_rate", 0.0))
#         )

#         return (
#             3.0 * mean_corr_trans_error
#             + 1.5 * (mean_corr_rot_error_deg / 180.0)
#             + 2.0 * final_drift_trans
#             + 0.5 * mean_icp_iterations
#             + 2.0 * (1.0 - success_rate)
#         )


class ScanMatchingScorer:
    @staticmethod
    def _require_metric(summary: RunSummaryScanMatching, name: str, allow_nan: bool = False) -> float:
        if not hasattr(summary, name):
            raise ValueError(f"Missing required score input '{name}'")

        value = getattr(summary, name)
        if value is None:
            raise ValueError(f"Missing required score input '{name}'")

        value_f = float(value)
        if not allow_nan and math.isnan(value_f):
            raise ValueError(f"Required score input '{name}' is NaN")
        if math.isinf(value_f):
            raise ValueError(f"Required score input '{name}' is infinite")
        return value_f


    @staticmethod
    def _require_metric_any(summary: RunSummaryScanMatching, names: Tuple[str, ...]) -> float:
        for name in names:
            if hasattr(summary, name):
                value = getattr(summary, name)
                if value is None:
                    continue
                value_f = float(value)
                if math.isnan(value_f):
                    raise ValueError(f"Required score input '{name}' is NaN")
                if math.isinf(value_f):
                    raise ValueError(f"Required score input '{name}' is infinite")
                return value_f

        joined = ", ".join(names)
        raise ValueError(f"Missing required score input; expected one of: {joined}")


    def score(self, summary: RunSummaryScanMatching) -> float:
        """
        Computes a score for scan-matching-only runs.
        Lower is better.
        """
        rmse_corr_trans_err = self._require_metric(summary, "rmse_corr_trans_err")
        rmse_corr_rot_err_deg = math.degrees(self._require_metric(summary, "rmse_corr_rot_err"))

        max_corr_trans_err = self._require_metric(summary, "max_corr_trans_err")
        max_corr_rot_err_deg = math.degrees(self._require_metric(summary, "max_corr_rot_err"))
        
        perc_95_corr_trans_err = self._require_metric(summary, "perc_95_corr_trans_err")
        perc_95_corr_rot_err_deg = math.degrees(self._require_metric(summary, "perc_95_corr_rot_err"))
        
        max_rolling_rmse_corr_trans_error = self._require_metric(
            summary,
            "max_rolling_rmse_corr_trans_error",
            allow_nan=True,
        )
        max_rolling_rmse_corr_rot_error_deg = math.degrees(
            self._require_metric(
                summary,
                "max_rolling_rmse_corr_rot_error",
                allow_nan=True,
            )
        )

        # Ignore rolling terms when they are undefined (NaN), e.g., window larger than run length.
        rolling_trans_term = (
            0.0
            if math.isnan(max_rolling_rmse_corr_trans_error)
            else 0.9 * max_rolling_rmse_corr_trans_error / 0.5
        )
        rolling_rot_term = (
            0.0
            if math.isnan(max_rolling_rmse_corr_rot_error_deg)
            else 0.6 * (max_rolling_rmse_corr_rot_error_deg / 7.0)
        )

        scan_match_success_rate = self._require_metric_any(summary, ("scan_match_success_rate", "success_rate"))
        scan_match_failed_rate = 1 - scan_match_success_rate

        corr_worse_rate_trans = self._require_metric(summary, "corr_worse_rate_trans")
        corr_worse_rate_rot = self._require_metric(summary, "corr_worse_rate_rot")

        final_drift_trans = self._require_metric(summary, "final_drift_trans")
        final_drift_rot_deg = math.degrees(self._require_metric(summary, "final_drift_rot"))

        allowed_fail_rate_icp = 0.03   
        icp_failure_term = (
            1.0 * scan_match_failed_rate
            + 8.0 * max(0.0, scan_match_failed_rate - allowed_fail_rate_icp)
        )

        return (
            # Overall trajectory quality
            1.0 * rmse_corr_trans_err / 0.2
            + 0.7 * (rmse_corr_rot_err_deg / 5.0)

            # Single-step SLAM safety, clipped
            + 0.8 * min(max_corr_trans_err / 0.5, 3.0)
            + 0.5 * min(max_corr_rot_err_deg / 15.0, 3.0)

            # High-error behavior
            + 0.8 * perc_95_corr_trans_err / 0.3
            + 0.5 * (perc_95_corr_rot_err_deg / 8.0)

            # Worst local unstable phase
            + rolling_trans_term
            + rolling_rot_term

            # Reliability
            + icp_failure_term

            # Did scan matching improve the prediction?
            + 0.1 * corr_worse_rate_trans
            + 0.05 * corr_worse_rate_rot

            # Weak tie-breaker
            + 0.1 * final_drift_trans / 0.2
            + 0.05 * (final_drift_rot_deg / 5.0)
        )
