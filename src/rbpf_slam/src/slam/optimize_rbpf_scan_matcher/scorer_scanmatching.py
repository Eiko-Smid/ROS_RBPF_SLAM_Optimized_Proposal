import math

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
    def score(self, summary: RunSummaryScanMatching) -> float:
        """
        Computes a score for scan-matching-only runs.
        Lower is better.
        """
        rmse_corr_trans_err = float(summary.rmse_corr_trans_err) 
        rmse_corr_rot_err_deg = math.degrees(float(summary.rmse_corr_rot_err)) 

        max_corr_trans_err = float(summary.max_corr_trans_err)
        max_corr_rot_err_deg = math.degrees(float(summary.max_corr_rot_err))
        
        perc_95_corr_trans_err = float(summary.perc_95_corr_trans_err)
        perc_95_corr_rot_err_deg = math.degrees(float(summary.perc_95_corr_rot_err))
        
        max_rolling_rmse_corr_trans_error = float(summary.max_rolling_rmse_corr_trans_error)
        max_rolling_rmse_corr_rot_error_deg = math.degrees(float(summary.max_rolling_rmse_corr_rot_error))

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

        scan_match_failed_rate = 1 - float(
            getattr(summary, "scan_match_success_rate", getattr(summary, "success_rate", 0.0))
        )

        corr_worse_rate_trans = float(summary.corr_worse_rate_trans)
        corr_worse_rate_rot = float(summary.corr_worse_rate_rot)

        final_drift_trans = float(summary.final_drift_trans)
        final_drift_rot_deg = math.degrees(float(summary.final_drift_rot))    

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
            + 0.4 * scan_match_failed_rate

            # Did scan matching improve the prediction?
            + 0.2 * corr_worse_rate_trans
            + 0.1 * corr_worse_rate_rot

            # Weak tie-breaker
            + 0.1 * final_drift_trans / 0.2
            + 0.05 * (final_drift_rot_deg / 5.0)
        )
