import math

from .evaluator_scanmatching import RunSummaryScanMatching

'''
TODO:

1) Think about useful scoring params
    - Currently we use mean values
    - In rbpf we are using rmse values
    - rmse vals penalize outliers more. Weighting those might prefer more stable version of scan-matching

'''

class ScanMatchingScorer:
    def score(self, summary: RunSummaryScanMatching) -> float:
        """
        Computes a score for scan-matching-only runs.
        Lower is better.
        """
        mean_corr_trans_error = float(summary.mean_corr_trans_error)
        mean_corr_rot_error_rad = float(summary.mean_corr_rot_error)
        mean_corr_rot_error_deg = math.degrees(mean_corr_rot_error_rad)
        final_drift = float(summary.final_drift)
        mean_icp_iterations = float(summary.mean_icp_iterations)
        success_rate = float(
            getattr(summary, "scan_match_success_rate", getattr(summary, "success_rate", 0.0))
        )

        return (
            3.0 * mean_corr_trans_error
            + 1.5 * (mean_corr_rot_error_deg / 180.0)
            + 2.0 * final_drift
            + 0.5 * mean_icp_iterations
            + 2.0 * (1.0 - success_rate)
        )
