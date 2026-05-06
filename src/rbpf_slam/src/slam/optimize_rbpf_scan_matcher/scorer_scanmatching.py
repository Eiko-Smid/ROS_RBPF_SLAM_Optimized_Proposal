from typing import Mapping
import math


class ScanMatchingScorer:
    def score(self, summary: Mapping[str, float]) -> float:
        """
        Computes a score for scan-matching-only runs.
        Lower is better.
        """
        mean_corr_trans_error = float(summary["mean_corr_trans_error"])
        mean_corr_rot_error_rad = float(summary["mean_corr_rot_error"])
        mean_corr_rot_error_deg = math.degrees(mean_corr_rot_error_rad)
        final_drift = float(summary["final_drift"])
        mean_icp_iterations = float(summary["mean_icp_iterations"])
        success_rate = float(summary["success_rate"])

        return (
            3.0 * mean_corr_trans_error
            + 1.5 * (mean_corr_rot_error_deg / 180.0)
            + 2.0 * final_drift
            + 0.5 * mean_icp_iterations
            + 2.0 * (1.0 - success_rate)
        )
