#!/usr/bin/env python3

class RunScorer:
    def score(self, summary: dict) -> float:
        return (
            1.0 * summary["rmse_translation_error"]
            + 0.3 * summary["mean_rotation_error"]
            + 0.2 * summary["mean_icp_iterations"]
            + 2.0 * summary["fallback_count"]
        )