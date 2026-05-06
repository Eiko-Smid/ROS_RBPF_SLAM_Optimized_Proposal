from typing import Mapping


class ScanMatchingScorer:
    def score(self, summary: Mapping[str, float]) -> float:
        """
        Computes a score for scan-matching-only runs.
        Lower is better.
        """
        return (
            1.5 * summary["rmse_translation_error"]
            + 0.7 * summary["rmse_rotation_error"]
            + 0.5 * summary["mean_translation_error"]
            + 0.3 * summary["mean_rotation_error"]
        )
