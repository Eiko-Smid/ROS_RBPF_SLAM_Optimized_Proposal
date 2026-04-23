

class RunScorer:
    def score(self, summary: dict) -> float:
        """
        Computes a single optimization score for one RBPF run.

        Lower score is better.
        """
        return (
            1.5 * summary["rmse_translation_error"]
            + 0.7 * summary["rmse_rotation_error"]
            + 1.0 * summary["drift"]
            + 0.5 * summary["mean_step_duration"]
            - 0.3 * summary["mean_neff"]
        )