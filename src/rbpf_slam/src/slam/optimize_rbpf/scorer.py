

class RunScorer:
    def score(self, summary: dict) -> float:
        """
        Computes a single optimization score for one RBPF run.

        Lower score is better.
        """
        return (
            1.5 * summary["rmse_translation_error"]
            + 0.7 * summary["rmse_rotation_error"]
            + 1.0 * summary["drift_trans_err"]
            + 0.5 * summary["mean_step_duration"]   # TODO: Should not be part go optimization pipeline -> delete!
            - 0.3 * summary["mean_neff"]            # Neff value depends on number of particles. Don't weight directly!
        )