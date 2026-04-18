#!/usr/bin/env python3

class RunScorer:
    def score(self, summary: dict) -> float:
        '''
        Weights the summary statistics for a run to compute a single score that can be used to rank different runs. 
        The lower the score, the better the run. The weights for each statistic can be adjusted to prioritize certain
        aspects of the run over others. 

        Parameters
        ----------
        summary: dict
            a dictionary containing the summary statistics for a run, including:
            - rmse_translation_error: the root mean square error of the translation error across all steps
            - mean_rotation_error: the mean rotation error across all steps
            - mean_icp_iterations: the mean number of ICP iterations across all steps
            - fallback_count: the total number of steps where a fallback prediction was used instead of a corrected pose
        
        Returns
        -------
        float
            a single score representing the overall performance of the run, where a lower score indicates a better run
        '''
        return (
            1.0 * summary["rmse_translation_error"]
            + 0.5 * summary["mean_rotation_error"]
            + 0.5 * summary["mean_step_duration"]
            + 0.2 * summary["mean_icp_iterations"]
            + 2.0 * summary["fallback_count"]
        )