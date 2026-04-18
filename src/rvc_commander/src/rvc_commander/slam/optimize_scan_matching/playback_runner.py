#!/usr/bin/env python3
import time

from .factory import ScanMatcherFactory
from .evaluator import RunResult, ScanMatcherEvaluator

from ..scan_match_playback_def import PlaybackData, ExperimentParams


class PlaybackRunner:
    '''
    Class for running a single playback of the scan matching process, given the playback data and experiment parameters.
    Infrastructure for running the scan matcher, evaluating the results and storing them into a RunResult object.
    '''
    def __init__(self, factory: ScanMatcherFactory, evaluator: ScanMatcherEvaluator):
        '''
        Initializes the PlaybackRunner with a scan matcher factory and an evaluator.
        '''
        self.factory = factory
        self.evaluator = evaluator


    def run(self, playback_data: PlaybackData, params: ExperimentParams) -> RunResult:
        '''
        Runs a single playback of the scan matching process using the provided playback data and experiment parameters.

        Parameters
        ---------
        playback_data: PlaybackData
            The data for the playback, including the steps and measurements.
        params: ExperimentParams
            The parameters for the experiment, including the scan matcher parameters and other settings.
        
        Returns
        ---------
        RunResult
            An object containing the results of the run, including the step results and a summary of the run.
        '''
        # Instantiate the scan matcher using the factory and the given data
        scan_matcher = self.factory.build(playback_data, params)

        # Extracts the steps (inputs) from the playback data
        steps = playback_data.step_data_list

        # Instantiate the RunResult object that will hold the results of this run
        run_result = RunResult(params=params)
        old_pose = steps[0].true_pose
        
        # Loop through the steps containing the inputs for the entire run
        for step_idx, step in enumerate(steps):
            # Measure step start time
            step_start_time = time.time()

            # Runs scan matching process -> pred and corrected pose for given inputs
            corr_pose, pred_pose = scan_matcher.update_pose(
                old_pose=old_pose,
                dl=step.dl,
                dr=step.dr,
                measurements=step.scan,
            )

            # Measure step end time and compute duration
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time

            used_fallback = corr_pose is None
            # Determines the final corrected pose to use for evaluation
            final_corr_pose = pred_pose if corr_pose is None else corr_pose

            # Evaluates the scan matching results 
            step_result = self.evaluator.evaluate_step(
                step_idx=step_idx,
                t=step.t,
                true_pose=step.true_pose,
                pred_pose=pred_pose,
                corr_pose=final_corr_pose,
                icp_info=scan_matcher.get_info(),
                used_fallback_prediction=used_fallback,
                step_duration=step_duration
            )

            # Stores results into step_results list
            run_result.step_results.append(step_result)
            # Update old pose
            old_pose = final_corr_pose

        
        run_result.summary = self.evaluator.summarize_run(run_result.step_results)
        return run_result