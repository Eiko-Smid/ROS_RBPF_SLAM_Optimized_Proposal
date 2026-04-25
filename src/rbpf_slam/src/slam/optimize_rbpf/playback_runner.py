#!/usr/bin/env python3

import time

from .evaluator import RunResult, RBPFEvaluator
from .playback_defs import ExperimentParams
from ..rbpf.rbpf import RBPFFactory
from ..rbpf.scan_match_factory import ScanMatchFactory


class PlaybackRunner:
    """
    Runs one RBPF playback experiment for a single parameter set.
    """

    def __init__(self, factory: RBPFFactory, evaluator: RBPFEvaluator):
        self.factory = factory
        self.evaluator = evaluator

    def run(self, playback_data, params: ExperimentParams) -> RunResult:
        """
        Executes one full RBPF run over all playback steps and returns evaluated results.
        """
        rbpf = self.factory.create(
            scan_match_fac=ScanMatchFactory(),
            particle_params=params.particle_params,
            occ_param=params.occupancy_params,
            sens_params=params.sensor_params,
            map_param=params.map_param,
            icp_params=params.icp_params,
            robot_params=params.robot_params,
            scan_matcher_params=params.scan_matcher_params,
            motion_model_params=params.motion_model_params,
            measurement_model_params=params.measurement_model_params,
        )

        steps = playback_data.step_data_list
        run_result = RunResult(params=params)

        every_nth = max(1, int(params.measurement_model_params.every_nth_scan))

        for step_idx, step in enumerate(steps):
            step_start_time = time.time()

            measurements = step.scan[::every_nth] if every_nth > 1 else step.scan

            neff, scan_match_failed, scan_match_fallback_failed, best_particle_pose = rbpf.step(
                odom=(step.dl, step.dr),
                measurements=measurements,
                proposal_sigma_xy=params.proposal_sigma_xy,
                proposal_sigma_theta=params.proposal_sigma_theta,
                proposal_n_samples=params.proposal_n_samples,
            )

            est_pose = rbpf.weighted_mean_pose()

            step_duration = time.time() - step_start_time

            step_result = self.evaluator.evaluate_step(
                step_idx=step_idx,
                t=step.t,
                true_pose=step.true_pose,
                est_pose=est_pose,
                best_particle_pose=best_particle_pose,
                scan_match_failed=scan_match_failed,
                scan_match_fallback_failed=scan_match_fallback_failed,
                neff=neff,
                step_duration=step_duration,
            )

            run_result.step_results.append(step_result)

        run_result.summary = self.evaluator.summarize_run(
            step_results=run_result.step_results,
            params=params,
        )
        return run_result