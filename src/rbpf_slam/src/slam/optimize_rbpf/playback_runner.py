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
        # Create rbpf instance for the current parameter set
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

        # Ensure valid nth scan value
        every_nth = max(1, int(params.every_nth_scan))
        print(f"Running RBPF with params: {params.tag} (every_nth_scan={every_nth})")

        for step_idx, step in enumerate(steps):
            step_start_time = time.time()

            # Subsample measurements
            measurements_map = step.scan
            measurements_proposal = step.scan[::every_nth] if every_nth > 1 else step.scan
            print("Scans used for current step:", len(measurements_proposal), "out of", len(measurements_map))

            # Run rbpf filter step
            rbpf.step(
                odom=(step.dl, step.dr),
                measurements_proposal=measurements_proposal,
                measurements_map_update=measurements_map,
                true_pose=step.true_pose,
                proposal_sigma_xy=params.proposal_sigma_xy,
                proposal_sigma_theta=params.proposal_sigma_theta,
                proposal_n_samples=params.proposal_n_samples,
            )

            # Measure step duration
            step_duration = time.time() - step_start_time
            
            # Extract evaluation info from the RBPF instance
            info = rbpf.step_info()
            step_idx_logged = info.get("step")
            true_pose_logged = info.get("true_pose")
            est_pose = info.get("weighted_mean_pose")
            best_particle_pose = info.get("best_particle_pose")
            neff = info.get("neff")
            scan_match_failed = info.get("scan_match_failed_any")
            scan_match_fallback_failed = info.get("scan_match_fallback_failed_any")
            particle_weight_min = info.get("particle_weight_min")
            particle_weight_max = info.get("particle_weight_max")
            particle_weight_mean = info.get("particle_weight_mean")

            # Evaluate the current step and store results
            step_result = self.evaluator.evaluate_step(
                step_idx=step_idx_logged if step_idx_logged is not None else step_idx,
                t=step.t,
                true_pose=true_pose_logged if true_pose_logged is not None else step.true_pose,
                est_pose=est_pose,
                best_particle_pose=best_particle_pose,
                scan_match_failed=scan_match_failed,
                scan_match_fallback_failed=scan_match_fallback_failed,
                neff=neff,
                particle_weight_min=particle_weight_min,
                particle_weight_max=particle_weight_max,
                particle_weight_mean=particle_weight_mean,
                step_duration=step_duration,
            )

            run_result.step_results.append(step_result)

        # Summarize the run results and store in the run result object
        run_result.summary = self.evaluator.summarize_run(
            step_results=run_result.step_results,
            params=params,
        )
        return run_result