#!/usr/bin/env python3

import time
import numpy as np

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

    @staticmethod
    def _aggregate_icp_counters(rbpf) -> dict:
        '''
        Aggregates ICP-related counters from all particles in the RBPF instance and returns a dictionary with 
        accumalated count values for each counter.
        '''
        counter_keys = [
            "count_too_few_points",
            "count_too_few_corresp",
            "infinite_h_or_g",
            "ill_cond_H",
            "infinite_dtransform",
            "infinite_mean_err",
            "best_transf_too_large",
            "best_mean_err_too_large",
        ]

        totals = {key: 0 for key in counter_keys}

        for particle in getattr(rbpf, "particles", []):
            icp_info = particle.scan_matcher.icp.get_info()
            for key in counter_keys:
                totals[key] += int(icp_info.get(key, 0) or 0)

        return totals


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

        # Ensure valid scan downsampling values for filter/proposal and map update.
        every_nth_filter = max(1, int(params.every_nth_scan_filter))
        every_nth_map = max(1, int(params.every_nth_scan_map))
        print(
            f"Running RBPF with params: {params.tag} "
            f"(every_nth_scan_filter={every_nth_filter}, every_nth_scan_map={every_nth_map})"
        )

        for step_idx, step in enumerate(steps):
            step_start_time = time.time()

            # Subsample and clean measurements
            measurements_proposal = (
                step.scan[::every_nth_filter] if every_nth_filter > 1 else step.scan
            )
            measurements_map = step.scan[::every_nth_map] if every_nth_map > 1 else step.scan

            measurements_proposal = [
                (r, b) for r, b in measurements_proposal if np.isfinite(r) and not np.isnan(r)
            ]
            measurements_map = [
                (r, b) for r, b in measurements_map if np.isfinite(r) and not np.isnan(r)
            ]

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
            info = rbpf.get_step_info()
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
        run_result.summary.update(self._aggregate_icp_counters(rbpf))
        timing_summary = rbpf.timing_summary()
        run_result.summary.update(timing_summary)

        def _to_ms(value):
            return value * 1000.0 if value is not None else None

        print("RBPF timing summary (mean per run):")
        print(f"  update_particles: {_to_ms(timing_summary.get('mean_timing_update_particles_s'))} ms")
        print(f"  normalize+neff: {_to_ms(timing_summary.get('mean_timing_normalize_neff_s'))} ms")
        print(f"  metrics: {_to_ms(timing_summary.get('mean_timing_metrics_s'))} ms")
        print(f"  resampling (when triggered): {_to_ms(timing_summary.get('mean_timing_resampling_s'))} ms")
        print("  update_particle internals:")
        print(
            f"    scan_match.update_pose: {_to_ms(timing_summary.get('mean_timing_scan_match_update_pose_s'))} ms "
            f"(count={timing_summary.get('timing_scan_match_update_pose_count')})"
        )
        print(
            f"    proposal.estimate_proposal: {_to_ms(timing_summary.get('mean_timing_proposal_estimation_s'))} ms "
            f"(count={timing_summary.get('timing_proposal_estimation_count')})"
        )
        print(
            f"    scan_match fallback block: {_to_ms(timing_summary.get('mean_timing_scan_match_fallback_s'))} ms "
            f"(count={timing_summary.get('timing_scan_match_fallback_count')})"
        )
        print(
            f"    map_extension_if_necessary loop: {_to_ms(timing_summary.get('mean_timing_map_extension_s'))} ms "
            f"(count={timing_summary.get('timing_map_extension_count')})"
        )
        print(
            f"    ogm.update_map: {_to_ms(timing_summary.get('mean_timing_map_update_s'))} ms "
            f"(count={timing_summary.get('timing_map_update_count')})"
        )

        return run_result