#!/usr/bin/env python3

import time

import numpy as np

from .playback_defs import PlaybackData, ExperimentParams
from .evaluator import RunResult, RBPFEvaluator
from ..rbpf.rbpf import RBPFFactory
from ..rbpf.raw_odom_estimator import RawOdomEstimator
from ..rbpf.scan_match_factory import ScanMatchFactory
from ..rbpf.particle_process_pool import ParticleProcessPool #_init_worker


class PlaybackRunner:
    """
    Runs one RBPF playback experiment for a single parameter set.
    """

    def __init__(
        self,
        factory: RBPFFactory,
        evaluator: RBPFEvaluator,
    ):
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


    def run(self, playback_data: PlaybackData, params: ExperimentParams) -> RunResult:
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
            neff_threshold=params.neff_threshold,
        )

        steps = playback_data.step_data_list
        run_result = RunResult(params=params)

        raw_odom_estimator = RawOdomEstimator(
            motion_model=rbpf.motion_model,
            start_pose=params.particle_params.start_pose,
        )

        # Ensure valid scan downsampling values for filter/proposal and map update.
        every_nth_filter = max(1, int(params.every_nth_scan_filter))
        every_nth_map = max(1, int(params.every_nth_scan_map))
        # print(
        #     f"Running RBPF with params: {params.tag} "
        #     f"(every_nth_scan_filter={every_nth_filter}, every_nth_scan_map={every_nth_map})"
        # )        
        
        for step_idx, step in enumerate(steps):
            step_start_time = time.time()
            step_duration = None

            raw_odom_pose = raw_odom_estimator.predict_pose(
                dl=step.dl,
                dr=step.dr,
            )

            # Subsample and clean measurements
            measurements_proposal = (
                step.scan[::every_nth_filter] if every_nth_filter > 1 else step.scan
            )
            
            measurements_proposal = [
                (r, b) for r, b in measurements_proposal if np.isfinite(r) and not np.isnan(r)
            ]

            measurements_map = step.scan[::every_nth_map] if every_nth_map > 1 else step.scan

            if np.isnan(measurements_proposal).any():
                print("\nPlayback runner: measurement model contains nan value after subsampling scans")

            # Use inf vals for map update, too -> clear free space faster
            # measurements_map = [
            #     (r, b) for r, b in measurements_map if np.isfinite(r) and not np.isnan(r)
            # ]

            # if step_idx == 517:
            #     print("Debug here")

            # Run rbpf filter step
            rbpf.step(
                odom=(step.dl, step.dr),
                measurements_proposal=measurements_proposal,
                measurements_map_update=measurements_map,
                proposal_sigma_xy=params.proposal_sigma_xy,
                proposal_sigma_theta=params.proposal_sigma_theta,
                proposal_n_samples=params.proposal_n_samples,
            )

            # Measure step duration
            step_duration = time.time() - step_start_time
            
            # Extract evaluation info from the RBPF instance
            info = rbpf.get_step_info()
            step_idx_logged = info.get("step")
            est_pose = info.get("weighted_mean_pose")
            best_particle_pose = info.get("best_particle_pose")
            neff = info.get("neff")
            scan_match_failed = info.get("scan_match_failed_any")
            scan_match_fallback_failed = info.get("scan_match_fallback_failed_any")
            particle_weight_min = info.get("particle_weight_min")
            particle_weight_max = info.get("particle_weight_max")
            particle_weight_mean = info.get("particle_weight_mean")
            proposal_metrics = info.get("proposal_metrics")
            measurement_model_counters_fallback = info.get("measurement_model_counters_fallback")

            # Evaluate the current step and store results
            step_result = self.evaluator.evaluate_step(
                step_idx=step_idx_logged if step_idx_logged is not None else step_idx,
                t=step.t,
                true_pose=step.true_pose,
                raw_odom_pose=raw_odom_pose,
                est_pose=est_pose,
                best_particle_pose=best_particle_pose,
                scan_match_failed=scan_match_failed,
                scan_match_fallback_failed=scan_match_fallback_failed,
                neff=neff,
                particle_weight_min=particle_weight_min,
                particle_weight_max=particle_weight_max,
                particle_weight_mean=particle_weight_mean,
                step_duration=step_duration,
                proposal_metrics=proposal_metrics,
                measurement_model_counters_fallback=measurement_model_counters_fallback,

            )

            run_result.step_results.append(step_result)

        # Store the final highest-weighted particle map and its metadata.
        run_result.best_part_map = info.get("best_particle_map", None)
        run_result.best_part_map_meta = info.get("best_particle_map_meta", None)

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
        print("Scan matcher timing summary (mean per run):")
        print(
            f"  scan matching (total): "
            f"{_to_ms(run_result.summary.get('mean_time_duration_scan_matching'))} ms"
        )
        print(
            f"  prediction: "
            f"{_to_ms(run_result.summary.get('mean_time_duration_prediction'))} ms"
        )
        print(
            f"  map extraction: "
            f"{_to_ms(run_result.summary.get('mean_time_duration_map_extraction'))} ms"
        )
        print(
            f"  correct pose: "
            f"{_to_ms(run_result.summary.get('mean_time_duration_correct_pose'))} ms"
        )

        print("ICP timing summary (mean per run):")
        print(
            f"  init icp transform: "
            f"{_to_ms(run_result.summary.get('mean_t_init_icp_trans'))} ms"
        )
        print(
            f"  init and train nn tree normals: "
            f"{_to_ms(run_result.summary.get('mean_t_init_and_train_nn_tree_normals'))} ms"
        )
        print(
            f"  downsampling pointcloud: "
            f"{_to_ms(run_result.summary.get('mean_t_downsampling_pointcloud'))} ms"
        )
        print(
            f"  compute normal: "
            f"{_to_ms(run_result.summary.get('mean_t_compute_normal'))} ms"
        )
        
        print(
            f"  find nn outlier rejection: "
            f"{_to_ms(run_result.summary.get('mean_t_find_nn_outlier_rejec'))} ms"
        )

        print(
            f"  outlier rejection: "
            f"{_to_ms(run_result.summary.get('mean_t_outlier_rejection'))} ms"
        )
        
        print(
            f"  prepare system: "
            f"{_to_ms(run_result.summary.get('mean_t_prepare_system'))} ms"
        )
        print(
            f"  solve least squares: "
            f"{_to_ms(run_result.summary.get('mean_t_solve_least_squares'))} ms"
        )
        print(
            f"  transf update and results: "
            f"{_to_ms(run_result.summary.get('mean_t_transf_update_and_results'))} ms"
        )
        print(
            f"  find trans total: "
            f"{_to_ms(run_result.summary.get('mean_t_find_trans'))} ms"
        )

        print(
            f"  update pose: "
            f"{_to_ms(run_result.summary.get('mean_time_duration_update_pose'))} ms"
        )

        print("Proposal timing summary (mean per run):")
        print(
            f"  sample poses: "
            f"{_to_ms(run_result.summary.get('mean_t_sample_poses'))} ms"
        )
        print(
            f"  predict poses: "
            f"{_to_ms(run_result.summary.get('mean_t_pred_poses'))} ms"
        )
        print(
            f"  motion model: "
            f"{_to_ms(run_result.summary.get('mean_t_motion_model'))} ms"
        )
        print(
            f"  measurement model: "
            f"{_to_ms(run_result.summary.get('mean_t_meas_model'))} ms"
        )
        print(
            f"  compute proposal params: "
            f"{_to_ms(run_result.summary.get('mean_t_compute_prop_params'))} ms"
        )
        print(
            f"  sample from proposal: "
            f"{_to_ms(run_result.summary.get('mean_t_sample_from_prop'))} ms"
        )
        

        return run_result
    

    # def run_rbpf_parallel(self, playback_data: PlaybackData, params: ExperimentParams) -> RunResult:
    #     """
    #     Executes one full RBPF run over all playback steps and returns evaluated results. This version uses
    #     the RBPF parallel step method with a multiprocessing pool for parallel particle updates. 
    #     """
    #     # Create rbpf instance for the current parameter set
    #     rbpf = self.factory.create(
    #         scan_match_fac=ScanMatchFactory(),
    #         particle_params=params.particle_params,
    #         occ_param=params.occupancy_params,
    #         sens_params=params.sensor_params,
    #         map_param=params.map_param,
    #         icp_params=params.icp_params,
    #         robot_params=params.robot_params,
    #         scan_matcher_params=params.scan_matcher_params,
    #         motion_model_params=params.motion_model_params,
    #         measurement_model_params=params.measurement_model_params,
    #         neff_threshold=params.neff_threshold,
    #     )

    #     steps = playback_data.step_data_list
    #     run_result = RunResult(params=params)

    #     raw_odom_estimator = RawOdomEstimator(
    #         motion_model=rbpf.motion_model,
    #         start_pose=params.particle_params.start_pose,
    #     )

    #     # Ensure valid scan downsampling values for filter/proposal and map update.
    #     every_nth_filter = max(1, int(params.every_nth_scan_filter))
    #     every_nth_map = max(1, int(params.every_nth_scan_map))

    #     # Init multi processing pool
    #     with ParticleProcessPool(
    #         n_workers=4,            
    #     ) as pool:
        
    #         for step_idx, step in enumerate(steps):
    #             step_start_time = time.time()
    #             step_duration = None

    #             raw_odom_pose = raw_odom_estimator.predict_pose(
    #                 dl=step.dl,
    #                 dr=step.dr,
    #             )

    #             # Subsample and clean measurements
    #             measurements_proposal = (
    #                 step.scan[::every_nth_filter] if every_nth_filter > 1 else step.scan
    #             )
                
    #             measurements_proposal = [
    #                 (r, b) for r, b in measurements_proposal if np.isfinite(r) and not np.isnan(r)
    #             ]

    #             measurements_map = step.scan[::every_nth_map] if every_nth_map > 1 else step.scan

    #             if np.isnan(measurements_proposal).any():
    #                 print("\nPlayback runner: measurement model contains nan value after subsampling scans")

    #             # Use inf vals for map update, too -> clear free space faster
    #             # measurements_map = [
    #             #     (r, b) for r, b in measurements_map if np.isfinite(r) and not np.isnan(r)
    #             # ]                

    #             # Run rbpf filter step
    #             rbpf.step_parallel(
    #                 particle_process_pool=pool,
    #                 odom=(step.dl, step.dr),
    #                 measurements_proposal=measurements_proposal,
    #                 measurements_map_update=measurements_map,
    #                 proposal_sigma_xy=params.proposal_sigma_xy,
    #                 proposal_sigma_theta=params.proposal_sigma_theta,
    #                 proposal_n_samples=params.proposal_n_samples,
    #             )

    #             # Measure step duration
    #             step_duration = time.time() - step_start_time
                
    #             # Extract evaluation info from the RBPF instance
    #             info = rbpf.get_step_info()
    #             step_idx_logged = info.get("step")
    #             est_pose = info.get("weighted_mean_pose")
    #             best_particle_pose = info.get("best_particle_pose")
    #             neff = info.get("neff")
    #             scan_match_failed = info.get("scan_match_failed_any")
    #             scan_match_fallback_failed = info.get("scan_match_fallback_failed_any")
    #             particle_weight_min = info.get("particle_weight_min")
    #             particle_weight_max = info.get("particle_weight_max")
    #             particle_weight_mean = info.get("particle_weight_mean")
    #             proposal_metrics = info.get("proposal_metrics")
    #             measurement_model_counters_fallback = info.get("measurement_model_counters_fallback")

    #             # Evaluate the current step and store results
    #             step_result = self.evaluator.evaluate_step(
    #                 step_idx=step_idx_logged if step_idx_logged is not None else step_idx,
    #                 t=step.t,
    #                 true_pose=step.true_pose,
    #                 raw_odom_pose=raw_odom_pose,
    #                 est_pose=est_pose,
    #                 best_particle_pose=best_particle_pose,
    #                 scan_match_failed=scan_match_failed,
    #                 scan_match_fallback_failed=scan_match_fallback_failed,
    #                 neff=neff,
    #                 particle_weight_min=particle_weight_min,
    #                 particle_weight_max=particle_weight_max,
    #                 particle_weight_mean=particle_weight_mean,
    #                 step_duration=step_duration,
    #                 proposal_metrics=proposal_metrics,
    #                 measurement_model_counters_fallback=measurement_model_counters_fallback,

    #             )

    #             run_result.step_results.append(step_result)

    #     # Store the final highest-weighted particle map and its metadata.
    #     run_result.best_part_map = info.get("best_particle_map", None)
    #     run_result.best_part_map_meta = info.get("best_particle_map_meta", None)

    #     # Summarize the run results and store in the run result object
    #     run_result.summary = self.evaluator.summarize_run(
    #         step_results=run_result.step_results,
    #         params=params,
    #     )
    #     run_result.summary.update(self._aggregate_icp_counters(rbpf))
    #     timing_summary = rbpf.timing_summary()
    #     run_result.summary.update(timing_summary)

    #     def _to_ms(value):
    #         return value * 1000.0 if value is not None else None

    #     print("RBPF timing summary (mean per run):")
    #     print(f"  update_particles: {_to_ms(timing_summary.get('mean_timing_update_particles_s'))} ms")
    #     print(f"  normalize+neff: {_to_ms(timing_summary.get('mean_timing_normalize_neff_s'))} ms")
    #     print(f"  metrics: {_to_ms(timing_summary.get('mean_timing_metrics_s'))} ms")
    #     print(f"  resampling (when triggered): {_to_ms(timing_summary.get('mean_timing_resampling_s'))} ms")
    #     print("  update_particle internals:")
    #     print(
    #         f"    scan_match.update_pose: {_to_ms(timing_summary.get('mean_timing_scan_match_update_pose_s'))} ms "
    #         f"(count={timing_summary.get('timing_scan_match_update_pose_count')})"
    #     )
    #     print(
    #         f"    proposal.estimate_proposal: {_to_ms(timing_summary.get('mean_timing_proposal_estimation_s'))} ms "
    #         f"(count={timing_summary.get('timing_proposal_estimation_count')})"
    #     )
    #     print(
    #         f"    scan_match fallback block: {_to_ms(timing_summary.get('mean_timing_scan_match_fallback_s'))} ms "
    #         f"(count={timing_summary.get('timing_scan_match_fallback_count')})"
    #     )
    #     print(
    #         f"    map_extension_if_necessary loop: {_to_ms(timing_summary.get('mean_timing_map_extension_s'))} ms "
    #         f"(count={timing_summary.get('timing_map_extension_count')})"
    #     )
    #     print(
    #         f"    ogm.update_map: {_to_ms(timing_summary.get('mean_timing_map_update_s'))} ms "
    #         f"(count={timing_summary.get('timing_map_update_count')})"
    #     )
    #     print("Scan matcher timing summary (mean per run):")
    #     print(
    #         f"  scan matching (total): "
    #         f"{_to_ms(run_result.summary.get('mean_time_duration_scan_matching'))} ms"
    #     )
    #     print(
    #         f"  prediction: "
    #         f"{_to_ms(run_result.summary.get('mean_time_duration_prediction'))} ms"
    #     )
    #     print(
    #         f"  map extraction: "
    #         f"{_to_ms(run_result.summary.get('mean_time_duration_map_extraction'))} ms"
    #     )
    #     print(
    #         f"  correct pose: "
    #         f"{_to_ms(run_result.summary.get('mean_time_duration_correct_pose'))} ms"
    #     )
    #     print("ICP timing summary (mean per run):")
    #     print(
    #         f"  downsampling pointcloud: "
    #         f"{_to_ms(run_result.summary.get('mean_t_downsampling_pointcloud'))} ms"
    #     )
    #     print(
    #         f"  compute normal: "
    #         f"{_to_ms(run_result.summary.get('mean_t_compute_normal'))} ms"
    #     )
    #     print(
    #         f"  outlier rejection: "
    #         f"{_to_ms(run_result.summary.get('mean_t_outlier_rejection'))} ms"
    #     )
    #     print(
    #         f"  prepare system: "
    #         f"{_to_ms(run_result.summary.get('mean_t_prepare_system'))} ms"
    #     )
    #     print(
    #         f"  solve least squares: "
    #         f"{_to_ms(run_result.summary.get('mean_t_solve_least_squares'))} ms"
    #     )

    #     return run_result
    

    
    # def run_without_proposal_pose(self, playback_data: PlaybackData, params: ExperimentParams) -> RunResult:
    #     """
    #     Scan Matching only variant of the rbpf runner. Created for training and evaluating the scan matcher.
    #     """
    #     # Create rbpf instance for the current parameter set
    #     rbpf = self.factory.create(
    #         scan_match_fac=ScanMatchFactory(),
    #         particle_params=params.particle_params,
    #         occ_param=params.occupancy_params,
    #         sens_params=params.sensor_params,
    #         map_param=params.map_param,
    #         icp_params=params.icp_params,
    #         robot_params=params.robot_params,
    #         scan_matcher_params=params.scan_matcher_params,
    #         motion_model_params=params.motion_model_params,
    #         measurement_model_params=params.measurement_model_params,
    #         neff_threshold=params.neff_threshold,
    #     )

    #     steps = playback_data.step_data_list
    #     run_result = RunResult(params=params)

    #     raw_odom_estimator = RawOdomEstimator(
    #         motion_model=rbpf.motion_model,
    #         start_pose=params.particle_params.start_pose,
    #     )

    #     # Ensure valid scan downsampling values for filter/proposal and map update.
    #     every_nth_filter = max(1, int(params.every_nth_scan_filter))
    #     every_nth_map = max(1, int(params.every_nth_scan_map))
    #     print(
    #         f"Running RBPF with params: {params.tag} "
    #         f"(every_nth_scan_filter={every_nth_filter}, every_nth_scan_map={every_nth_map})"
    #     )

    #     for step_idx, step in enumerate(steps):
    #         step_start_time = time.time()
    #         step_duration = None

    #         raw_odom_pose = raw_odom_estimator.predict_pose(
    #             dl=step.dl,
    #             dr=step.dr,
    #         )

    #         # Subsample and clean measurements
    #         measurements_proposal = (
    #             step.scan[::every_nth_filter] if every_nth_filter > 1 else step.scan
    #         )
    #         measurements_map = step.scan[::every_nth_map] if every_nth_map > 1 else step.scan

    #         measurements_proposal = [
    #             (r, b) for r, b in measurements_proposal if np.isfinite(r) and not np.isnan(r)
    #         ]

    #         # Use inf vals for map update, too -> clear free space faster
    #         # measurements_map = [
    #         #     (r, b) for r, b in measurements_map if np.isfinite(r) and not np.isnan(r)
    #         # ]

    #         # Run rbpf filter step
    #         rbpf.step_rbpf_without_proposal_pose(
    #             odom=(step.dl, step.dr),
    #             measurements_proposal=measurements_proposal,
    #             measurements_map_update=measurements_map,
    #             proposal_sigma_xy=params.proposal_sigma_xy,
    #             proposal_sigma_theta=params.proposal_sigma_theta,
    #             proposal_n_samples=params.proposal_n_samples,
    #             meas_kernel_size=params.meas_kernel_size,
    #             gaussian_sigma=params.gaussian_sigma,
    #             proposal_alpha=params.proposal_alpha,
    #             proposal_beta=params.proposal_beta,
    #         )

    #         # Measure step duration
    #         step_duration = time.time() - step_start_time
            
    #         # Extract evaluation info from the RBPF instance
    #         info = rbpf.get_step_info()
    #         step_idx_logged = info.get("step")
    #         est_pose = info.get("weighted_mean_pose")
    #         best_particle_pose = info.get("best_particle_pose")
    #         neff = info.get("neff")
    #         scan_match_failed = info.get("scan_match_failed_any")
    #         scan_match_fallback_failed = info.get("scan_match_fallback_failed_any")
    #         particle_weight_min = info.get("particle_weight_min")
    #         particle_weight_max = info.get("particle_weight_max")
    #         particle_weight_mean = info.get("particle_weight_mean")
    #         proposal_metrics = info.get("proposal_metrics")

    #         # Evaluate the current step and store results
    #         step_result = self.evaluator.evaluate_step(
    #             step_idx=step_idx_logged if step_idx_logged is not None else step_idx,
    #             t=step.t,
    #             true_pose=step.true_pose,
    #             raw_odom_pose=raw_odom_pose,
    #             est_pose=est_pose,
    #             best_particle_pose=best_particle_pose,
    #             scan_match_failed=scan_match_failed,
    #             scan_match_fallback_failed=scan_match_fallback_failed,
    #             neff=neff,
    #             particle_weight_min=particle_weight_min,
    #             particle_weight_max=particle_weight_max,
    #             particle_weight_mean=particle_weight_mean,
    #             step_duration=step_duration,
    #             proposal_metrics=proposal_metrics,
    #         )

    #         run_result.step_results.append(step_result)

    #     # Summarize the run results and store in the run result object
    #     run_result.summary = self.evaluator.summarize_run(
    #         step_results=run_result.step_results,
    #         params=params,
    #     )
    #     run_result.summary.update(self._aggregate_icp_counters(rbpf))
    #     timing_summary = rbpf.timing_summary()
    #     run_result.summary.update(timing_summary)

    #     def _to_ms(value):
    #         return value * 1000.0 if value is not None else None

    #     print("RBPF timing summary (mean per run):")
    #     print(f"  update_particles: {_to_ms(timing_summary.get('mean_timing_update_particles_s'))} ms")
    #     print(f"  normalize+neff: {_to_ms(timing_summary.get('mean_timing_normalize_neff_s'))} ms")
    #     print(f"  metrics: {_to_ms(timing_summary.get('mean_timing_metrics_s'))} ms")
    #     print(f"  resampling (when triggered): {_to_ms(timing_summary.get('mean_timing_resampling_s'))} ms")
    #     print("  update_particle internals:")
    #     print(
    #         f"    scan_match.update_pose: {_to_ms(timing_summary.get('mean_timing_scan_match_update_pose_s'))} ms "
    #         f"(count={timing_summary.get('timing_scan_match_update_pose_count')})"
    #     )
    #     print(
    #         f"    proposal.estimate_proposal: {_to_ms(timing_summary.get('mean_timing_proposal_estimation_s'))} ms "
    #         f"(count={timing_summary.get('timing_proposal_estimation_count')})"
    #     )
    #     print(
    #         f"    scan_match fallback block: {_to_ms(timing_summary.get('mean_timing_scan_match_fallback_s'))} ms "
    #         f"(count={timing_summary.get('timing_scan_match_fallback_count')})"
    #     )
    #     print(
    #         f"    map_extension_if_necessary loop: {_to_ms(timing_summary.get('mean_timing_map_extension_s'))} ms "
    #         f"(count={timing_summary.get('timing_map_extension_count')})"
    #     )
    #     print(
    #         f"    ogm.update_map: {_to_ms(timing_summary.get('mean_timing_map_update_s'))} ms "
    #         f"(count={timing_summary.get('timing_map_update_count')})"
    #     )
    #     print("Scan matcher timing summary (mean per run):")
    #     print(
    #         f"  scan matching (total): "
    #         f"{_to_ms(run_result.summary.get('mean_time_duration_scan_matching'))} ms"
    #     )
    #     print(
    #         f"  prediction: "
    #         f"{_to_ms(run_result.summary.get('mean_time_duration_prediction'))} ms"
    #     )
    #     print(
    #         f"  map extraction: "
    #         f"{_to_ms(run_result.summary.get('mean_time_duration_map_extraction'))} ms"
    #     )
    #     print(
    #         f"  correct pose: "
    #         f"{_to_ms(run_result.summary.get('mean_time_duration_correct_pose'))} ms"
    #     )

    #     print("ICP timing summary (mean per run):")
    #     print(
    #         f"  downsampling pointcloud: "
    #         f"{_to_ms(run_result.summary.get('mean_t_downsampling_pointcloud'))} ms"
    #     )
    #     print(
    #         f"  compute normal: "
    #         f"{_to_ms(run_result.summary.get('mean_t_compute_normal'))} ms"
    #     )
    #     print(
    #         f"  outlier rejection: "
    #         f"{_to_ms(run_result.summary.get('mean_t_outlier_rejection'))} ms"
    #     )
    #     print(
    #         f"  prepare system: "
    #         f"{_to_ms(run_result.summary.get('mean_t_prepare_system'))} ms"
    #     )
    #     print(
    #         f"  solve least squares: "
    #         f"{_to_ms(run_result.summary.get('mean_t_solve_least_squares'))} ms"
    #     )

    #     return run_result
