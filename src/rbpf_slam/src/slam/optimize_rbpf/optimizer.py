from dataclasses import dataclass
from typing import Iterable, List, Optional
import time

import json
import hashlib

import numpy as np

from tqdm import tqdm

from .playback_runner import PlaybackRunner
from .scorer import RunScorer
from .playback_defs import ExperimentParams, PlaybackData, StepData
from ..infrastructure.playback_converter import PlaybackConverter


@dataclass
class RankedRun:
    params: ExperimentParams
    summary: dict
    score: float
    step_results: list
    seed: Optional[int]
    dataset_id: Optional[int] = None        # Optional field to identify the dataset used for this run
    map_name: Optional[str] = None          # Optional field to identify the map used for this run (name of the map for example)
    parameter_tag: Optional[str] = None      # Optional field to add the unique parameter combination (ID)
    parameter_hash: Optional[str] = None     # Optional field to add the unique parameter combination hash (ID)



class RBPFOptimizer:
    """
    Optimizes RBPF parameters by replaying one dataset for each parameter set.
    """

    def __init__(self, runner: PlaybackRunner, scorer: RunScorer):
        self.runner = runner
        self.scorer = scorer


    @staticmethod
    def generate_params_for_hash(exp_params: ExperimentParams) -> dict:
        """
        Build a fixed-name, 1:1 parameter mapping used as stable hash input.
        """
        parameter_for_hash = {
            # OccupancyParams
            "occupancy_min_distance_to_border": exp_params.occupancy_params.min_distance_to_border,
            "occupancy_prior_probability": exp_params.occupancy_params.prior_probability,
            "occupancy_increasing_probability": exp_params.occupancy_params.increasing_probability,
            "occupancy_decreasing_probability": exp_params.occupancy_params.decreasing_probability,
            "occupancy_min_log_odds": exp_params.occupancy_params.min_log_odds,
            "occupancy_max_log_odds": exp_params.occupancy_params.max_log_odds,

            # SensorParams
            "sensor_min_sensor_range": exp_params.sensor_params.min_sensor_range,
            "sensor_max_sensor_range": exp_params.sensor_params.max_sensor_range,

            # MapParameter
            "map_map_width": exp_params.map_param.map_width,
            "map_map_height": exp_params.map_param.map_height,
            "map_grid_resolution_m": exp_params.map_param.grid_resolution_m,

            # ICPParams
            "icp_max_n_points": exp_params.icp_params.max_n_points,
            "icp_max_correspondence_distance": exp_params.icp_params.max_correspondence_distance,
            "icp_neighbors_pca": exp_params.icp_params.neighbors_pca,
            "icp_max_iterations": exp_params.icp_params.max_iterations,
            "icp_epsilon_rel": exp_params.icp_params.epsilon_rel,
            "icp_no_improvement_limit": exp_params.icp_params.no_improvement_limit,
            "icp_min_error": exp_params.icp_params.min_error,
            "icp_min_dtrans": exp_params.icp_params.min_dtrans,
            "icp_min_drot": exp_params.icp_params.min_drot,
            "icp_min_points": exp_params.icp_params.min_points,
            "icp_min_corresp": exp_params.icp_params.min_corresp,
            "icp_min_hessian_rank": exp_params.icp_params.min_hessian_rank,
            "icp_max_hessian_condition": exp_params.icp_params.max_hessian_condition,
            "icp_max_translation_jump": exp_params.icp_params.max_translation_jump,
            "icp_max_rotation_jump": exp_params.icp_params.max_rotation_jump,
            "icp_max_acceptable_mean_error": exp_params.icp_params.max_acceptable_mean_error,

            # RobotParams
            "robot_wheel_separation": exp_params.robot_params.wheel_separation,

            # ScanMatcherParams
            "scan_matcher_occ_thres": exp_params.scan_matcher_params.occ_thres,
            "scan_matcher_delta_r": exp_params.scan_matcher_params.delta_r,
            "scan_matcher_surface_radius_m": exp_params.scan_matcher_params.surface_radius_m,
            "scan_matcher_min_free_ratio": exp_params.scan_matcher_params.min_free_ratio,

            # ParticleParams
            "particle_start_pose": exp_params.particle_params.start_pose,
            "particle_n_particles": exp_params.particle_params.n_particles,

            # MotionModelParams
            "motion_sigma_x": exp_params.motion_model_params.sigma_x,
            "motion_sigma_y": exp_params.motion_model_params.sigma_y,
            "motion_sigma_theta": exp_params.motion_model_params.sigma_theta,
            "motion_wheel_separation": exp_params.motion_model_params.wheel_separation,
            "motion_ctrl_motion_fac": exp_params.motion_model_params.ctrl_motion_fac,
            "motion_ctrl_turn_fac": exp_params.motion_model_params.ctrl_turn_fac,

            # MeasurementModelParams
            "measurement_sigma_measurement": exp_params.measurement_model_params.sigma_measurement,

            # ExperimentParams
            "every_nth_scan_filter": exp_params.every_nth_scan_filter,
            "every_nth_scan_map": exp_params.every_nth_scan_map,
            "neff_threshold": exp_params.neff_threshold,
            "proposal_sigma_xy": exp_params.proposal_sigma_xy,
            "proposal_sigma_theta": exp_params.proposal_sigma_theta,
            "proposal_n_samples": exp_params.proposal_n_samples,
            "meas_kernel_size": exp_params.meas_kernel_size,
            "gaussian_sigma": exp_params.gaussian_sigma,
            "proposal_alpha": exp_params.proposal_alpha,
            "proposal_beta": exp_params.proposal_beta,
            "measurement_noise_stddev": exp_params.measurement_noise_stddev,
            "used_meas_model": exp_params.used_meas_model,
        }

        return parameter_for_hash


    @staticmethod
    def _apply_measurement_noise_per_seed(
        playback_data: PlaybackData,
        measurement_stddev: Optional[float],
        measurement_noise_seed: Optional[int],
        min_range: float,
        max_range: float,
    ) -> PlaybackData:
        """
        Recreate playback scans with deterministic noise for the given measurement seed.
        The source playback_data must contain clean (non-noised) scans.
        """
        if measurement_stddev is None:
            return playback_data

        # Keep measurement-noise RNG isolated from global numpy seeding.
        rng = np.random.default_rng(measurement_noise_seed)
        noisy_steps: List[StepData] = []

        for step in playback_data.step_data_list:
            ranges = [r for r, _ in step.scan]
            bearings = [b for _, b in step.scan]

            noisy_ranges = PlaybackConverter.add_measurement_noise(
                ranges=ranges,
                stddev=measurement_stddev,
                min_range=min_range,
                max_range=max_range,
                rng=rng,
            )

            noisy_scan = [(float(r), float(b)) for r, b in zip(noisy_ranges, bearings)]
            noisy_steps.append(
                StepData(
                    t=step.t,
                    dl=step.dl,
                    dr=step.dr,
                    scan=noisy_scan,
                    true_pose=step.true_pose,
                )
            )

        return PlaybackData(step_data_list=noisy_steps)


    def optimize(
        self,
        playback_data: PlaybackData,
        param_grid: Iterable[ExperimentParams],
        seeds: Iterable[int] = None,
        dataset_id: Optional [int] = None,
        map_name: Optional[str] = None,
        use_seed_list_for_measurement_noise: bool = True,
    ) -> List[RankedRun]:
        """
        Runs the RBPF once per parameter set and ranks all runs by score (lower is better).
        """
        # Convert params and seed to lists
        params_list = list(param_grid)
        seed_list = [int(s) for s in seeds] if seeds is not None else [None]
        total_runs = len(params_list)

        if not seed_list:
            seed_list = [None]

        if total_runs == 0:
            print("No parameter combinations provided. Nothing to optimize.")
            return []

        print(f"Starting RBPF optimization with {total_runs * len(seed_list)} run(s)...")
        ranked_runs: List[RankedRun] = []

        # Measure starting time
        start_time = time.time()

        # Run filter over all params by seeds
        for params in tqdm(params_list, total=total_runs, desc="RBPF optimization", unit="run"):
            # Compute hash id that represents current parameter set
            parameter_for_hash = self.generate_params_for_hash(params)
            param_json = json.dumps(parameter_for_hash, sort_keys=True)
            param_hash = hashlib.sha256(param_json.encode()).hexdigest()[:12]  

            for run_seed in seed_list:
                if run_seed is not None:
                    np.random.seed(run_seed)

                # Decide whether to use the run seed for measurement noise or not
                if use_seed_list_for_measurement_noise:
                    measurement_noise_seed = run_seed
                else:
                    measurement_noise_seed = None

                # Apply measurement noise
                run_playback_data = self._apply_measurement_noise_per_seed(
                    playback_data=playback_data,
                    measurement_stddev=params.measurement_noise_stddev,
                    measurement_noise_seed=measurement_noise_seed,
                    min_range=params.sensor_params.min_sensor_range,
                    max_range=params.sensor_params.max_sensor_range,
                )

                # Run the rbpf filter on one parameter set and compute the rating score
                run_result = self.runner.run(run_playback_data, params)
                score = self.scorer.score(run_result.summary)

                # Store run results
                ranked_runs.append(
                    RankedRun(
                        params=params,
                        summary=run_result.summary,
                        score=score,
                        step_results=run_result.step_results,
                        seed=run_seed,
                        dataset_id=dataset_id,
                        map_name=map_name,
                        parameter_tag=params.tag,
                        parameter_hash=param_hash,
                    )
                )

        # Measure ending time and print info
        optm_duration_s = time.time() - start_time
        n_runs = total_runs * len(seed_list)
        print(f"Finished RBPF optimization: {n_runs}/{n_runs} runs in {optm_duration_s:.2f}s")
        
        # Sort runs by score (ascending order)
        # ranked_runs.sort(key=lambda x: x.score)
        
        return ranked_runs
    

    def optimize_without_proposal_pose(
        self,
        playback_data: PlaybackData,
        param_grid: Iterable[ExperimentParams],
        seeds: Optional[Iterable[int]] = None,
        dataset_id: Optional [int] = None,
        map_name: Optional[str] = None,
        use_seed_list_for_measurement_noise: bool = True,
    ) -> List[RankedRun]:
        """
        Runs the RBPF once per parameter set and ranks all runs by score (lower is better).
        """
        params_list = list(param_grid)
        seed_list = [int(s) for s in seeds] if seeds is not None else [None]
        total_runs = len(params_list)

        if not seed_list:
            seed_list = [None]

        if total_runs == 0:
            print("No parameter combinations provided. Nothing to optimize.")
            return []

        print(f"Starting RBPF optimization with {total_runs * len(seed_list)} runs...")
        ranked_runs: List[RankedRun] = []

        start_time = time.time()

        for params in tqdm(params_list, total=total_runs, desc="RBPF optimization", unit="run"): 
            # Compute hash id that represents current parameter set
            parameter_for_hash = self.generate_params_for_hash(params)
            param_json = json.dumps(parameter_for_hash, sort_keys=True)
            param_hash = hashlib.sha256(param_json.encode()).hexdigest()[:12]

            for run_seed in seed_list:
                if run_seed is not None:
                    np.random.seed(run_seed)

                # Decide whether to use the run seed for measurement noise or not
                if use_seed_list_for_measurement_noise:
                    measurement_noise_seed = run_seed
                else:
                    measurement_noise_seed = None

                run_playback_data = self._apply_measurement_noise_per_seed(
                    playback_data=playback_data,
                    measurement_stddev=params.measurement_noise_stddev,
                    measurement_noise_seed=measurement_noise_seed,
                    min_range=params.sensor_params.min_sensor_range,
                    max_range=params.sensor_params.max_sensor_range,
                )

                run_result = self.runner.run_without_proposal_pose(run_playback_data, params)
                score = self.scorer.score(run_result.summary)

                ranked_runs.append(
                    RankedRun(
                        params=params,
                        summary=run_result.summary,
                        score=score,
                        step_results=run_result.step_results,
                        seed=run_seed,
                        dataset_id=dataset_id,
                        map_name=map_name,
                        parameter_tag=params.tag,
                        parameter_hash=param_hash,
                    )
                )

        # Measure ending time and print info
        optm_duration_s = time.time() - start_time
        n_runs = total_runs * len(seed_list)
        print(f"Finished RBPF optimization: {n_runs}/{n_runs} runs in {optm_duration_s:.2f}s")

        # Sort runs by score (ascending order)
        # ranked_runs.sort(key=lambda x: x.score)
        return ranked_runs


class ScanMatcherOptimizer(RBPFOptimizer):
    """
    Backward-compatible alias used by existing imports in the RBPF tuning script.
    """
    pass