import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import time

import json
import hashlib

import numpy as np

from tqdm import tqdm

from .playback_runner import PlaybackRunner
from .scorer import RunScorer
from .playback_defs import ExperimentParams, PlaybackData, StepData
from .evaluator import RBPFEvaluator
from ..infrastructure.playback_converter import PlaybackConverter
from ..infrastructure.map_data_handler import MapDataHandler


from ..rbpf.rbpf import (
    RBPFFactory
)

# Declare glöobal var to store playabck data for each worker process.
_WORKER_PLAYBACK_DATA = None
_WORKER_RUNNER = None
_WORKER_SCORER = None
_WORKER_RUN_STORAGE_DIR = None
_WORKER_STORE_MAP_DATA = False


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




def _init_rbpf_worker(
    playback_data: PlaybackData,
    run_storage_dir: Optional[str] = None,
    store_map_data: bool = False,
) -> None:
    """
    Store playback data once per worker process.

    This avoids sending the full playback data again for every single parameter job.
    """
    global _WORKER_PLAYBACK_DATA
    global _WORKER_RUNNER
    global _WORKER_SCORER
    global _WORKER_RUN_STORAGE_DIR
    global _WORKER_STORE_MAP_DATA

    # Define playback storage for worker
    _WORKER_PLAYBACK_DATA = playback_data
    
    # Define runner and scorer
    _WORKER_RUNNER = PlaybackRunner(
        factory=RBPFFactory(),
        evaluator=RBPFEvaluator(),
    )

    _WORKER_SCORER = RunScorer()
    _WORKER_RUN_STORAGE_DIR = run_storage_dir
    _WORKER_STORE_MAP_DATA = bool(store_map_data)

    print(f"Worker PID: {os.getpid()}", flush=True)


def _run_rbpf_job(job: dict) -> RankedRun:
    """
    Run one independent rbpf optimization job.

    One job = one parameter set + one seed + one dataset.
    Each worker creates its own runner/scorer/RBPF/scan matcher.
    """
    # Reference the global playback data storage
    global _WORKER_PLAYBACK_DATA
    global _WORKER_RUN_STORAGE_DIR
    global _WORKER_STORE_MAP_DATA

    # Check if worker vars have been initialized
    if _WORKER_PLAYBACK_DATA is None:
        raise RuntimeError("Worker playback data has not been initialized.")
    
    if _WORKER_RUNNER is None:
        raise RuntimeError("Worker runner has not been initialized.")

    if _WORKER_SCORER is None:
        raise RuntimeError("Worker scorer has not been initialized.")

    if _WORKER_STORE_MAP_DATA and not _WORKER_RUN_STORAGE_DIR:
        raise RuntimeError(
            "STORE_MAP_DATA is enabled, but worker run storage directory is not configured."
        )

    # Extract data to execute the job
    params: ExperimentParams = job["params"]
    run_seed: Optional[int] = job["seed"]
    dataset_id: Optional[str] = job["dataset_id"]
    map_name: Optional[str] = job["map_name"]
    use_seed_list_for_measurement_noise: bool = job["use_seed_list_for_measurement_noise"]
    keep_step_results: bool = job["keep_step_results"]

    # Define seed
    if run_seed is not None:
        np.random.seed(run_seed)

    measurement_noise_seed = (
        run_seed if use_seed_list_for_measurement_noise else None
    )

    run_playback_data = RBPFOptimizer._apply_measurement_noise_per_seed(
        playback_data=_WORKER_PLAYBACK_DATA,
        measurement_stddev=params.measurement_noise_stddev,
        measurement_noise_seed=measurement_noise_seed,
        min_range=params.sensor_params.min_sensor_range,
        max_range=params.sensor_params.max_sensor_range,
    )

    # Generate parameter hash to identify the parameter set used for the job
    parameter_for_hash = RBPFOptimizer.generate_params_for_hash(params)
    param_json = json.dumps(parameter_for_hash, sort_keys=True)
    param_hash = hashlib.sha256(param_json.encode()).hexdigest()[:12]

    # Run the runner and scorer for the job
    run_result = _WORKER_RUNNER.run(run_playback_data, params)
    score = _WORKER_SCORER.score(run_result.summary)

    # Store the final highest-weighted particle map for this run.
    if _WORKER_STORE_MAP_DATA:
        seed_part = str(run_seed) if run_seed is not None else "none"
        ds_id_part = str(dataset_id) if dataset_id is not None else "unknown_dataset"
        map_part = str(map_name) if map_name is not None else "unknown_map"
        run_dir = os.path.join(
            _WORKER_RUN_STORAGE_DIR,
            map_part + "_" + ds_id_part + "_" + str(param_hash) + "_" + seed_part,
        )

        best_p_map = run_result.best_part_map
        best_p_map_meta = run_result.best_part_map_meta

        if best_p_map is not None and best_p_map_meta is not None:
            MapDataHandler.save(
                output_dir=run_dir,
                log_odds_map=best_p_map,
                resolution=best_p_map_meta.get("grid_resolution_m"),
                shift_x=best_p_map_meta.get("shift_x"),
                shift_y=best_p_map_meta.get("shift_y"),
                occupied_threshold=params.measurement_model_params.occ_thresh,
                free_threshold=params.measurement_model_params.free_thresh,
                min_log_odds=params.occupancy_params.min_log_odds,
                max_log_odds=params.occupancy_params.max_log_odds,
            )

    return RankedRun(
        params=params,
        summary=run_result.summary,
        score=score, 
        step_results=run_result.step_results if keep_step_results else [],
        seed=run_seed,
        dataset_id=dataset_id,
        map_name=map_name,
        parameter_tag=params.tag,
        parameter_hash=param_hash,
    )


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
            "particle_n_particles": exp_params.particle_params.n_particles,

            # MotionModelParams
            "motion_sigma_x": exp_params.motion_model_params.sigma_x,
            "motion_sigma_y": exp_params.motion_model_params.sigma_y,
            "motion_sigma_theta": exp_params.motion_model_params.sigma_theta,
            "motion_wheel_separation": exp_params.motion_model_params.wheel_separation,
            "motion_ctrl_motion_fac": exp_params.motion_model_params.ctrl_motion_fac,
            "motion_ctrl_turn_fac": exp_params.motion_model_params.ctrl_turn_fac,

            # MeasurementModelParams
            "occ_thresh": exp_params.measurement_model_params.occ_thresh,
            "free_thresh": exp_params.measurement_model_params.free_thresh,
            "unknown_ratio_thresh": exp_params.measurement_model_params.unknown_ratio_thresh,
            "known_free_ratio_thresh": exp_params.measurement_model_params.known_free_ratio_thresh,
            
            "sigma_hit": exp_params.measurement_model_params.sigma_hit,
            "w_hit": exp_params.measurement_model_params.w_hit,
            "w_short": exp_params.measurement_model_params.w_short,
            "lambda_short": exp_params.measurement_model_params.lambda_short,
            "w_max": exp_params.measurement_model_params.w_max,
            "w_rand": exp_params.measurement_model_params.w_rand,
            
            "p_unknown": exp_params.measurement_model_params.p_unknown,
            "p_out_of_map": exp_params.measurement_model_params.p_out_of_map,
            "p_unexpected_known_free": exp_params.measurement_model_params.p_unexpected_known_free,
            "p_pred_below_min": exp_params.measurement_model_params.p_pred_below_min,
            
            "alpha_meas": exp_params.measurement_model_params.alpha_meas,
            "beam_step": exp_params.measurement_model_params.beam_step,
            "eps": exp_params.measurement_model_params.eps,

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
        keep_step_results: bool = False,
        run_storage_dir: Optional[str] = None,
        store_map_data: bool = False,
    ) -> Tuple[List[RankedRun], float]:
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
            return [], None

        if store_map_data and not run_storage_dir:
            raise RuntimeError(
                "STORE_MAP_DATA is enabled, but run storage directory is not configured."
            )

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

                # Store the final highest-weighted particle map for this run.
                if store_map_data:
                    seed_part = str(run_seed) if run_seed is not None else "none"
                    ds_id_part = str(dataset_id) if dataset_id is not None else "unknown_dataset"
                    map_part = str(map_name) if map_name is not None else "unknown_map"
                    run_dir = os.path.join(
                        run_storage_dir,
                        map_part + "_" + ds_id_part + "_" + str(param_hash) + "_" + seed_part,
                    )

                    best_p_map = run_result.best_part_map
                    best_p_map_meta = run_result.best_part_map_meta

                    if best_p_map is not None and best_p_map_meta is not None:
                        MapDataHandler.save(
                            output_dir=run_dir,
                            log_odds_map=best_p_map,
                            resolution=best_p_map_meta.get("grid_resolution_m"),
                            shift_x=best_p_map_meta.get("shift_x"),
                            shift_y=best_p_map_meta.get("shift_y"),
                            occupied_threshold=params.measurement_model_params.occ_thresh,
                            free_threshold=params.measurement_model_params.free_thresh,
                            min_log_odds=params.occupancy_params.min_log_odds,
                            max_log_odds=params.occupancy_params.max_log_odds,
                        )

                # Store run results
                ranked_runs.append(
                    RankedRun(
                        params=params,
                        summary=run_result.summary,
                        score=score,
                        step_results=run_result.step_results if keep_step_results else [],
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
        
        return ranked_runs, optm_duration_s
    


    def optimize_parallel(
        self,
        playback_data: PlaybackData,
        param_grid: Iterable[ExperimentParams],
        seeds: Optional[Iterable[int]] = None,
        dataset_id: Optional[str] = None,
        map_name: Optional[str] = None,
        use_seed_list_for_measurement_noise: bool = True,
        max_workers: Optional[int] = None,
        keep_step_results: bool = False,
        run_storage_dir: Optional[str] = None,
        store_map_data: bool = False,
    ) -> List[RankedRun]:
        # Convert params and seeds
        params_list = list(param_grid)
        seed_list = [int(s) for s in seeds] if seeds is not None else [None]

        # Data safety check
        if not seed_list:
            seed_list = [None]

        if not params_list:
            print("No parameter combinations provided. Nothing to optimize.")
            return [], None

        # Compute number of runs
        total_n_runs = len(params_list) * len(seed_list)

        # Define number of workers to use
        if max_workers is None:
            cpu_count = os.cpu_count() or 1
            max_workers = max(1, cpu_count - 1)

        print(
            f"Starting PARALLEL scan-matching optimization with "
            f"{total_n_runs} run(s), max_workers={max_workers}..."
        )

        # Define storage for jobs that needs to be done by workers
        jobs = []

        # Define jobs
        for params in params_list:
            for run_seed in seed_list:
                jobs.append(
                    {
                        "params": params,
                        "seed": run_seed,
                        "dataset_id": dataset_id,
                        "map_name": map_name,
                        "use_seed_list_for_measurement_noise": use_seed_list_for_measurement_noise,
                        "keep_step_results": keep_step_results,
                    }                    
                )

        # Measure starting time
        start_time = time.perf_counter()
        ranked_runs: List[RankedRun] = []

        # Define mp start method 
        if "fork" in mp.get_all_start_methods():
            mp_context = mp.get_context("fork")
        else:
            mp_context = None

        # Define arguments for ProcessPoolExecutor
        executor_kwargs = {
            "max_workers": max_workers,
            # Init is called by executor for each worker process. We got one central storage for the playback data, 
            # so we only need to send it once at the beginning of each worker process.
            "initializer": _init_rbpf_worker,
            # This is the argument for the init function. Define multiple ones here if u wanne extend init function
            "initargs": (playback_data, run_storage_dir, store_map_data),
        }

        if mp_context is not None:
            executor_kwargs["mp_context"] = mp_context

        # Measure progress
        with tqdm(
            total=total_n_runs,
            desc="Scan matching optimization parallel",
            unit="run",
        ) as pbar:
            pbar.refresh()

            # Run wroker processes
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                # futures = [
                #     executor.submit(_run_rbpf_job, job)
                #     for job in jobs
                # ]
                futures = []

                for job in jobs:
                    future = executor.submit(_run_rbpf_job, job)
                    futures.append(future)

                for future in as_completed(futures):
                    ranked_run = future.result()
                    ranked_runs.append(ranked_run)
                    pbar.update(1)


        # Compute optimization time and print 
        end_time = time.perf_counter()
        optm_duration_s = end_time - start_time

        print(
            f"Finished PARALLEL Scan Matcher optimization: "
            f"{total_n_runs}/{total_n_runs} runs in {optm_duration_s:.2f}s"
        )

        # Sort individual ranked runs by score
        ranked_runs.sort(key=lambda x: x.score)

        return ranked_runs, optm_duration_s



class ScanMatcherOptimizer(RBPFOptimizer):
    """
    Backward-compatible alias used by existing imports in the RBPF tuning script.
    """
    pass
