from dataclasses import dataclass
from typing import Iterable, List, Optional
import time

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


class RBPFOptimizer:
    """
    Optimizes RBPF parameters by replaying one dataset for each parameter set.
    """

    def __init__(self, runner: PlaybackRunner, scorer: RunScorer):
        self.runner = runner
        self.scorer = scorer

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
        seeds: Optional[Iterable[int]] = None,
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

        print(f"Starting RBPF optimization with {total_runs * len(seed_list)} run(s)...")
        ranked_runs: List[RankedRun] = []

        start_time = time.time()

        for params in tqdm(params_list, total=total_runs, desc="RBPF optimization", unit="run"):
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
                    )
                )

        # Measure ending time and print info
        optm_duration_s = time.time() - start_time
        n_runs = total_runs * len(seed_list)
        print(f"Finished RBPF optimization: {n_runs}/{n_runs} runs in {optm_duration_s:.2f}s")
        
        # Sort runs by score (ascending order)
        ranked_runs.sort(key=lambda x: x.score)
        
        return ranked_runs
    

    def optimize_without_proposal_pose(
        self,
        playback_data: PlaybackData,
        param_grid: Iterable[ExperimentParams],
        seeds: Optional[Iterable[int]] = None,
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
                    )
                )

        # Measure ending time and print info
        optm_duration_s = time.time() - start_time
        n_runs = total_runs * len(seed_list)
        print(f"Finished RBPF optimization: {n_runs}/{n_runs} runs in {optm_duration_s:.2f}s")

        # Sort runs by score (ascending order)
        ranked_runs.sort(key=lambda x: x.score)
        return ranked_runs


class ScanMatcherOptimizer(RBPFOptimizer):
    """
    Backward-compatible alias used by existing imports in the RBPF tuning script.
    """
    pass