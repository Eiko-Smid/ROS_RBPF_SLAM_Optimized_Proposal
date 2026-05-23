from dataclasses import dataclass
import time
from typing import Iterable, List, Optional

import numpy as np
from tqdm import tqdm

from ..optimize_rbpf.playback_defs import ExperimentParams, PlaybackData
from ..optimize_rbpf.playback_defs import StepData
from ..infrastructure.playback_converter import PlaybackConverter
from .playback_runner_scanmatching import PlaybackRunnerScanMatching
from .scorer_scanmatching import ScanMatchingScorer
from .evaluator_scanmatching import RunSummaryScanMatching, StepResultScanMatching


@dataclass
class RankedRunScanMatching:
    params: ExperimentParams
    summary: RunSummaryScanMatching
    score: float
    step_results: List[StepResultScanMatching]
    seed: Optional[int]


class ScanMatchingOptimizer:
    def __init__(self, runner: PlaybackRunnerScanMatching, scorer: ScanMatchingScorer):
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
        Recreate playback scans with deterministic noise for the given run seed.
        The source playback_data must contain clean (non-noised) scans.
        """
        if measurement_stddev is None:
            return playback_data

        # Use a dedicated RNG for measurement noise to keep it independent from
        # global numpy seeding used elsewhere in the pipeline.
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
    ) -> List[RankedRunScanMatching]:
        params_list = list(param_grid)
        seed_list = [int(s) for s in seeds] if seeds is not None else [None]

        if not seed_list:
            seed_list = [None]

        if not params_list:
            print("No parameter combinations provided. Nothing to optimize.")
            return []

        total_n_runs = len(params_list) * len(seed_list)
        print(f"Starting RBPF optimization with {total_n_runs} run(s)...")
        ranked_runs: List[RankedRunScanMatching] = []

        # Measure starting time
        start_time = time.perf_counter()

        for params in tqdm(
            params_list,
            total=len(params_list),
            desc="Scan matching optimization",
            unit="param",
        ):
            for run_seed in seed_list:
                if run_seed is not None:
                    np.random.seed(run_seed)

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

                # Test if seed works
                # prob_val = np.random.normal(1.5, 1.0)
                # print(f"Seed {run_seed} produced value: {prob_val}")

                run_result = self.runner.run(run_playback_data, params)
                score = self.scorer.score(run_result.summary)

                ranked_runs.append(
                    RankedRunScanMatching(
                        params=params,
                        summary=run_result.summary,
                        score=score,
                        step_results=run_result.step_results,
                        seed=run_seed,
                    )
                )
        # Measure ending time and print info
        end_time = time.perf_counter()
        optm_duration_s = end_time - start_time
        print(f"Finished RBPF optimization: {total_n_runs}/{total_n_runs} runs in {optm_duration_s:.2f}s")

        # Sort runs by score (ascending order)
        ranked_runs.sort(key=lambda x: x.score)
        
        return ranked_runs
