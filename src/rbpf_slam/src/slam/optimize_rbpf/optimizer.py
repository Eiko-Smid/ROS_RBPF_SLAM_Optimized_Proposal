from dataclasses import dataclass
from typing import Iterable, List, Optional
import time

import numpy as np

from tqdm import tqdm

from .playback_runner import PlaybackRunner
from .scorer import RunScorer
from .playback_defs import ExperimentParams


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

    def optimize(
        self,
        playback_data,
        param_grid: Iterable[ExperimentParams],
        seeds: Optional[Iterable[int]] = None,
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

                run_result = self.runner.run(playback_data, params)
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
    

    def optimize_without_proposal_pose(
        self,
        playback_data,
        param_grid: Iterable[ExperimentParams],
        seeds: Optional[Iterable[int]] = None,
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

                run_result = self.runner.run_without_proposal_pose(playback_data, params)
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