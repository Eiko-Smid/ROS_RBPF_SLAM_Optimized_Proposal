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
        base_seed: Optional[int] = None,
        reseed_each_run: bool = False,
    ) -> List[RankedRun]:
        """
        Runs the RBPF once per parameter set and ranks all runs by score (lower is better).
        """
        params_list = list(param_grid)
        total_runs = len(params_list)

        if total_runs == 0:
            print("No parameter combinations provided. Nothing to optimize.")
            return []

        print(f"Starting RBPF optimization with {total_runs} runs...")
        ranked_runs: List[RankedRun] = []

        start_time = time.time()

        for run_idx, params in enumerate(
            tqdm(params_list, total=total_runs, desc="RBPF optimization", unit="run")
        ):
            if base_seed is not None:
                run_seed = base_seed if reseed_each_run else (base_seed + run_idx)
                np.random.seed(run_seed)

            run_result = self.runner.run(playback_data, params)
            score = self.scorer.score(run_result.summary)

            ranked_runs.append(
                RankedRun(
                    params=params,
                    summary=run_result.summary,
                    score=score,
                    step_results=run_result.step_results,
                )
            )

        elapsed_s = time.time() - start_time
        print(f"Finished RBPF optimization: {total_runs}/{total_runs} runs in {elapsed_s:.2f}s")

        ranked_runs.sort(key=lambda x: x.score)
        return ranked_runs
    

    def optimize_without_proposal_pose(
        self,
        playback_data,
        param_grid: Iterable[ExperimentParams],
        base_seed: Optional[int] = None,
        reseed_each_run: bool = False,
    ) -> List[RankedRun]:
        """
        Runs the RBPF once per parameter set and ranks all runs by score (lower is better).
        """
        params_list = list(param_grid)
        total_runs = len(params_list)

        if total_runs == 0:
            print("No parameter combinations provided. Nothing to optimize.")
            return []

        print(f"Starting RBPF optimization with {total_runs} runs...")
        ranked_runs: List[RankedRun] = []

        start_time = time.time()

        for run_idx, params in enumerate(
            tqdm(params_list, total=total_runs, desc="RBPF optimization", unit="run")
        ):
            if base_seed is not None:
                run_seed = base_seed if reseed_each_run else (base_seed + run_idx)
                np.random.seed(run_seed)

            run_result = self.runner.run_without_proposal_pose(playback_data, params)
            score = self.scorer.score(run_result.summary)

            ranked_runs.append(
                RankedRun(
                    params=params,
                    summary=run_result.summary,
                    score=score,
                    step_results=run_result.step_results,
                )
            )

        elapsed_s = time.time() - start_time
        print(f"Finished RBPF optimization: {total_runs}/{total_runs} runs in {elapsed_s:.2f}s")

        ranked_runs.sort(key=lambda x: x.score)
        return ranked_runs


class ScanMatcherOptimizer(RBPFOptimizer):
    """
    Backward-compatible alias used by existing imports in the RBPF tuning script.
    """
    pass