from dataclasses import dataclass
from typing import Iterable, List
import time

from tqdm import tqdm

from .playback_runner import PlaybackRunner
from .scorer import RunScorer
from .playback_defs import ExperimentParams


@dataclass
class RankedRun:
    params: ExperimentParams
    summary: dict
    score: float


class RBPFOptimizer:
    """
    Optimizes RBPF parameters by replaying one dataset for each parameter set.
    """

    def __init__(self, runner: PlaybackRunner, scorer: RunScorer):
        self.runner = runner
        self.scorer = scorer

    def optimize(self, playback_data, param_grid: Iterable[ExperimentParams]) -> List[RankedRun]:
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

        for params in tqdm(params_list, total=total_runs, desc="RBPF optimization", unit="run"):
            run_result = self.runner.run(playback_data, params)
            score = self.scorer.score(run_result.summary)

            ranked_runs.append(
                RankedRun(
                    params=params,
                    summary=run_result.summary,
                    score=score,
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