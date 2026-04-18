from dataclasses import dataclass
from typing import List, Iterable
import time

from tqdm import tqdm

from ..scan_match_playback_def import ExperimentParams, PlaybackData
from .playback_runner import PlaybackRunner
from .evaluator import ScanMatcherEvaluator
from .scorer import RunScorer


@dataclass
class RankedRun:
    params: ExperimentParams
    summary: dict
    score: float


class ScanMatcherOptimizer:
    '''
    Class for optimizing the scan matcher parameters using playback data and a parameter grid.
    '''
    def __init__(self, runner: PlaybackRunner, scorer: RunScorer):
        self.runner = runner
        self.scorer = scorer

    def optimize(self, playback_data: PlaybackData, param_grid: Iterable[ExperimentParams]) -> List[RankedRun]:
        '''
        Get's all steps of the playback data and the parameter grid and runs the scan matcher for each parameter set.
        Computes a score for each runs and appends it to a list of ranked runs. Finally, it sorts the list by score and
        returns it.

        Parameters
        ----------
        playback_data: PlaybackData
            The data to run the scan matcher on.
        param_grid: Iterable[ExperimentParams]
            The parameter grid to run the scan matcher with.
        
        Returns
        -------
        list of RankedRun
            A list of ranked runs, sorted by score.
        '''
        # Transform grid to list
        params_list = list(param_grid)
        total_runs = len(params_list)

        # Check if we have any parameters to optimize
        if total_runs == 0:
            print("No parameter combinations provided. Nothing to optimize.")
            return []

        print(f"Starting optimization with {total_runs} runs...")
        ranked_runs = []

        # Measure start time
        start_time = time.time()

        # Do scan matching for each parameters set and compute score for each run
        for params in tqdm(params_list, total=total_runs, desc="Optimization runs", unit="run"):
            run_result = self.runner.run(playback_data, params)
            score = self.scorer.score(run_result.summary)

            ranked_runs.append(
                RankedRun(
                    params=params,
                    summary=run_result.summary,
                    score=score,
                )
            )

        # Compute and print elapsed time
        elapsed_s = time.time() - start_time
        print(f"Finished optimization: {total_runs}/{total_runs} runs in {elapsed_s:.2f}s")

        # Sort runs by score  
        ranked_runs.sort(key=lambda x: x.score)
        
        return ranked_runs

