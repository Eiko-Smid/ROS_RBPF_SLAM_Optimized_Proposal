from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from tqdm import tqdm

from ..optimize_rbpf.playback_defs import ExperimentParams, PlaybackData
from .playback_runner_scanmatching import PlaybackRunnerScanMatching
from .scorer_scanmatching import ScanMatchingScorer


@dataclass
class RankedRunScanMatching:
    params: ExperimentParams
    summary: Dict[str, Any]
    score: float
    step_results: List[Any]


class ScanMatchingOptimizer:
    def __init__(self, runner: PlaybackRunnerScanMatching, scorer: ScanMatchingScorer):
        self.runner = runner
        self.scorer = scorer

    def optimize(
        self,
        playback_data: PlaybackData,
        param_grid: Iterable[ExperimentParams],
        base_seed: Optional[int] = None,
        reseed_each_run: bool = False,
    ) -> List[RankedRunScanMatching]:
        params_list = list(param_grid)

        if not params_list:
            print("No parameter combinations provided. Nothing to optimize.")
            return []

        ranked_runs: List[RankedRunScanMatching] = []

        for run_idx, params in enumerate(
            tqdm(params_list, total=len(params_list), desc="Scan matching optimization", unit="run")
        ):
            if base_seed is not None:
                run_seed = base_seed if reseed_each_run else (base_seed + run_idx)
                np.random.seed(run_seed)

            run_result = self.runner.run(playback_data, params)
            score = self.scorer.score(run_result.summary)

            ranked_runs.append(
                RankedRunScanMatching(
                    params=params,
                    summary=run_result.summary,
                    score=score,
                    step_results=run_result.step_results,
                )
            )

        ranked_runs.sort(key=lambda x: x.score)
        return ranked_runs
