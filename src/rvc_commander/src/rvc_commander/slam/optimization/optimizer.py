from dataclasses import dataclass
from typing import List
from scan_match_playback_def import ExperimentParams
from playback_runner import PlaybackRunner
from evaluator import ScanMatcherEvaluator
from scorer import RunScorer


@dataclass
class RankedRun:
    params: ExperimentParams
    summary: dict
    score: float


class ScanMatcherOptimizer:
    def __init__(self, runner: PlaybackRunner, scorer: RunScorer):
        self.runner = runner
        self.scorer = scorer

    def optimize(self, playback_data, param_grid) -> list[RankedRun]:
        ranked = []

        for params in param_grid:
            run_result = self.runner.run(playback_data, params)
            score = self.scorer.score(run_result.summary)

            ranked.append(
                RankedRun(
                    params=params,
                    summary=run_result.summary,
                    score=score,
                )
            )

        ranked.sort(key=lambda x: x.score)
        return ranked