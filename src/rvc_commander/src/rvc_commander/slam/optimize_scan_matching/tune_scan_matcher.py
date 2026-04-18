#!/usr/bin/env python3

import itertools

from ..scan_match_playback_def import ExperimentParams, ICPParams, ScanMatcherParams, PlaybackData
from ..playback_loader import load_playback_dataset

from .evaluator import ScanMatcherEvaluator
from .factory import ScanMatcherFactory
from .playback_runner import PlaybackRunner
from .scorer import RunScorer
from .optimizer import ScanMatcherOptimizer
from .result_writer import ResultWriter


PLAYBACK_DATA_PATH_PREF = '/home/smide/work/ros_workspaces/ros_ws/src/rvc_commander/data/scan_match/python_playback/1776425398_python_playback'
OPTIMIZATION_RESULT_PATH= '/home/smide/work/ros_workspaces/ros_ws/src/rvc_commander/data/scan_match/optimization_results/1776425398_fix_err_v3.csv'


def generate_param_grid():
    '''
    Defined the parameter grid for the scan matcher optimization. This is a generator that yields ExperimentParams for
    each combination of parameters in the grid.
    '''
    max_corrs = [0.6]               # Max correspondence distance for ICP
    neighbors = [10]                # Number of neighbors for PCA in ICP
    occ_thres = [49.0]              # Occupancy threshold for scan matcher. Considers only cells with log-odds above this threshold
    delta_r = [0.6]                 # We search in circular area around the pred robots pose (max_sensor_range+dr) to extract the map
    max_n_points = [400]            # The true pointclous data will be subsampled to this amount, in every run (before outlier rejection, etc)

    for max_corr, k, occ, delta_r_, max_n in itertools.product(
        max_corrs, neighbors, occ_thres, delta_r, max_n_points
    ):
        yield ExperimentParams(
            icp=ICPParams(
                max_n_points=max_n,
                max_correspondence_distance=max_corr,
                neighbors_pca=k,
                max_iterations=6,      # TODO: test with reduced max iterations (10->6)
                epsilon_rel=1e-3,
                no_improvement_limit=3,
                min_error=5e-4,
                min_dtrans=1e-3,
                min_drot=1e-2,
            ),
            scan_matcher=ScanMatcherParams(
                occ_thres=occ,
                delta_r=delta_r_,
            ), 
            tag=f"corr{max_corr}_k{k}_occ{occ}_dr{delta_r_}_maxn{max_n}",
        )


def build_optimizer():
    # Init objects
    # Init Playback runner
    scan_match_fac = ScanMatcherFactory()
    scan_match_eval = ScanMatcherEvaluator()
    scan_match_playback_run = PlaybackRunner(
        factory=scan_match_fac,
        evaluator=scan_match_eval,
    )

    # Init optimizer
    run_scorer = RunScorer()
    scan_match_optimizer = ScanMatcherOptimizer(
        runner=scan_match_playback_run,
        scorer=run_scorer,
    )
    
    return scan_match_optimizer



def main():
    # Load playback data
    meta_data, steps = load_playback_dataset(base_path_prefix=PLAYBACK_DATA_PATH_PREF)

    # Build playback data
    playback_data = PlaybackData(
        meta_data=meta_data,
        step_data_list=steps,
    )

    # Init optimizer
    scan_match_optimizer = build_optimizer()

    # Build result writer
    result_writer = ResultWriter()

    # Run optimizer
    ranked_runs = scan_match_optimizer.optimize(
        playback_data=playback_data,
        param_grid=generate_param_grid(),
    )

    # Save results
    result_writer.write_ranked_runs_csv(
        path=OPTIMIZATION_RESULT_PATH,
        ranked_runs=ranked_runs,
        override=False
    )

    print("Test success")

    


if __name__ == "__main__":
    main()