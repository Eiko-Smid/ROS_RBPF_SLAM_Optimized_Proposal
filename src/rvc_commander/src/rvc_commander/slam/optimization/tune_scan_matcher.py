#!/usr/bin/env python3

import itertools

def generate_param_grid():
    max_corrs = [0.2, 0.4, 0.6, 0.8]
    neighbors = [5, 8, 10, 15]
    occ_thres = [1.0, 2.0, 3.0]
    delta_r = [0.0, 0.2, 0.5]

    for max_corr, k, occ, delta_r_ in itertools.product(
        max_corrs, neighbors, occ_thres, delta_r
    ):
        yield ExperimentParams(
            icp=ICPParams(
                max_correspondence_distance=max_corr,
                neighbors_pca=k,
                max_iterations=10,
                epsilon_rel=1e-3,
                no_improvement_limit=2,
                min_error=1.0,
                min_dtrans=1e-4,
                min_drot=1e-1,
            ),
            scan_matcher=ScanMatcherParams(
                occ_thres=occ,
                delta_r=delta_r_,
            ),
            tag=f"corr{max_corr}_k{k}_occ{occ}_dr{delta_r_}",
        )