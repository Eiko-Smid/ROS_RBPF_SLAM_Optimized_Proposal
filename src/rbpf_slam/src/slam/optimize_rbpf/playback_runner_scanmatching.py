#!/usr/bin/env python3

import time
import numpy as np

from .evaluator import RunResult, RBPFEvaluator
from .playback_defs import ExperimentParams
from ..rbpf.rbpf import RBPFFactory
from ..rbpf.scan_match_factory import ScanMatchFactory


class PlaybackRunnerScanMatching:
    """
    Runs one scan-matching-only playback experiment for a single parameter set.
    """

    def __init__(self, factory: RBPFFactory, evaluator: RBPFEvaluator):
        self.factory = factory
        self.evaluator = evaluator

    @staticmethod
    def _aggregate_icp_counters(rbpf) -> dict:
        counter_keys = [
            "count_too_few_points",
            "count_too_few_corresp",
            "infinite_h_or_g",
            "ill_cond_H",
            "infinite_dtransform",
            "infinite_mean_err",
            "best_transf_too_large",
            "best_mean_err_too_large",
        ]

        totals = {key: 0 for key in counter_keys}

        for particle in getattr(rbpf, "particles", []):
            icp_info = particle.scan_matcher.icp.get_info()
            for key in counter_keys:
                totals[key] += int(icp_info.get(key, 0) or 0)

        return totals

    def run(self, playback_data, params: ExperimentParams) -> RunResult:
        rbpf = self.factory.create(
            scan_match_fac=ScanMatchFactory(),
            particle_params=params.particle_params,
            occ_param=params.occupancy_params,
            sens_params=params.sensor_params,
            map_param=params.map_param,
            icp_params=params.icp_params,
            robot_params=params.robot_params,
            scan_matcher_params=params.scan_matcher_params,
            motion_model_params=params.motion_model_params,
            measurement_model_params=params.measurement_model_params,
        )

        steps = playback_data.step_data_list
        run_result = RunResult(params=params)

        every_nth_filter = max(1, int(params.every_nth_scan_filter))
        print(
            f"Running scan-matching-only mode with params: {params.tag} "
            f"(every_nth_scan_filter={every_nth_filter})"
        )

        for step_idx, step in enumerate(steps):
            step_start_time = time.time()

            measurements = step.scan[::every_nth_filter] if every_nth_filter > 1 else step.scan
            measurements = [
                (r, b) for r, b in measurements if np.isfinite(r) and not np.isnan(r)
            ]

            rbpf.step_scan_match_only(
                odom=(step.dl, step.dr),
                measurements=measurements,
                true_pose=step.true_pose,
            )

            step_duration = time.time() - step_start_time

            info = rbpf.get_step_info()
            step_result = self.evaluator.evaluate_step(
                step_idx=info.get("step") if info.get("step") is not None else step_idx,
                t=step.t,
                true_pose=info.get("true_pose") if info.get("true_pose") is not None else step.true_pose,
                est_pose=info.get("weighted_mean_pose"),
                best_particle_pose=info.get("best_particle_pose"),
                scan_match_failed=info.get("scan_match_failed_any"),
                scan_match_fallback_failed=info.get("scan_match_fallback_failed_any"),
                neff=info.get("neff"),
                particle_weight_min=info.get("particle_weight_min"),
                particle_weight_max=info.get("particle_weight_max"),
                particle_weight_mean=info.get("particle_weight_mean"),
                step_duration=step_duration,
            )
            run_result.step_results.append(step_result)

        run_result.summary = self.evaluator.summarize_run(
            step_results=run_result.step_results,
            params=params,
        )
        run_result.summary.update(self._aggregate_icp_counters(rbpf))

        return run_result
