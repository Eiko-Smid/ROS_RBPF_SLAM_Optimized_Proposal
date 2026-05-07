#!/usr/bin/env python3

import time
from typing import Any, Dict

import numpy as np

from .evaluator_scanmatching import RunResultScanMatching, ScanMatchingEvaluator
from ..optimize_rbpf.playback_defs import ExperimentParams, PlaybackData
from ..rbpf.rbpf import RBPFFactory
from ..rbpf.scan_match_factory import ScanMatchFactory


class PlaybackRunnerScanMatching:
    """
    Runs one scan-matching-only playback experiment for a single parameter set.
    """

    def __init__(self, factory: RBPFFactory, evaluator: ScanMatchingEvaluator):
        self.factory = factory
        self.evaluator = evaluator

    @staticmethod
    def _aggregate_icp_counters(rbpf: Any) -> Dict[str, int]:
        # Keep the same counter set as in the old tuning pipeline.
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

        reason_to_counter = {
            "Too few input points": "count_too_few_points",
            "Too few correspondences in first iteration": "count_too_few_corresp",
            "Too few correspondences": "count_too_few_corresp",
            "Non-finite H or g": "infinite_h_or_g",
            "Ill-conditioned Hessian": "ill_cond_H",
            "Non-finite transformation update": "infinite_dtransform",
            "Infinite mean error": "infinite_mean_err",
            "Best Transformation too large": "best_transf_too_large",
            "Best mean error too large": "best_mean_err_too_large",
        }

        totals = {key: 0 for key in counter_keys}

        mapped_totals = {key: 0 for key in counter_keys}

        icp_total_runs = 0
        icp_success_count = 0

        for particle in getattr(rbpf, "particles", []):
            icp_obj = particle.scan_matcher.icp
            icp_info = icp_obj.get_info()
            legacy_counters = getattr(icp_obj, "legacy_counters", {}) or {}

            for key in counter_keys:
                totals[key] += int(legacy_counters.get(key, icp_info.get(key, 0)) or 0)

            reason_counts = icp_info.get("stop_reason_counts", {}) or {}
            if isinstance(reason_counts, dict):
                for reason, count in reason_counts.items():
                    mapped_key = reason_to_counter.get(reason)
                    if mapped_key is not None:
                        mapped_totals[mapped_key] += int(count or 0)

                particle_total_runs = sum(int(v or 0) for v in reason_counts.values())
                particle_success = int(reason_counts.get("All safety checks passed", 0) or 0)
                icp_total_runs += particle_total_runs
                icp_success_count += particle_success

        # Prefer whichever source reports more events.
        for key in counter_keys:
            totals[key] = max(int(totals[key]), int(mapped_totals[key]))

        totals["icp_total_runs"] = int(icp_total_runs)
        totals["icp_success_count"] = int(icp_success_count)
        totals["icp_failed_count"] = int(max(0, icp_total_runs - icp_success_count))

        return totals


    @staticmethod
    def _merge_summary_dict(summary: Any, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if hasattr(summary, key):
                setattr(summary, key, value)


    def run(self, playback_data: PlaybackData, params: ExperimentParams) -> RunResultScanMatching:
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
        run_result = RunResultScanMatching(params=params)

        every_nth_scan_filter = max(1, int(params.every_nth_scan_filter))
        every_nth_scan_map = max(1, int(params.every_nth_scan_map))

        print(
            f"Running scan-matching-only mode with params: {params.tag} "
            f"(every_nth_scan_filter={every_nth_scan_filter}, every_nth_scan_map={every_nth_scan_map})"
        )

        for step_idx, step in enumerate(steps):
            step_start_time = time.time()

            measurements_filter = (
                step.scan[::every_nth_scan_filter]
                if every_nth_scan_filter > 1
                else step.scan
            )

            # For map update we want to use inf values for freeing space!
            measurements_map_update = (
                step.scan[::every_nth_scan_map]
                if every_nth_scan_map > 1
                else step.scan
            )

            measurements_filter = [
                (r, b) for r, b in measurements_filter if np.isfinite(r) and not np.isnan(r)
            ]
            # measurements_map_update = [
            #     (r, b) for r, b in measurements_map_update if np.isfinite(r) and not np.isnan(r)
            # ]

            rbpf.step_scan_match_only(
                odom=(step.dl, step.dr),
                measurements_filter=measurements_filter,
                measurements_map_update=measurements_map_update,
            )

            step_duration = time.time() - step_start_time

            rbpf_sc_only_info = rbpf.get_step_info_scan_match_only()
            icp_info = rbpf.particles[0].scan_matcher.icp.get_info()
            scan_match_info = rbpf.particles[0].scan_matcher.get_info()
            scan_match_failed = bool(rbpf_sc_only_info.get("scan_match_failed", False))
            step_stop_reason = icp_info.get("stop_reason")

            # If scan matching failed before starting ICP, note that!
            if scan_match_failed and scan_match_info.get("timing_correct_pose") is None:
                step_stop_reason = "scan matcher failed before icp"

            step_result = self.evaluator.evaluate_step(
                step_idx=rbpf_sc_only_info.get("step") if rbpf_sc_only_info.get("step") is not None else step_idx,
                t=step.t,
                true_pose=step.true_pose,
                pred_pose=scan_match_info.get("pred_pose"),
                corr_pose=rbpf_sc_only_info.get("particle_pose"),
                best_transformation=icp_info.get("best_transformation"),
                icp_iterations=icp_info.get("icp_iterations"),
                icp_mean_error=icp_info.get("icp_mean_error"),
                n_correspondences=icp_info.get("n_correspondences"),
                use_transformation=icp_info.get("use_transformation"),
                stop_reason=step_stop_reason,
                n_measurements_total=len(step.scan),
                n_valid_measurements_filter=len(measurements_filter),
                n_valid_measurements_map_update=len(measurements_map_update),
                n_map_points_extracted=scan_match_info.get("map_points_count"),
                t_ogm=rbpf_sc_only_info.get("timing_ogm_update"),
                t_scan_matching=scan_match_info.get("timing_scan_matching"),
                t_prediction=scan_match_info.get("timing_prediction"),
                t_map_extraction=scan_match_info.get("timing_map_extraction"),
                t_correct_pose=scan_match_info.get("timing_correct_pose"),
                scan_match_failed=scan_match_failed,
                step_duration=step_duration,
                timing_update_particle=rbpf_sc_only_info.get("timing_update_particle"),
            )
            run_result.step_results.append(step_result)

        # Summarize the results of the entire run
        run_result.summary = self.evaluator.summarize_run(
            step_results=run_result.step_results,
            params=params,
        )

        self._merge_summary_dict(run_result.summary, self._aggregate_icp_counters(rbpf))
        self._merge_summary_dict(run_result.summary, rbpf.timing_summary_scan_match_only())

        return run_result
