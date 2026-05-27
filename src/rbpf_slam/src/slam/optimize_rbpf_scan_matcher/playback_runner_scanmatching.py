#!/usr/bin/env python3

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .evaluator_scanmatching import RunResultScanMatching, ScanMatchingEvaluator
from ..optimize_rbpf.playback_defs import ExperimentParams, PlaybackData
from ..rbpf.rbpf import RBPFFactory
from ..rbpf.motion_model import MotionModel
from ..rbpf.scan_match_factory import ScanMatchFactory


Pose2D = Tuple[float, float, float]


class RawOdometryPropagator:
    """Computes raw odometry poses using a fresh motion model per run."""

    def estimate(self, steps: List[Any], start_pose: Pose2D, wheel_separation: float) -> List[Pose2D]:
        pose = (float(start_pose[0]), float(start_pose[1]), float(start_pose[2]))
        odom_poses: List[Pose2D] = []
        motion_model = MotionModel(wheel_separation=float(wheel_separation))

        for step in steps:
            pose = motion_model.predict_pose(pose=pose, dl=step.dl, dr=step.dr)
            odom_poses.append((float(pose[0]), float(pose[1]), float(pose[2])))

        return odom_poses


class PlaybackRunnerScanMatching:
    """
    Runs one scan-matching-only playback experiment for a single parameter set.
    """

    def __init__(self, factory: RBPFFactory, evaluator: ScanMatchingEvaluator, raw_odom_propagator: Optional[RawOdometryPropagator] = None):
        self.factory = factory
        self.evaluator = evaluator
        self.raw_odom_propagator = raw_odom_propagator or RawOdometryPropagator()
        self._raw_odom_cache_key: Optional[Tuple[int, Optional[float], Optional[float], float, float, float, float]] = None
        self._raw_odom_poses_cache: Optional[List[Pose2D]] = None


    @staticmethod
    def _build_raw_odom_cache_key(
        playback_data: PlaybackData,
        start_pose: Pose2D,
        wheel_separation: float,
    ) -> Tuple[int, Optional[float], Optional[float], float, float, float, float]:
        """
        Build a cache key for raw-odometry baseline reuse.

        Raw odometry depends on playback, start pose, and wheel separation.
        """
        steps = playback_data.step_data_list
        n_steps = len(steps)
        x0, y0, theta0 = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
        wheel_sep = float(wheel_separation)
        if n_steps == 0:
            return (0, None, None, x0, y0, theta0, wheel_sep)
        return (n_steps, float(steps[0].t), float(steps[-1].t), x0, y0, theta0, wheel_sep)


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

        # Compute raw-odometry baseline once per unique input set and reuse it
        # across optimization runs to avoid repeated deterministic work.
        wheel_separation = float(params.robot_params.wheel_separation)
        cache_key = self._build_raw_odom_cache_key(
            playback_data=playback_data,
            start_pose=params.particle_params.start_pose,
            wheel_separation=wheel_separation,
        )
        if self._raw_odom_poses_cache is None or self._raw_odom_cache_key != cache_key:
            self._raw_odom_poses_cache = self.raw_odom_propagator.estimate(
                steps=steps,
                start_pose=params.particle_params.start_pose,
                wheel_separation=wheel_separation,
            )
            self._raw_odom_cache_key = cache_key

        # Ensure valid scan downsampling values for filter/proposal and map update.
        every_nth_scan_filter = max(1, int(params.every_nth_scan_filter))
        every_nth_scan_map = max(1, int(params.every_nth_scan_map))

        print(
            f"Running scan-matching-only mode with params: {params.tag} "
            f"(every_nth_scan_filter={every_nth_scan_filter}, every_nth_scan_map={every_nth_scan_map})"
        )

        # Process playback data 
        for step_idx, step in enumerate(steps):
            step_start_time = time.time()
            step_duration = None

            # Filter measruements for scan matching and map update
            measurements_filter = (
                step.scan[::every_nth_scan_filter]
                if every_nth_scan_filter > 1
                else step.scan
            )

            measurements_map_update = (
                step.scan[::every_nth_scan_map]
                if every_nth_scan_map > 1
                else step.scan
            )

            # Filter inf values (only for filter, cause for map update we need inf information)
            measurements_filter = [
                (r, b) for r, b in measurements_filter if np.isfinite(r) and not np.isnan(r)
            ]
            # measurements_map_update = [
            #     (r, b) for r, b in measurements_map_update if np.isfinite(r) and not np.isnan(r)
            # ]

            # Check if initialization done


            # Run  RBPF filter step with scan matching only
            _, _ = rbpf.step_scan_match_only(
                odom=(step.dl, step.dr),
                measurements_filter=measurements_filter,
                measurements_map_update=measurements_map_update,
            )

            step_duration = time.time() - step_start_time

            rbpf_sc_only_info = rbpf.get_step_info_scan_match_only()
            icp_info = rbpf.particles[0].scan_matcher.icp.get_info()
            scan_match_info = rbpf.particles[0].scan_matcher.get_info()
            mode = rbpf_sc_only_info.get("mode")
            is_initialization_step = (mode == "initialization")

            scan_match_failed = (
                None if is_initialization_step else bool(rbpf_sc_only_info.get("scan_match_failed", False))
            )
            step_stop_reason = icp_info.get("stop_reason")

            # If scan matching failed before starting ICP, note that!
            if scan_match_failed and scan_match_info.get("time_duration_correct_pose") is None:
                step_stop_reason = "scan matcher failed before icp"

            step_result = self.evaluator.evaluate_step(
                step_idx=rbpf_sc_only_info.get("step") if rbpf_sc_only_info.get("step") is not None else step_idx,
                t=step.t,
                true_pose=step.true_pose,
                # Inject baseline pose for direct comparison against scan matcher outputs.
                raw_odom_pose=(
                    self._raw_odom_poses_cache[step_idx]
                    if self._raw_odom_poses_cache is not None and step_idx < len(self._raw_odom_poses_cache)
                    else None
                ),
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
                n_map_points_extracted=(
                    None if is_initialization_step else scan_match_info.get("map_points_count")
                ),
                t_ogm=rbpf_sc_only_info.get("timing_ogm_update"),
                t_scan_matching=scan_match_info.get("time_duration_scan_matching"),
                t_prediction=scan_match_info.get("time_duration_prediction"),
                t_map_extraction=scan_match_info.get("time_duration_map_extraction"),
                t_correct_pose=scan_match_info.get("time_duration_correct_pose"),
                scan_match_failed=scan_match_failed,
                step_duration=step_duration,
                t_update_particle=rbpf_sc_only_info.get("timing_update_particle"),
            )
            run_result.step_results.append(step_result)

        # Summarize the results of the entire run
        run_result.summary = self.evaluator.summarize_run(
            step_results=run_result.step_results,
            params=params,
        )

        self._merge_summary_dict(run_result.summary, self._aggregate_icp_counters(rbpf))
        self._merge_summary_dict(run_result.summary, rbpf.time_summary_scan_match_only())

        return run_result
