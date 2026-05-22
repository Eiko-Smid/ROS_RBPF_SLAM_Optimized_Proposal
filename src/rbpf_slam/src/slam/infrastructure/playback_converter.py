from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .playback_loader import PlaybackData, PlaybackStep, RawScan
from ..optimize_rbpf.playback_defs import StepData, PlaybackData as OptimizationPlaybackData



class PlaybackConverter:
    def convert(
        self,
        playback_data: PlaybackData,
        measurement_stddev: Optional[float] = None,
    ) -> OptimizationPlaybackData:
        conv_playback_steps = []
        for step in playback_data.steps:
            conv_step = StepData(
                t = step.t,
                dl = step.dl,
                dr = step.dr,
                scan=self.convert_scans(step.scan, measurement_stddev=measurement_stddev),
                true_pose=step.true_pose
            )

            conv_playback_steps.append(conv_step)

        return OptimizationPlaybackData(step_data_list=conv_playback_steps)

        
    @staticmethod
    def convert_scans(
        raw_scan: RawScan,
        measurement_stddev: Optional[float] = None,
    ):
        measurements = []
        ranges = raw_scan.ranges

        if measurement_stddev is not None:
            ranges = PlaybackConverter.add_measurement_noise(
                ranges=ranges,
                stddev=measurement_stddev,
            )

        bearing = raw_scan.angle_min
        for r in ranges:
            measurements.append((r, bearing))
            bearing += raw_scan.angle_increment

        return measurements


    @staticmethod
    def add_measurement_noise(ranges, stddev: float):
        '''
        Gets the measurement ranges and the desired standard deviation and add nromal noise with that stddev to the measurement ranges.
        Only add noise to finite values.
        '''
        noisy_ranges = np.array(ranges, dtype=np.float32, copy=True)
        finite_mask = np.isfinite(noisy_ranges)

        if np.any(finite_mask):
            noise = np.random.normal(loc=0.0, scale=stddev, size=int(np.sum(finite_mask)))
            noisy_ranges[finite_mask] = noisy_ranges[finite_mask] + noise

        return noisy_ranges



