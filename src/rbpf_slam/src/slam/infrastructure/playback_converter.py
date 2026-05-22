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
        min_range: float = 0.1,
        max_range: float = 10.0,
    ) -> OptimizationPlaybackData:
        conv_playback_steps = []
        for step in playback_data.steps:
            conv_step = StepData(
                t = step.t,
                dl = step.dl,
                dr = step.dr,

                # Convert raw scan to measurements with optional noise addition and range clipping
                scan=self.convert_scans(
                    raw_scan=step.scan,
                    measurement_stddev=measurement_stddev,
                    min_range=min_range,
                    max_range=max_range,
                ),
                true_pose=step.true_pose
            )

            conv_playback_steps.append(conv_step)

        return OptimizationPlaybackData(step_data_list=conv_playback_steps)

        
    @staticmethod
    def convert_scans(
        raw_scan: RawScan,
        measurement_stddev: Optional[float] = None,
        min_range: float = 0.1,
        max_range: float = 10.0,
    ):
        measurements = []
        ranges = raw_scan.ranges

        if measurement_stddev is not None:
            ranges = PlaybackConverter.add_measurement_noise(
                ranges=ranges,
                stddev=measurement_stddev,
                min_range=min_range,
                max_range=max_range,
            )

        bearing = raw_scan.angle_min
        for r in ranges:
            measurements.append((r, bearing))
            bearing += raw_scan.angle_increment

        return measurements


    @staticmethod
    def add_measurement_noise(ranges, stddev: float, min_range: float = 0.1, max_range: float = 10.0):
        '''
        Gets the measurement ranges and the desired standard deviation and add random normal noise with that stddev 
        to the measurement ranges. Only add noise to finite values. Clips the ranges to the given min/max values.
        '''
        # Define array and create finite mask
        noisy_ranges = np.array(ranges, dtype=np.float32, copy=True)
        finite_mask = np.isfinite(noisy_ranges)

        # Add noise to finite values
        if np.any(finite_mask):
            # Add noise
            noise = np.random.normal(loc=0.0, scale=stddev, size=int(np.sum(finite_mask)))
            noisy_ranges[finite_mask] = noisy_ranges[finite_mask] + noise

            # Clip distances to min and max distance
            np.clip(noisy_ranges[finite_mask], a_min=min_range, a_max=max_range)

        return noisy_ranges
    
