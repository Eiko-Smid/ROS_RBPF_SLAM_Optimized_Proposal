from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .playback_loader import PlaybackData, PlaybackStep, RawScan
from ..optimize_rbpf.playback_defs import StepData, PlaybackData as OptimizationPlaybackData



class PlaybackConverter:
    '''
    Class for converting playback data to a format suitable for the optimization process. 
    '''
    def convert(
        self,
        playback_data: PlaybackData,
        measurement_stddev: Optional[float] = None,
        min_range: float = 0.1,
        max_range: float = 10.0,
    ) -> OptimizationPlaybackData:
        '''
        Converts the given playback data. Add noise to measurements and clip distances to the given min and max range.
        Also converts the raw scans to a list of (range, bearing) tuples.

        Parameters
        ----------
        playback_data : PlaybackData
            The playback data to convert.
        measurement_stddev : Optional[float], optional
            The standard deviation of the measurement noise to add. If None, no noise is added.
        min_range : float, optional
            The minimum range to clip the measurements to. Default is 0.1.
        max_range : float, optional
            The maximum range to clip the measurements to. Default is 10.0.
        '''
        conv_playback_steps = []
        for step in playback_data.steps:
            conv_step = StepData(
                t = step.t,
                dl = step.dl,
                dr = step.dr,

                # Convert raw scan to measurements with optional noise addition and range clipping
                scan=self._convert_scans(
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
    def _convert_scans(
        raw_scan: RawScan,
        measurement_stddev: Optional[float] = None,
        min_range: float = 0.1,
        max_range: float = 10.0,
    ) -> List[Tuple[float, float]]:
        '''
        Converts the raw scan to a list of (range, bearing) tuples.
        Adds optional measurement noise and clips the ranges to the given min and max values.

        Parameters
        ----------
        raw_scan : RawScan
            The raw scan to convert.
        measurement_stddev : Optional[float], optional
            The standard deviation of the measurement noise to add. If None, no noise is added.
        min_range : float, optional
            The minimum range to clip the measurements to. Default is 0.1.
        max_range : float, optional
            The maximum range to clip the measurements to. Default is 10.0.

        Returns
        -------
        List[Tuple[float, float]]
            A list of (range, bearing) tuples representing the converted measurements.
        '''
        # Define storage for the converted measurements
        measurements = []
        ranges = raw_scan.ranges

        # Add measurement noise if a standard deviation is provided, and clip the ranges to the specified min and max values
        if measurement_stddev is not None:
            ranges = PlaybackConverter._add_measurement_noise(
                ranges=ranges,
                stddev=measurement_stddev,
                min_range=min_range,
                max_range=max_range,
            )

        # Validate ranges
        if np.isnan(ranges).any():
            print("Playback converter: Measurement model contains nan value")

        # Convert the ranges to (range, bearing) tuples
        bearing = raw_scan.angle_min
        for r in ranges:
            measurements.append((r, bearing))
            bearing += raw_scan.angle_increment

        return measurements


    @staticmethod
    def _add_measurement_noise(
        ranges,
        stddev: float,
        min_range: float = 0.1,
        max_range: float = 10.0,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        '''
        Gets the measurement ranges and the desired standard deviation and add random normal noise with that stddev 
        to the measurement ranges. Only add noise to finite values. Clips the ranges to the given min/max values.

        Parameters
        ----------
        ranges : np.ndarray
            The original measurement ranges.
        stddev : float
            The standard deviation of the noise to add.
        min_range : float, optional
            The minimum range to clip the measurements to. Default is 0.1.
        max_range : float, optional
            The maximum range to clip the measurements to. Default is 10.0.
        seed : Optional[int], optional
            An optional seed for the random number generator. If None, a random seed is used.
        rng : Optional[np.random.Generator], optional
            An optional random number generator. If None, a new generator is created with the given seed.
        
        Returns
        -------
        np.ndarray
            The noisy measurement ranges, clipped to the specified min and max ranges.  
        '''
        # Define array and create finite mask
        noisy_ranges = np.array(ranges, dtype=np.float32, copy=True)
        finite_mask = np.isfinite(noisy_ranges)

        # Prefer provided RNG for deterministic per-run noise generation.
        # If no RNG is provided, optionally create a seeded local generator.
        local_rng = rng if rng is not None else (np.random.default_rng(seed) if seed is not None else None)

        # Add noise to finite values
        if np.any(finite_mask):
            # Add noise
            if local_rng is not None:
                noise = local_rng.normal(loc=0.0, scale=stddev, size=int(np.sum(finite_mask)))
            else:
                noise = np.random.normal(loc=0.0, scale=stddev, size=int(np.sum(finite_mask)))
            noisy_ranges[finite_mask] = noisy_ranges[finite_mask] + noise

            # Clip distances to min and max distance
            # TODO: Test code with clipped ranges 
            # noisy_ranges = np.clip(noisy_ranges[finite_mask], a_min=min_range, a_max=max_range)

        return noisy_ranges
    
