from typing import List, Tuple
from dataclasses import dataclass
from .playback_loader import PlaybackData, PlaybackStep, RawScan
from ..optimize_rbpf.playback_defs import StepData, PlaybackData as OptimizationPlaybackData



class PlaybackConverter:
    def convert(self, playback_data: PlaybackData) -> OptimizationPlaybackData:
        conv_playback_steps = []
        for step in playback_data.steps:
            conv_step = StepData(
                t = step.t,
                dl = step.dl,
                dr = step.dr,
                scan=self.convert_scans(step.scan),
                true_pose=step.true_pose
            )

            conv_playback_steps.append(conv_step)

        return OptimizationPlaybackData(step_data_list=conv_playback_steps)

        
    @staticmethod
    def convert_scans(raw_scan: RawScan):
        measurements = []
        bearing = raw_scan.angle_min
        for r in raw_scan.ranges:
            measurements.append((r, bearing))
            bearing += raw_scan.angle_increment

        return measurements



