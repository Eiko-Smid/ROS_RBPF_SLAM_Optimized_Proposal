
from typing import List, Tuple

import numpy as np

from .measurement_model import MeasurementModel
from sklearn.neighbors import NearestNeighbors

from slam.scan_matcher.scan_matcher import ScanMatcher
from slam.infrastructure.defs import Pose2D



class LikelihoodFiledModel(MeasurementModel):
    def __init__(self, sigma: float=0.1) -> None:
        self.sigma = sigma


    @staticmethod
    def world_to_grid(x, y, origin, resolution):
        gx = int(np.floor((x - origin[0]) / resolution))
        gy = int(np.floor((y - origin[1]) / resolution))
        return gx, gy


    @staticmethod
    def grid_to_world(gx, gy, origin, resolution):
        x = origin[0] + (gx + 0.5) * resolution
        y = origin[1] + (gy + 0.5) * resolution
        return x, y


    @staticmethod
    def inside_grid(cell, grid):
        gx, gy = cell
        return 0 <= gx < grid.shape[1] and 0 <= gy < grid.shape[0]


    @staticmethod
    def logodds_to_prob(l):
        return 1.0 / (1.0 + np.exp(-l))
    

    def gmapping_likelihood_one_pose(
        self,
        pose,
        ranges,
        bearings,
        log_odds_map,
        origin,
        resolution,
        usable_range,
        kernel_size=1,
        fullness_threshold=0.55,
        free_threshold=0.50,
        gaussian_sigma=0.05,
        likelihood_sigma=1.0,
        free_cell_ratio=np.sqrt(2.0),
        likelihood_skip=0,
        laser_pose=(0.0, 0.0, 0.0),
    ):
        x, y, theta = pose
        lx, ly, ltheta = laser_pose

        # Transform laser pose into world
        laser_x = x + np.cos(theta) * lx - np.sin(theta) * ly
        laser_y = y + np.sin(theta) * lx + np.cos(theta) * ly
        laser_theta = theta + ltheta

        score = 0.0
        log_likelihood = 0.0
        matched_count = 0

        null_likelihood = -0.5
        no_hit = null_likelihood / likelihood_sigma

        free_delta = resolution * free_cell_ratio

        skip = 0

        for r, b in zip(ranges, bearings):
            skip += 1
            if skip > likelihood_skip:
                skip = 0

            if not np.isfinite(r) or r <= 0.0 or r > usable_range:
                continue

            if skip:
                continue

            angle = laser_theta + b

            # endpoint
            phit_x = laser_x + r * np.cos(angle)
            phit_y = laser_y + r * np.sin(angle)

            iphit = self.world_to_grid(phit_x, phit_y, origin, resolution)

            # free point before endpoint
            pfree_x = laser_x + (r - free_delta) * np.cos(angle)
            pfree_y = laser_y + (r - free_delta) * np.sin(angle)

            ipfree_abs = self.world_to_grid(pfree_x, pfree_y, origin, resolution)

            # offset from hit cell to free-before-hit cell
            free_offset = (
                ipfree_abs[0] - iphit[0],
                ipfree_abs[1] - iphit[1],
            )

            found = False
            best_dist2 = 0.0

            for dx in range(-kernel_size, kernel_size + 1):
                for dy in range(-kernel_size, kernel_size + 1):
                    hit_cell = (iphit[0] + dx, iphit[1] + dy)

                    free_cell = (
                        hit_cell[0] + free_offset[0],
                        hit_cell[1] + free_offset[1],
                    )

                    if not self.inside_grid(hit_cell, log_odds_map):
                        continue

                    if not self.inside_grid(free_cell, log_odds_map):
                        continue

                    hit_prob = self.logodds_to_prob(log_odds_map[hit_cell[1], hit_cell[0]])
                    free_prob = self.logodds_to_prob(log_odds_map[free_cell[1], free_cell[0]])

                    if hit_prob > fullness_threshold and free_prob < free_threshold:
                        cell_x, cell_y = self.grid_to_world(
                            hit_cell[0],
                            hit_cell[1],
                            origin,
                            resolution,
                        )

                        dxw = phit_x - cell_x
                        dyw = phit_y - cell_y
                        dist2 = dxw * dxw + dyw * dyw

                        if (not found) or (dist2 < best_dist2):
                            best_dist2 = dist2
                            found = True

            if found:
                score += np.exp(-best_dist2 / gaussian_sigma)
                log_likelihood += -best_dist2 / likelihood_sigma
                matched_count += 1
            else:
                log_likelihood += no_hit

        return score, log_likelihood, matched_count


    