
from typing import List, Tuple

import numpy as np

from .measurement_model import MeasurementModel
from sklearn.neighbors import NearestNeighbors

from slam.scan_matcher.scan_matcher import ScanMatcher
from slam.scan_matcher.ogm_scan_matching import OGM
from slam.infrastructure.defs import Pose2D


class LikelihoodFiledModel(MeasurementModel):
    def __init__(self, sigma: float=0.1) -> None:
        self.sigma = sigma
         
    
    def likelihood(
        self,
        pose: Pose2D,
        measurements: List[Tuple[float, float]],
        scan_matcher: ScanMatcher,
        neighbor: NearestNeighbors,
    ) -> float:
        '''
        Gmapping version of likelihood computation
        '''
        # Likelihood
        log_likelihood = 0.0

        # Compute no git lieklihood for punishment
        null_likelihood = -0.5
        no_hit = null_likelihood / self.sigma
        # Define max distance 
        max_distance = 0.3

        # Safety checks
        if scan_matcher is None or neighbor is None:
            return 1e-9

        if len(measurements) < 3:
            return 1e-9

        # Transform to points
        scan_points = scan_matcher.transform_measurements_to_points(
            pose=pose,
            measurements=measurements,
        )

        # Check if enough scan points available
        if len(scan_points) < 3:
            return 1e-9

        # Get distances to nearest neighbor for every scan point
        distances, _ = neighbor.kneighbors(scan_points, n_neighbors=1)
        distances = distances[:, 0]

        # Devide distances into valid and invalid
        valid_distances2 = distances[distances <= max_distance]**2
        invalid_distances = distances[distances > max_distance]

        log_likelihood = np.sum(-valid_distances2 / self.sigma) 
        log_likelihood += no_hit * invalid_distances.shape[0]

        return float(log_likelihood)

    
    def likelihood_batch(
        self,
        poses: np.ndarray,
        measurements: List[Tuple[float, float]],
        scan_matcher: ScanMatcher,
        neighbor: NearestNeighbors,
    ) -> np.ndarray:
        '''
        Gmapping variant of lieklihood computation.

        TODO: 
        If used later on we need to make the compuation of the max_distances dependend on grid resolution. 
        max_distance = np.sqrt(2.0) * (kernel_size + 0.5) * resolution

        
        '''
        # Compute no git lieklihood for punishment
        null_likelihood = -0.5
        no_hit = null_likelihood / self.sigma
        # Define max distance 
        max_distance = 0.3

        n_poses = poses.shape[0]

        if scan_matcher is None or neighbor is None:
            return np.full(n_poses, 1e-9)

        if len(measurements) < 3:
            return np.full(n_poses, 1e-9)

        # --------------------------------------------------
        # Precompute local scan points once
        # --------------------------------------------------

        ranges = np.array([m[0] for m in measurements])
        bearings = np.array([m[1] for m in measurements])

        local_x = ranges * np.cos(bearings)
        local_y = ranges * np.sin(bearings)

        n_beams = len(ranges)

        # --------------------------------------------------
        # Transform all poses at once
        # --------------------------------------------------

        px = poses[:, 0][:, None]
        py = poses[:, 1][:, None]
        pt = poses[:, 2][:, None]

        c = np.cos(pt)
        s = np.sin(pt)

        world_x = px + c * local_x - s * local_y
        world_y = py + s * local_x + c * local_y

        # shape -> (Nposes * Nbeams, 2)
        all_points = np.stack(
            [world_x.reshape(-1), world_y.reshape(-1)],
            axis=1,
        )

        # --------------------------------------------------
        # ONE nearest-neighbor call
        # --------------------------------------------------

        distances, _ = neighbor.kneighbors(
            all_points,
            n_neighbors=1,
        )

        distances = distances[:, 0]

        # reshape back
        distances = distances.reshape(n_poses, n_beams)

        # Create mask to access all valid distances
        valid_mask = distances <= max_distance

        # Compute likelihood of valid distances
        valid_dist2 = np.where(valid_mask, distances ** 2, 0.0)
        log_likelihoods = np.sum(-valid_dist2 / self.sigma, axis=1)

        # Add likelihood for invalid distances
        n_invalid = np.sum(~valid_mask, axis=1)
        log_likelihoods += no_hit * n_invalid

        # Convert log-likelihoods to stable positive weights/probs
        log_likelihoods = log_likelihoods - np.max(log_likelihoods)
        probs = np.exp(log_likelihoods)

        return probs


    def likelihood_batch_copy(
        self,
        poses: np.ndarray,
        measurements: List[Tuple[float, float]],
        scan_matcher: ScanMatcher,
        neighbor: NearestNeighbors,
    ) -> np.ndarray:

        n_poses = poses.shape[0]

        if scan_matcher is None or neighbor is None:
            return np.full(n_poses, 1e-9)

        if len(measurements) < 3:
            return np.full(n_poses, 1e-9)

        # --------------------------------------------------
        # Precompute local scan points once
        # --------------------------------------------------

        ranges = np.array([m[0] for m in measurements])
        bearings = np.array([m[1] for m in measurements])

        local_x = ranges * np.cos(bearings)
        local_y = ranges * np.sin(bearings)

        n_beams = len(ranges)

        # --------------------------------------------------
        # Transform all poses at once
        # --------------------------------------------------

        px = poses[:, 0][:, None]
        py = poses[:, 1][:, None]
        pt = poses[:, 2][:, None]

        c = np.cos(pt)
        s = np.sin(pt)

        world_x = px + c * local_x - s * local_y
        world_y = py + s * local_x + c * local_y

        # shape -> (Nposes * Nbeams, 2)
        all_points = np.stack(
            [world_x.reshape(-1), world_y.reshape(-1)],
            axis=1,
        )

        # --------------------------------------------------
        # ONE nearest-neighbor call
        # --------------------------------------------------

        distances, _ = neighbor.kneighbors(
            all_points,
            n_neighbors=1,
        )

        distances = distances[:, 0]

        # reshape back
        distances = distances.reshape(n_poses, n_beams)

        # TODO: Add clipping again later on
        distances = np.clip(distances, 0.0, 1.0)

        mean_error = np.mean(
            (distances / self.sigma) ** 2,
            axis=1,
        )

        k = 5.0
        scaled_mean = -0.5 * k * mean_error
        
        # probs = np.exp(-0.5 * mean_error)
        probs = np.exp(scaled_mean)

        return probs
    

    def gmapping_likelihood(
        self,
        pose: Pose2D,
        measurements: Tuple[float, float],
        ogm: OGM,        
        usable_range: float,
        kernel_size: int=1,
        fullness_threshold: float=1.2,
        free_threshold: float=1.2,
        gaussian_sigma: float=0.05,
        free_cell_ratio: float=np.sqrt(2.0),
    ) -> Tuple[float, float, int]:
        # Init
        score = 0.0
        log_likelihood = 0.0
        matched_count = 0

        null_likelihood = -0.5
        no_hit = null_likelihood / self.sigma

        # Access grid resolution
        grid_resolution = ogm.grid_resolution_m 
        if grid_resolution is None:
             raise ValueError(
                "OGM grid_grid_resolution_m is None. "
                "Cannot compute gmapping likelihood before the OGM map is initialized."
            )
        
        free_delta = grid_resolution * free_cell_ratio

        log_odds_map = ogm.return_log_odds_map()
        
        x, y, theta = pose

        # Compute likelihood for each beam
        for r, b in measurements:
            
            # Check if measurement is valid
            if not np.isfinite(r) or r <= 0.0 or r > usable_range:
                continue
            
            # Compute beam endpoint
            angle = theta + b
            phit_x = x + r * np.cos(angle)
            phit_y = y + r * np.sin(angle)

            # Tranform endpoint to grid coordinates
            iphit = ogm.transform_point_to_grid_cell((phit_x, phit_y))

            # Find cell before endpoint cell in beam direction
            r_free = r - free_delta
            if r_free < 0.0:
                continue

            pfree_x = x + r_free * np.cos(angle)
            pfree_y = y + r_free * np.sin(angle)
            ipfree_abs = ogm.transform_point_to_grid_cell((pfree_x, pfree_y))

            # offset from hit cell to free-before-hit cell
            free_offset = (
                ipfree_abs[0] - iphit[0],
                ipfree_abs[1] - iphit[1],
            )

            # Init grid search vars
            found = False
            best_dist2 = 0.0

            # Find the closest valid cell in the neighborhood of the endpoint cell
            for di in range(-kernel_size, kernel_size + 1):
                for dj in range(-kernel_size, kernel_size + 1):
                    # Compute cell candidate coordinates
                    hit_cell_cand = (iphit[0] + di, iphit[1] + dj)

                    free_cell_cand = (
                        hit_cell_cand[0] + free_offset[0],
                        hit_cell_cand[1] + free_offset[1],
                    )

                    # Check if candidate cells are inside the map
                    if not ogm.cell_inside_map(hit_cell_cand):
                        continue

                    if not ogm.cell_inside_map(free_cell_cand):
                        continue
                    
                    # Get logOdds values
                    hit_cand_log_odds = log_odds_map[hit_cell_cand[0], hit_cell_cand[1]]
                    free_cand_log_odds = log_odds_map[free_cell_cand[0], free_cell_cand[1]]

                    # Check if candidates are valid -> Use for finding closest valid cell
                    if hit_cand_log_odds > fullness_threshold and free_cand_log_odds < free_threshold:
                        # Compute squared error between candidate cell and measurement endpoint
                        cell_x, cell_y = ogm.transform_grid_cell_to_point(hit_cell_cand)

                        dxw = phit_x - cell_x
                        dyw = phit_y - cell_y
                        dist2 = dxw * dxw + dyw * dyw
                        
                        # Find closest valid cell
                        if (not found) or (dist2 < best_dist2):
                            best_dist2 = dist2
                            found = True

            # Update measurement likelihood for each beam 
            if found:
                score += np.exp(-best_dist2 / gaussian_sigma)
                log_likelihood += -best_dist2 / self.sigma
                matched_count += 1
            else:
                log_likelihood += no_hit

        return score, log_likelihood, matched_count

