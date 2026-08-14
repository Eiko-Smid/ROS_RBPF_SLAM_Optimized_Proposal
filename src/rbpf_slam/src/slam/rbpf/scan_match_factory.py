from dataclasses import dataclass

from ..scan_matcher.scan_matcher import ScanMatcher
from ..scan_matcher.ogm_scan_matching import OGM
from ..scan_matcher.icp_scan_matching import IterativeClosestPoint


@dataclass(frozen=True)
class OccupancyParams:
    '''
    Parameters for the occupancy grid map (OGM).
    '''
    # Min distance to keep to the border in meters. If distance is below -> extend map
    min_distance_to_border: float = 10.0
    # The prior probability of the ogm
    prior_probability: float = 0.5
    # The increasing probability of the ogm
    increasing_probability: float = 0.85
    # The decreasing probability of the ogm
    decreasing_probability: float = 0.15
    # The minimum log odds of the ogm
    min_log_odds: float = -5.0
    # The maximum log odds of the ogm
    max_log_odds: float = 5.0


@dataclass(frozen=True)
class SensorParams:
    '''
    Parameters for the sensor.'''
    # The minimum sensor range of the robot
    min_sensor_range: float = 0.1
    # The maximum sensor range of the robot
    max_sensor_range: float = 10.0


@dataclass(frozen=True)
class MapParameter:
    '''
    Parameters for the map.
    '''
    # The width of the map in meters
    map_width: float
    # The height of the map in meters
    map_height: float
    # The resolution of the map in meters per cell
    grid_resolution_m: float = 0.5


@dataclass(frozen=True)
class ICPParams:
    '''
    Parameters for ICP algorithm
    '''
    # The first downsample stage divides the distances into discrete cells. Only one point in each cell is kept.
    # grid size in meters.
    downsample_grid_size: float = 0.1
    # The max number of points to use for ICP. If more points left they will be downsampled.
    max_n_points: int = 400

    skip_subsampling: bool = False
    max_correspondence_distance: float = 0.6
    neighbors_pca: int = 10
    max_iterations: int = 5
    epsilon_rel: float = 1e-3
    no_improvement_limit: int = 3
    min_error: float = 5e-4
    min_dtrans: float = 1e-3 
    min_drot: float = 1e-2
    min_points: int = 20
    min_corresp: int = 15
    min_hessian_rank: int = 3
    max_hessian_condition: float = 1e8
    max_translation_jump: float = 0.3
    max_rotation_jump: float = 1.0471975512  # 60 deg in rad
    max_acceptable_mean_error: float = 2.5e-3


@dataclass(frozen=True)
class ScanMatcherParams:
    '''
    Parameters for the scan matcher.
    '''

    occ_thres: float
    delta_r: float
    surface_radius_m: float = 0.1
    min_free_ratio: float = 0.25


@dataclass(frozen=True)
class RobotParams:
    wheel_separation: float = 0.5


class ScanMatchFactory:
    @staticmethod
    def build(
        occ_param: OccupancyParams,
        sens_params: SensorParams,
        map_param: MapParameter,
        icp_params: ICPParams,
        robo_param: RobotParams,
        sm_params: ScanMatcherParams,
    ) -> ScanMatcher:
        # init OGM algorithm
        ogm = OGM(
            map_parameter=occ_param.min_distance_to_border,
            occupancy_parameter= [
                occ_param.prior_probability,
                occ_param.increasing_probability,
                occ_param.decreasing_probability,
                occ_param.min_log_odds,
                occ_param.max_log_odds,
            ],
            sensor_parameter= [
                sens_params.min_sensor_range,
                sens_params.max_sensor_range,
            ]
        )

        # Init empty map with predefined prior probs
        ogm.init_map(
            map_width=map_param.map_width,
            map_height=map_param.map_height,
            grid_resolution=map_param.grid_resolution_m
        )

        # Init ICP algorithm
        icp = IterativeClosestPoint(
            stop_params={
                "max_iterations": icp_params.max_iterations,
                "epsilon_rel": icp_params.epsilon_rel,
                "no_improvement_limit": icp_params.no_improvement_limit,
                "min_error": icp_params.min_error,
                "min_dtrans": icp_params.min_dtrans,
                "min_drot": icp_params.min_drot,
                "min_points": icp_params.min_points,
                "min_corresp": icp_params.min_corresp,
                "min_hessian_rank": icp_params.min_hessian_rank,
                "max_hessian_condition": icp_params.max_hessian_condition,
                "max_translation_jump": icp_params.max_translation_jump,
                "max_rotation_jump": icp_params.max_rotation_jump,
                "max_acceptable_mean_error": icp_params.max_acceptable_mean_error,
                "downsample_grid_size": icp_params.downsample_grid_size,
            },
            max_n_points=icp_params.max_n_points,
            max_correspondence_distance=icp_params.max_correspondence_distance,
            n_neighbors=icp_params.neighbors_pca,
            skip_subsampling=icp_params.skip_subsampling,
        )

        # Init scan matcher
        scan_matcher = ScanMatcher(
            ogm=ogm,
            icp=icp,
            robo_param=((0.0, 0.0, 0.0), robo_param.wheel_separation),  
            sensor_parameters=(
                sens_params.min_sensor_range,
                sens_params.max_sensor_range,
                sm_params.delta_r,
            ),
            occ_thres=sm_params.occ_thres,
            surface_radius_m=sm_params.surface_radius_m,
            min_free_ratio=sm_params.min_free_ratio,
        )
        
        return scan_matcher
