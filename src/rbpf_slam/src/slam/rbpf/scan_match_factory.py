from dataclasses import dataclass, field

from ..scan_matcher.scan_matcher import ScanMatcher
from ..scan_matcher.ogm_scan_matching import OGM
from ..scan_matcher.icp_scan_matching import IterativeClosestPoint


@dataclass(frozen=True)
class OccupancyParams:
    min_distance_to_border: float = 10.0
    prior_probability: float = 0.5
    increasing_probability: float = 0.65
    decreasing_probability: float = 0.35
    min_log_odds: float = -100
    max_log_odds: float = 100


@dataclass(frozen=True)
class SensorParams:
    min_sensor_range: float = 0.1
    max_sensor_range: float = 10.0


@dataclass(frozen=True)
class MapParameter:
    map_width: float
    map_height: float
    grid_resolution_m: float = 0.5


@dataclass(frozen=True)
class ICPParams:
    '''
    Parameters for ICP algorithm
    '''
    max_n_points: int = 400
    max_correspondence_distance: float = 0.6
    neighbors_pca: int = 10
    max_iterations: int = 5
    epsilon_rel: float = 1e-3
    no_improvement_limit: int = 3
    min_error: float = 5e-4
    min_dtrans: float = 1e-3 
    min_drot: float = 1e-2


@dataclass(frozen=True)
class ScanMatcherParams:
    '''
    Parameters for the scan matcher.
    '''
    occ_thres: float
    delta_r: float


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
    ):
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
            },
            max_n_points=icp_params.max_n_points,
            max_correspondence_distance=icp_params.max_correspondence_distance,
            n_neighbors=icp_params.neighbors_pca,
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
        )
        
        return scan_matcher