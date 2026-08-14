
from typing import Optional

from slam.infrastructure.defs import Pose2D
from slam.rbpf.motion_model import MotionModel


class RawOdomEstimator:
    '''
    RawOdomEstimator estimates the robot's pose based on raw odometry data. Get's a motion model and a start pose
    and then can be used to predict the robot's pose. 
    '''
    DEFAULT_START_POSE: Pose2D = (0.0, 0.0, 0.0)

    def __init__(
        self,
        motion_model: MotionModel,
        start_pose: Optional[Pose2D] = None
    ):
        self.motion_model = motion_model
        self.pose = start_pose if start_pose is not None else self.DEFAULT_START_POSE


    def predict_pose(self, dl: float, dr: float) -> Pose2D:
        '''
        Predicts the robot's pose based on the given odometry (dl, dr).

        Parameters
        ----------
        dl: float
            The distance traveled by the left wheel since the last update.
        dr: float
            The distance traveled by the right wheel since the last update.
        
        Returns
        -------
        Pose2D
            The predicted pose of the robot (x, y, theta).
        '''
        self.pose = self.motion_model.predict_pose(self.pose, dl, dr)
        return self.pose


    def get_pose(self) -> Pose2D:
        '''
        Returns the current pose of the robot.

        Returns
        -------
        Pose2D
            The current pose of the robot (x, y, theta).
        '''
        return self.pose
    

    def reset(self, start_pose: Optional[Pose2D] = None):
        '''
        Resets the robot's pose to the given start pose or to the default start pose if no start pose has been given. 

        Parameters
        ----------
        start_pose: Optional[Pose2D]
            The pose to reset the robot to. If None, the default start pose will be used.
        '''
        if start_pose is not None:
            self.pose = start_pose
        else: 
            self.pose = self.DEFAULT_START_POSE