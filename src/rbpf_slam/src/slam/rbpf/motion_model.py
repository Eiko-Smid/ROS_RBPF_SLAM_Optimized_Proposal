from math import cos, sin, pi, atan2
import numpy as np

from slam.infrastructure.defs import Pose2D


class MotionModel:
    def __init__(
            self,
            sigma_x: float = 0.1,
            sigma_y: float = 0.1,
            sigma_theta: float = 0.05,
            wheel_separation: float = 0.5
    ):
        # Init parameters
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_theta = sigma_theta
        self.wheel_separation = wheel_separation


    def predict_pose(self, pose: Pose2D, dl: float, dr: float) -> Pose2D:
        '''
        Predicts the pose based in the given control values and the wheel separation of the DDMR.

        Parameters
        ----------
        pose: Pose2D
            The current pose of the robot, given as a tuple (x, y, theta).
        dl: float
            The distance traveled by the left wheel since the last update.
        dr: float
            The distance traveled by the right wheel since the last update. 
        
        Returns
        -------
        Pose2D
            The predicted pose of the robot (x, y, theta) 

        '''
        # Extract pose
        x, y, theta = pose
        
        # predict pose for the case that we turned
        if dr != dl:
            alpha = (dr - dl) / self.wheel_separation
            rad = dl/alpha
            g1 = x + (rad + self.wheel_separation/2.)*(sin(theta+alpha) - sin(theta))
            g2 = y + (rad + self.wheel_separation/2.)*(-cos(theta+alpha) + cos(theta))
            g3 = (theta + alpha + pi) % (2*pi) - pi

        # Predict pose for the case we drove on a straight line
        else:
            g1 = x + dl * cos(theta)
            g2 = y + dl * sin(theta)
            g3 = theta
        
        return (g1, g2, g3)


    def motion_probability(self, x_new: Pose2D, x_prev: Pose2D) -> float:
        '''
        Get's the new psoe and the previous pose and computes the motion probability based on the difference
        between the two poses and the noise parameters of the motion model.

        Parameters
        ----------
        x_new: Pose2D
            The new pose of the robot, given as a tuple (x, y, theta).
        x_prev: Pose2D
            The previous pose of the robot, given as a tuple (x, y, theta).
        
        Returns
        -------
        float: 
            The motion probability. The closer the poses are to each other, the higher the probability. 
        '''
        dx = x_new[0] - x_prev[0]
        dy = x_new[1] - x_prev[1]
        dtheta = x_new[2] - x_prev[2]
        dtheta = atan2(sin(dtheta), cos(dtheta))
        
        return np.exp(
        -0.5 * (
            (dx / self.sigma_x) ** 2 +
            (dy / self.sigma_y) ** 2 +
            (dtheta / self.sigma_theta) ** 2
        )
    )