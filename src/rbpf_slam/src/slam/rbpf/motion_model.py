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
        # Noise parameters for sampling noisy control values
        self.ctrl_motion_fac = 0.05
        self.ctrl_turn_fac = 0.15    

        # Uncertainty parameters motion probability
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_theta = sigma_theta

        # Robot wheel separation
        self.wheel_separation = wheel_separation


    def sample_noisy_ctrl(self, dl, dr):
        '''
        Samples noisy control values based on the given control values and the noise parameters of the motion model.
        The bigger the turn, the higher the noise in the motion.

        Parameters
        ----------
        dl: float
            The distance traveled by the left wheel since the last update.
        dr: float
            The distance traveled by the right wheel since the last update. 
        
        Returns
        -------
        Tuple[float, float]
            The sampled control values (dl, dr) with added noise.
        '''
        # Compute odometry difference
        ctrl_diff = dl - dr

        # Compute control stddv
        ctrl_turn_var = (self.ctrl_turn_fac * ctrl_diff)**2
        dl_ctrl_var = (self.ctrl_motion_fac * dl)**2 + ctrl_turn_var
        dr_ctrl_var = (self.ctrl_motion_fac * dr)**2 + ctrl_turn_var
        dl_ctrl_stddv = np.sqrt(dl_ctrl_var)
        dr_ctrl_stddv = np.sqrt(dr_ctrl_var)

        # Sample control values
        dl_sampled = np.random.normal(dl, dl_ctrl_stddv)
        dr_sampled = np.random.normal(dr, dr_ctrl_stddv)

        return dl_sampled, dr_sampled
        


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
        Get's the new pose and the previous pose and computes the motion probability based on the difference
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