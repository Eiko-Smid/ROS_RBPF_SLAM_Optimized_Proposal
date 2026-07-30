from typing import Union
import numpy as np

from .defs import Pose2D


class Transformations2D:
    """
    Stateless helper class for 2D rigid-body transformations.

    A pose (x, y, theta) represents the pose of a child frame expressed
    in a parent/reference frame.

    The corresponding matrix transforms points from the child frame
    into the parent/reference frame.
    """

    @staticmethod
    def pose_to_matrix(pose: Union[Pose2D, np.ndarray]) -> np.ndarray:
        """
        Convert a pose (x, y, theta) into a 3x3 homogeneous
        transformation matrix.
        """
        pose_array = np.asarray(pose, dtype=np.float64)

        if pose_array.shape != (3,):
            raise ValueError(
                f"Pose must have shape (3,), but got {pose_array.shape}."
            )

        if not np.all(np.isfinite(pose_array)):
            raise ValueError(
                f"Pose contains non-finite values: {pose_array}."
            )

        x, y, theta = pose_array

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return np.array(
            [
                [cos_theta, -sin_theta, x],
                [sin_theta,  cos_theta, y],
                [0.0,        0.0,       1.0],
            ],
            dtype=np.float64,
        )
    

    @staticmethod
    def matrix_to_pose(transform: np.ndarray) -> Pose2D:
        """
        Convert a 3x3 homogeneous transformation matrix into
        a pose (x, y, theta).
        """
        transform = np.asarray(transform, dtype=np.float64)

        if transform.shape != (3, 3):
            raise ValueError(
                "Transformation matrix must have shape (3, 3), "
                f"but got {transform.shape}."
            )

        if not np.all(np.isfinite(transform)):
            raise ValueError(
                "Transformation matrix contains non-finite values."
            )

        x = transform[0, 2]
        y = transform[1, 2]

        theta = np.arctan2(
            transform[1, 0],
            transform[0, 0],
        )

        return float(x), float(y), float(theta)


    @classmethod
    def inverse(cls, transform: Union[Pose2D, np.ndarray]) -> Pose2D:
        """
        Invert a 2D transformation.

        If the input represents:

            parent -> child

        the returned pose represents:

            child -> parent
        """
        transform_matrix = cls.pose_to_matrix(transform)
        inverse_matrix = np.linalg.inv(transform_matrix)

        return cls.matrix_to_pose(inverse_matrix)


    @classmethod
    def compose(
        cls,
        first_transform: Union[Pose2D, np.ndarray],
        second_transform: Union[Pose2D, np.ndarray],
    ) -> Pose2D:
        """
        Compose two transformations.

        Matrix equation:

            T_result = T_first @ T_second

        Therefore, second_transform is applied first, followed by
        first_transform.

        Example:

            T_A_C = T_A_B @ T_B_C
        """
        first_matrix = cls.pose_to_matrix(first_transform)
        second_matrix = cls.pose_to_matrix(second_transform)

        result_matrix = first_matrix @ second_matrix

        return cls.matrix_to_pose(result_matrix)


    @classmethod
    def relative_transform(
        cls,
        source_pose: Union[Pose2D, np.ndarray],
        target_pose: Union[Pose2D, np.ndarray],
        ) -> Pose2D:
        """
        Compute the transform that expresses the source frame in
        the target frame.

        Both source_pose and target_pose must be expressed in the
        same reference frame.

        Given:

            T_reference_source
            T_reference_target

        this computes:

            T_target_source
                = inverse(T_reference_target)
                  @ T_reference_source

        The returned pose therefore transforms coordinates from the
        source frame into the target frame.
        """
        reference_to_source = cls.pose_to_matrix(source_pose)
        reference_to_target = cls.pose_to_matrix(target_pose)

        target_to_source = (
            np.linalg.inv(reference_to_target)
            @ reference_to_source
        )

        return cls.matrix_to_pose(target_to_source)
        