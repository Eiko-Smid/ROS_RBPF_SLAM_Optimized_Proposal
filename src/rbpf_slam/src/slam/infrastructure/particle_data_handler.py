from pathlib import Path
from typing import Sequence, Union

import numpy as np


class ParticleDataHandler:
    """
    Stores and loads the particle poses collected during one tuning run.

    The particles are stored in one three-dimensional NumPy array with the
    shape ``(steps, particles, 3)``. The final axis contains ``x``, ``y`` and
    ``theta`` for one particle pose. The particle poses from before resampling
    are used so that the stored data still contains the weighted particle
    distribution produced at each step.
    """

    @staticmethod
    def _prepare_particle_poses(
        particle_poses: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> np.ndarray:
        """
        Convert the supplied particle poses into a non-empty numeric 3D array.
        """
        try:
            particle_array = np.asarray(particle_poses)
        except ValueError as exc:
            raise ValueError(
                "particle_poses must contain the same number of particles "
                "at every step."
            ) from exc

        if particle_array.size == 0:
            raise ValueError("particle_poses must not be empty.")

        if particle_array.ndim != 3 or particle_array.shape[2] != 3:
            raise ValueError(
                "particle_poses must have shape (steps, particles, 3), got "
                "shape {}.".format(particle_array.shape)
            )

        if not np.issubdtype(particle_array.dtype, np.number):
            raise ValueError("particle_poses must contain numeric values.")

        if not np.all(np.isfinite(particle_array)):
            raise ValueError("particle_poses must contain only finite values.")

        return particle_array

    @classmethod
    def save(
        cls,
        output_dir: str,
        particle_poses: Union[np.ndarray, Sequence[np.ndarray]],
        particle_filename: str = "particles.npy",
    ) -> None:
        """
        Save all particle poses from one run.

        Parameters
        ----------
        output_dir:
            Directory in which the particle file is stored.

        particle_poses:
            Particle pose arrays for all steps, with shape
            ``(steps, particles, 3)``.

        particle_filename:
            Filename to use for the NumPy particle file.
        """
        particle_array = cls._prepare_particle_poses(particle_poses)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        particle_path = output_path / particle_filename

        np.save(
            str(particle_path),
            particle_array,
            allow_pickle=False,
        )

    @classmethod
    def load(
        cls,
        input_dir: str,
        particle_filename: str = "particles.npy",
    ) -> np.ndarray:
        """
        Load all particle poses from one run.

        Parameters
        ----------
        input_dir:
            Directory containing the particle file created by ``save()``.

        particle_filename:
            Filename of the NumPy particle file to load.

        Returns
        -------
        particle_poses:
            Loaded particle poses with shape ``(steps, particles, 3)``.
        """
        input_path = Path(input_dir)
        particle_path = input_path / particle_filename

        if not particle_path.is_file():
            raise FileNotFoundError(
                "Particle file not found: {}".format(particle_path)
            )

        particle_poses = np.load(
            str(particle_path),
            allow_pickle=False,
        )

        return cls._prepare_particle_poses(particle_poses)
