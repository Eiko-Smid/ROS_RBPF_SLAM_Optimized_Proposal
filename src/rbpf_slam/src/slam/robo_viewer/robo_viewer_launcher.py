
# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

import os
from typing import Dict, List, Tuple, Optional

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd

from .robo_viewer import RoboViewer
from ..infrastructure.map_data_handler import MapDataHandler
from ..infrastructure.particle_data_handler import ParticleDataHandler


class RoboViewerLauncher:
    '''
    Small GUI used to select and load map and trajectory data.

    The launcher loads:

        1. A map directory containing:
            - log_odds_map.npy
            - log_odds_map_metadata.json
            - particles.npy

        2. A CSV file containing the trajectory pose columns.

    After loading the data, it creates and starts a RoboViewer instance.
    '''

    def __init__(
        self,
        trajectory_columns: List[Tuple[str, Tuple[str, ...]]],
        map_filename: str = "log_odds_map.npy",
        metadata_filename: str = "log_odds_map_metadata.json",
        particle_filename: str = "particles.npy",
        start_directory: Optional[str] = None,
    ) -> None:
        '''
        Initialize the loader GUI.

        Parameters
        ----------
        trajectory_columns:
            List containing each trajectory name and its CSV columns.
            The first three columns contain x, y and theta. Additional
            trajectory data, such as the best-particle weight, is retained.

            Example:

                [
                    (
                        "map_traj",
                        (
                            "map_traj_x",
                            "map_traj_y",
                            "map_traj_theta",
                        ),
                    ),
                ]

        map_filename:
            Filename of the stored NumPy log-odds map.

        metadata_filename:
            Filename of the stored map metadata.

        particle_filename:
            Filename of the stored particle poses.
        '''
        self.trajectory_columns = trajectory_columns
        self.map_filename = map_filename
        self.metadata_filename = metadata_filename
        self.particle_filename = particle_filename

        # Set to current dir if none, else use given start directory.
        if start_directory is None or os.path.isdir(start_directory) is False:
            self.start_directory = os.getcwd()
        else:   
            self.start_directory = start_directory

        # Define window and window size
        self.root = tk.Tk()
        self.root.title("RoboViewer Loader")
        self.root.geometry("750x190")
        
        self.run_directory = tk.StringVar()
        self.trajectory_csv_path = tk.StringVar()

        self._create_gui()


    def _create_gui(self) -> None:
        '''
        Create labels, path fields and buttons for the loader window.
        '''
        self.root.columnconfigure(1, weight=1)

        # Define run directory selection and text
        run_label = tk.Label(
            self.root,
            text="Run directory:",
        )
        # Define text pos
        run_label.grid(
            row=0,
            column=0,
            padx=10,
            pady=15,
            sticky="w",
        )

        # Define text field for run dir
        run_entry = tk.Entry(
            self.root,
            textvariable=self.run_directory,
        )

        run_entry.grid(
            row=0,
            column=1,
            padx=10,
            pady=15,
            sticky="ew",
        )

        # Define button for selecting run dir (file search dialog)
        run_button = tk.Button(
            self.root,
            text="Browse",
            command=self._select_run_directory,
        )
        run_button.grid(
            row=0,
            column=2,
            padx=10,
            pady=15,
        )

        # Define trajectory selection and text
        trajectory_label = tk.Label(
            self.root,
            text="Trajectory CSV:",
        )
        # Define text pos
        trajectory_label.grid(
            row=1,
            column=0,
            padx=10,
            pady=15,
            sticky="w",
        )

        # Define text field for trajectory CSV
        trajectory_entry = tk.Entry(
            self.root,
            textvariable=self.trajectory_csv_path,
        )
        trajectory_entry.grid(
            row=1,
            column=1,
            padx=10,
            pady=15,
            sticky="ew",
        )

        # Define button for selecting trajectory CSV (file search dialog)
        trajectory_button = tk.Button(
            self.root,
            text="Browse",
            command=self._select_trajectory_csv,
        )
        trajectory_button.grid(
            row=1,
            column=2,
            padx=10,
            pady=15,
        )

        # Start viewer.
        start_button = tk.Button(
            self.root,
            text="Open RoboViewer",
            command=self._open_robo_viewer,
        )
        start_button.grid(
            row=2,
            column=0,
            columnspan=3,
            padx=10,
            pady=15,
        )


    def _select_run_directory(self) -> None:
        '''
        Open a GUI dialog for selecting the stored run directory. Start directory is self.start_directory.
        '''
        selected_directory = filedialog.askdirectory(
            title="Select run directory",
            initialdir=self.start_directory,
        )

        if selected_directory:
            self.run_directory.set(selected_directory)


    def _select_trajectory_csv(self) -> None:
        '''
        Open a GUI dialog for selecting the trajectory CSV.
        '''
        selected_file = filedialog.askopenfilename(
            title="Select trajectory CSV",
            initialdir=self.start_directory,
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )

        if selected_file:
            self.trajectory_csv_path.set(selected_file)


    def _load_map(self):
        '''
        Load the map and its metadata using MapDataHandler.

        Returns
        -------
        ogm:
            Two-dimensional log-odds map.

        metadata:
            Dictionary containing the map metadata.
        '''
        run_directory = self.run_directory.get()

        if not run_directory:
            raise ValueError("No run directory was selected. Please select run directory first!")

        return MapDataHandler.load(
            input_dir=run_directory,
            map_filename=self.map_filename,
            metadata_filename=self.metadata_filename,
        )


    def _load_particles(self) -> Optional[np.ndarray]:
        '''Load all particle poses from the selected run directory.'''
        run_directory = self.run_directory.get()

        if not run_directory:
            raise ValueError(
                "No run directory was selected. "
                "Please select run directory first!"
            )

        try:
            return ParticleDataHandler.load(
                input_dir=run_directory,
                particle_filename=self.particle_filename,
            )
        except FileNotFoundError:
            messagebox.showwarning(
                title="particles not available",
                message=(
                    "There is no particle file available in the chosen run "
                    "directory. Viewer will start without particles!"
                ),
            )
            return None


    def __extract_map_dir_info_for_trajectory_loading(self, run_directory: str):
        dir_name = os.path.basename(
            os.path.normpath(run_directory)
        )

        parts = dir_name.rsplit("_", 3)

        if len(parts) != 4:
            raise ValueError(
                "Invalid run-directory name: '{}'. Expected "
                "<map_name>_<dataset_id>_<parameter_hash>_<seed>."
                .format(dir_name)
            )
        
        _, dataset_id, param_hash, seed = parts

        return (dataset_id, param_hash, seed        )


    def _load_trajectory(self, des_cols) -> np.ndarray:
        '''
        Load one trajectory from the selected CSV file.

        Returns
        -------
        trajectory:
            NumPy array with shape (N, M), where the first three columns
            contain x, y and theta and optional remaining columns contain
            additional trajectory data.
        '''
        # Extract playback id, hash and seed from filename
        run_directory = self.run_directory.get()

        if not run_directory:
            raise ValueError("No run directory selected.")
        
        dataset_id, param_hash, seed = self.__extract_map_dir_info_for_trajectory_loading(
            run_directory=run_directory
        )
        
        filepath = self.trajectory_csv_path.get()

        # Check if path exists
        if not filepath:
            raise ValueError("No trajectory CSV was selected.")
        
        # Read csv
        df = pd.read_csv(
            filepath_or_buffer=filepath
        )

        # Check if cols exist, else raise error
        req_cols = ["dataset_id", "parameter_hash", "seed"]

        for col in req_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Trajectory CSV is missing column: {col}"
                )

        # Ensure required columns are of type str
        df["dataset_id"] = df["dataset_id"].astype(str)
        df["parameter_hash"] = df["parameter_hash"].astype(str)
        df["seed"] = df["seed"].astype(str)

        # Filter rows based on dataset_id, param_hash and seed
        df_mask = (
            (df["dataset_id"] == dataset_id) &  
            (df["parameter_hash"] == param_hash) &
            (df["seed"] == seed)
        )
        filtered_df = df[df_mask].copy()

        if filtered_df.empty:
            raise ValueError(
                "No trajectory found for dataset_id={}, "
                "parameter_hash={}, seed={}."
                .format(dataset_id, param_hash, seed)
            )

        # Check if cols are inside df
        for col in des_cols:
            if col not in filtered_df.columns:
                raise ValueError(
                    f"Trajectory CSV is missing column {col}"
                )


        # Transform the trajectory to numpy arr
        trajectory = filtered_df.loc[
            :,
            list(des_cols),
        ].to_numpy(dtype=np.float64)

        if trajectory.ndim != 2 or trajectory.shape[1] < 3:
            raise ValueError(
                "Loaded trajectory must have at least three columns: "
                "[x, y, theta]."
            )

        if trajectory.shape[0] == 0:
            raise ValueError(
                "The loaded trajectory contains no poses."
            )

        if not np.all(np.isfinite(trajectory)):
            raise ValueError(
                "The loaded trajectory contains NaN or infinite values."
            )
        
        # Transform all angles from deg -> rad
        trajectory[:, 2] = np.deg2rad(trajectory[:, 2])

        # Ensure valid angles in [-pi, pi]
        # trajectory[:, 2] = (trajectory[:, 2] + np.pi) % (2 * np.pi) - np.pi

        return trajectory


    def _load_trajectories(self) -> Dict[str, np.ndarray]:
        '''Load all configured trajectories from the selected CSV file.'''
        trajectories = {}

        for trajectory_name, des_cols in self.trajectory_columns:
            trajectories[trajectory_name] = self._load_trajectory(
                des_cols=des_cols,
            )

        return trajectories


    def _open_robo_viewer(self) -> None:
        '''
        Load the selected files and start RoboViewer.

        Errors are displayed inside a GUI message box.
        '''
        try:
            # Load map, particles and trajectories
            ogm, map_metadata = self._load_map()
            particle_poses = self._load_particles()
            trajectories = self._load_trajectories()

            # Extract map metadata to init the viewer
            origin = map_metadata["origin"]

            origin_x = float(origin["x"])
            origin_y = float(origin["y"])
            resolution = float(map_metadata["resolution"])

            log_odds_limits = map_metadata["log_odds_limits"]
            occ_thres = float(log_odds_limits["max"])
            free_thres = float(log_odds_limits["min"])

        except Exception as error:
            messagebox.showerror(
                title="Could not load RoboViewer data",
                message=str(error),
            )
            return

        # Close the loader window before opening Matplotlib.
        self.root.destroy()

        # Init an run robo viewer
        viewer = RoboViewer(
            ogm=ogm,
            trajectories=trajectories,
            particle_poses=particle_poses,
            resolution=resolution,
            origin_xy=(origin_x, origin_y),
            occ_thres=occ_thres,
            free_thres=free_thres,
            heading_vector_length=1.0,
        )

        viewer.show()


    def run(self) -> None:
        '''Start the loader GUI.'''
        self.root.mainloop()



def main() -> None:
    '''
    Start the RoboViewerLauncher GUI.

    This function is called when running this file as a script.
    '''
    # Define start directory for file seach in the TK gui created by the class RoboViewerLauncher
    start_directory = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optm_results_mult_part/"
    
    # Define dict containing the name of the trajectory to load/use and the corresponding columns names inside the step.csv
    trajectory_columns = [
        (
            "true_pose_traj",
            (
                "true_pose_x",
                "true_pose_y",
                "true_pose_theta",
            ),
        ),
        (
            "raw_odom_traj",
            (
                "raw_odom_pose_x",
                "raw_odom_pose_y",
                "raw_odom_pose_theta",
            ),
        ),
        (
            "weighted_mean_traj",
            (
                "weighted_mean_pose_x",
                "weighted_mean_pose_y",
                "weighted_mean_pose_theta",
            ),
        ),
        (
            "best_particle_traj",
            (
                "best_particle_pose_x",
                "best_particle_pose_y",
                "best_particle_pose_theta",
                "best_particle_weight",
            ),
        ),
        (
            "map_traj",
            (
                "map_traj_x",
                "map_traj_y",
                "map_traj_theta",
            ),
        ),
    ]

    # Init the robo view launcher -> launches the robo viewer
    robot_view_launcher = RoboViewerLauncher(
        trajectory_columns=trajectory_columns,
        start_directory=start_directory,
    )
    robot_view_launcher.run()


if __name__ == "__main__":
    main()
    
