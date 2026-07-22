
# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

import os
from typing import Tuple, Iterable

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd

from .robo_viewer import RoboViewer
from ..infrastructure.map_data_handler import MapDataHandler


class RoboViewerLauncher:
    '''
    Small GUI used to select and load map and trajectory data.

    The launcher loads:

        1. A map directory containing:
            - log_odds_map.npy
            - log_odds_map_metadata.json

        2. A CSV file containing the trajectory pose columns.

    After loading the data, it creates and starts a RoboViewer instance.
    '''

    def __init__(
        self,
        trajectory_columns: Tuple[str, str, str],
        map_filename: str = "log_odds_map.npy",
        metadata_filename: str = "log_odds_map_metadata.json",
        start_directory: str = "",
    ) -> None:
        '''
        Initialize the loader GUI.

        Parameters
        ----------
        trajectory_columns:
            Names of the CSV columns containing x, y and theta.

            Example:

                (
                    "map_traj_x",
                    "map_traj_y",
                    "map_traj_theta",
                )

        map_filename:
            Filename of the stored NumPy log-odds map.

        metadata_filename:
            Filename of the stored map metadata.
        '''
        self.trajectory_columns = trajectory_columns
        self.map_filename = map_filename
        self.metadata_filename = metadata_filename
        self.start_directory = start_directory

        self.map_data_handler = MapDataHandler()

        self.root = tk.Tk()
        self.root.title("RoboViewer Loader")
        self.root.geometry("750x190")

        self.map_directory = tk.StringVar()
        self.trajectory_csv_path = tk.StringVar()

        self._create_gui()


    def _create_gui(self) -> None:
        '''
        Create labels, path fields and buttons for the loader.
        '''
        self.root.columnconfigure(1, weight=1)

        # Map directory selection.
        map_label = tk.Label(
            self.root,
            text="Map directory:",
        )
        map_label.grid(
            row=0,
            column=0,
            padx=10,
            pady=15,
            sticky="w",
        )

        map_entry = tk.Entry(
            self.root,
            textvariable=self.map_directory,
        )
        map_entry.grid(
            row=0,
            column=1,
            padx=10,
            pady=15,
            sticky="ew",
        )

        map_button = tk.Button(
            self.root,
            text="Browse",
            command=self._select_map_directory,
        )
        map_button.grid(
            row=0,
            column=2,
            padx=10,
            pady=15,
        )

        # Trajectory CSV selection.
        trajectory_label = tk.Label(
            self.root,
            text="Trajectory CSV:",
        )
        trajectory_label.grid(
            row=1,
            column=0,
            padx=10,
            pady=15,
            sticky="w",
        )

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


    def _select_map_directory(self) -> None:
        '''
        Open a GUI dialog for selecting the stored map directory.
        '''
        selected_directory = filedialog.askdirectory(
            title="Select map directory",
            initialdir=self.start_directory,
        )

        if selected_directory:
            self.map_directory.set(selected_directory)


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
        map_directory = self.map_directory.get()

        if not map_directory:
            raise ValueError("No map directory was selected.")

        return self.map_data_handler.load(
            input_dir=map_directory,
            map_filename=self.map_filename,
            metadata_filename=self.metadata_filename,
        )
    

    def __extract_map_dir_info_for_trajectory_loading(self, map_directory: str):
        dir_name = os.path.basename(
            os.path.normpath(map_directory)
        )

        parts = dir_name.rsplit("_", 3)

        if len(parts) != 4:
            raise ValueError(
                "Invalid map-directory name: '{}'. Expected "
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
            NumPy array with shape (N, 3), containing x, y and theta.
        '''
        # Extract playback id, hash and seed from filename
        map_directory = self.map_directory.get()

        if not map_directory:
            raise ValueError("No map directory selected.")
        
        dataset_id, param_hash, seed = self.__extract_map_dir_info_for_trajectory_loading(
            map_directory=map_directory
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

        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError(
                "Loaded trajectory must have shape (N, 3)."
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


    def _open_robo_viewer(self) -> None:
        '''
        Load the selected files and start RoboViewer.

        Errors are displayed inside a GUI message box.
        '''
        try:
            ogm, map_metadata = self._load_map()
            trajectory = self._load_trajectory(des_cols=self.trajectory_columns)

            origin = map_metadata["origin"]

            origin_x = float(origin["x"])
            origin_y = float(origin["y"])
            resolution = float(map_metadata["resolution"])

        except Exception as error:
            messagebox.showerror(
                title="Could not load RoboViewer data",
                message=str(error),
            )
            return

        # Close the loader window before opening Matplotlib.
        self.root.destroy()

        viewer = RoboViewer(
            ogm=ogm,
            trajectory=trajectory,
            resolution=resolution,
            origin_xy=(origin_x, origin_y),
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
    start_directory = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optm_results_mult_part/"
    
    trajectory_columns = (
        "map_traj_x",
        "map_traj_y",
        "map_traj_theta",
    )


    robot_view_launcher = RoboViewerLauncher(
        trajectory_columns=trajectory_columns,
        start_directory=start_directory,
    )
    robot_view_launcher.run()



if __name__ == "__main__":
    main()