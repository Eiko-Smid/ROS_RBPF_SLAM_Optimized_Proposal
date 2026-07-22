#!/usr/bin/env python3
"""
Interactive test viewer for a fixed occupancy grid map and a growing
2D robot trajectory.

The currently active pose is displayed together with a heading arrow.

Compatible with Python 3.8.
"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button, Slider

# Import Map handler
from ..infrastructure.map_data_handler import MapDataHandler


def create_test_map() -> np.ndarray:
    """
    Create a deterministic test occupancy grid.

    Cell values:
        0 = occupied
        1 = unknown
        2 = free
    """
    grid = np.ones((18, 24), dtype=np.uint8)

    # Free room area.
    grid[2:16, 2:22] = 2

    # Outer walls.
    grid[2, 2:22] = 0
    grid[15, 2:22] = 0
    grid[2:16, 2] = 0
    grid[2:16, 21] = 0

    # Internal walls and obstacles.
    grid[5:13, 8] = 0
    grid[5, 8:15] = 0
    grid[10:15, 15] = 0
    grid[8:11, 18:20] = 0

    # Door-like gaps.
    grid[9, 8] = 2
    grid[5, 11] = 2
    grid[12, 15] = 2

    return grid


def create_test_trajectory() -> np.ndarray:
    """
    Create ten poses with columns [x, y, theta].

    theta is expressed in radians.
    """
    return np.array(
        [
            [1.5, 1.5, 0.00],
            [2.5, 1.8, 0.15],
            [3.5, 2.2, 0.25],
            [4.5, 3.0, 0.45],
            [5.5, 4.0, 0.65],
            [6.5, 5.0, 0.75],
            [7.5, 5.8, 0.45],
            [8.5, 6.3, 0.20],
            [9.5, 6.7, 0.05],
            [10.5, 7.0, 0.00],
        ],
        dtype=np.float64,
    )


class RoboViewer:
    """Interactive viewer for one fixed map and one trajectory."""
    BLACK = 0
    GRAY = 1
    WHITE = 2

    def __init__(
        self,
        ogm: np.ndarray,
        trajectory: np.ndarray,
        resolution: float = 0.5,
        origin_xy: Tuple[float, float] = (0.0, 0.0),
        occ_thresh: float = 1.4,
        free_thres: float = -1.4,
        heading_vector_length: float = 1.0,
        trajectory_marker_size: float = 6.0,
        trajectory_line_width: float = 4.0,
        current_pose_marker_size: float = 8.0,
        heading_line_width: float = 3.0,
        heading_head_size: float = 20.0,
    ) -> None:
        # Validate inputs
        # map info
        if ogm.ndim != 2:
            raise ValueError("occupancy_grid must be a 2D array.")

        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError(
                "trajectory must have shape (N, 3): [x, y, theta]."
            )

        if trajectory.shape[0] < 1:
            raise ValueError("trajectory must contain at least one pose.")

        if resolution <= 0.0:
            raise ValueError("resolution must be positive.")
        
        # Visual indicators
        if heading_vector_length <= 0.0:
            raise ValueError("heading_vector_length must be positive.")
        
        if trajectory_marker_size <= 0.0:
            raise ValueError("trajectory_marker_size must be positive.")
    
        if trajectory_line_width <= 0.0:
            raise ValueError("trajectory_line_width must be positive.")
        
        if current_pose_marker_size <= 0.0:
            raise ValueError("current_pose_marker_size must be positive.")

        if heading_line_width <= 0.0:
            raise ValueError("heading_line_width must be positive.")
        
        if heading_head_size <= 0.0:
            raise ValueError("heading_head_size must be positive.")            

        # Store inputs
        # map and info
        self.ogm = ogm
        self.trajectory = trajectory
        self.resolution = float(resolution)
        self.origin_x = float(origin_xy[0])
        self.origin_y = float(origin_xy[1])
        self.occ_thres = occ_thresh
        self.free_thres = free_thres

        # Visual indicators
        self.heading_vector_length = float(heading_vector_length)
        # Store base visualization sizes.
        self.base_trajectory_marker_size = float(
            trajectory_marker_size
        )
        self.base_trajectory_line_width = float(
            trajectory_line_width
        )
        self.base_current_pose_marker_size = float(
            current_pose_marker_size
        )
        self.base_heading_line_width = float(
            heading_line_width
        )
        self.base_heading_head_size = float(
            heading_head_size
        )
        self.base_heading_vector_length = float(
            heading_vector_length
        )

        # Init visual scale
        self.visual_scale = 1.0

        self.heading_vector_length = (
            self.base_heading_vector_length
        )
        
        self.n_steps = trajectory.shape[0]
        self.current_step = 1

        # Define figure and axes
        self.figure, self.ax = plt.subplots(figsize=(10, 7))
        self.figure.subplots_adjust(bottom=0.24, right=0.82)

        # Process and create visual elements
        self._create_map()
        self._create_trajectory_artists()
        self._create_slider()
        self._create_visual_scale_slider()
        self._create_buttons()
        self._connect_keyboard_controls()

        self.update_display(step_number=1)


    def discretize_map(self):
        ogm_disc = np.full(
            self.ogm.shape,
            self.GRAY,
            dtype=np.int8
        )

        ogm_disc[self.ogm <= self.free_thres] = self.WHITE
        ogm_disc[self.ogm >= self.occ_thres] = self.BLACK
        return ogm_disc


    def _create_map(self) -> None:
        """Draw the occupancy grid once. It remains constant."""
        # Define extend
        n_rows, n_cols = self.ogm.shape

        extent = (
            self.origin_x,
            self.origin_x + n_cols * self.resolution,
            self.origin_y,
            self.origin_y + n_rows * self.resolution,
        )

        # Discretize the ogm
        ogm_disc = self.discretize_map()

        map_colormap = ListedColormap(["black", "lightgray", "white"])

        self.ax.imshow(
            ogm_disc,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=map_colormap,
            vmin=0,
            vmax=2,
        )

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x position [m]")
        self.ax.set_ylabel("y position [m]")
        self.ax.grid(False)


    def _create_trajectory_artists(self) -> None:
        """Create the trajectory, current-pose marker and heading arrow."""
        self.trajectory_line, = self.ax.plot(
            [],
            [],
            linestyle=":",
            marker="o",
            linewidth=self.base_trajectory_line_width,
            markersize=self.base_trajectory_marker_size,
            label="Trajectory",
            zorder=3,
        )

        self.current_pose_marker, = self.ax.plot(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=self.base_current_pose_marker_size,
            label="Current pose",
            zorder=5,
        )

        self.heading_arrow = FancyArrowPatch(
            posA=(0.0, 0.0),
            posB=(1.0, 0.0),
            arrowstyle="-|>",
            mutation_scale=self.base_heading_head_size,
            linewidth=self.base_heading_line_width,
            label="Current heading",
            zorder=6,
        )

        self.ax.add_patch(self.heading_arrow)
        self.ax.legend(loc="upper left")


    def _create_slider(self) -> None:
        slider_ax = self.figure.add_axes([0.18, 0.10, 0.50, 0.04])

        self.step_slider = Slider(
            ax=slider_ax,
            label="Step",
            valmin=1,
            valmax=self.n_steps,
            valinit=1,
            valstep=1,
            valfmt="%0.0f",
        )
        self.step_slider.on_changed(self._on_slider_changed)


    def _create_visual_scale_slider(self) -> None:
        """
        Create a slider that scales all trajectory visualization elements.

        A value of 1.0 represents the sizes given to the constructor.
        """
        scale_slider_ax = self.figure.add_axes(
            [0.18, 0.04, 0.50, 0.04]
        )

        self.visual_scale_slider = Slider(
            ax=scale_slider_ax,
            label="Trajectory scale",
            valmin=0.25,
            valmax=3.0,
            valinit=1.0,
            valstep=0.05,
            valfmt="%.2f",
        )

        self.visual_scale_slider.on_changed(
            self._on_visual_scale_changed
        )


    def _create_buttons(self) -> None:
        previous_ax = self.figure.add_axes([0.70, 0.08, 0.10, 0.07])
        next_ax = self.figure.add_axes([0.81, 0.08, 0.10, 0.07])

        self.previous_button = Button(previous_ax, "Previous")
        self.next_button = Button(next_ax, "Next")

        self.previous_button.on_clicked(self._show_previous_step)
        self.next_button.on_clicked(self._show_next_step)


    def _connect_keyboard_controls(self) -> None:
        self.figure.canvas.mpl_connect(
            "key_press_event",
            self._on_key_press,
        )


    def _on_slider_changed(self, slider_value: float) -> None:
        self.update_display(step_number=int(slider_value))


    def _on_visual_scale_changed(
        self,
        scale_value: float,
    ) -> None:
        """
        Scale trajectory dots, lines, current-pose marker and heading arrow.
        """
        scale = float(scale_value)
        self.visual_scale = scale

        # Trajectory dots and connecting line.
        self.trajectory_line.set_markersize(
            self.base_trajectory_marker_size * scale
        )
        self.trajectory_line.set_linewidth(
            self.base_trajectory_line_width * scale
        )

        # Current active pose.
        self.current_pose_marker.set_markersize(
            self.base_current_pose_marker_size * scale
        )

        # Heading-vector appearance.
        self.heading_arrow.set_linewidth(
            self.base_heading_line_width * scale
        )
        self.heading_arrow.set_mutation_scale(
            self.base_heading_head_size * scale
        )

        # Heading-vector physical length.
        self.heading_vector_length = (
            self.base_heading_vector_length * scale
        )

        # Recompute the heading-vector endpoint for the active pose.
        self.update_display(self.current_step)


    def _show_previous_step(self, _event) -> None:
        new_step = max(1, self.current_step - 1)
        self.step_slider.set_val(new_step)


    def _show_next_step(self, _event) -> None:
        new_step = min(self.n_steps, self.current_step + 1)
        self.step_slider.set_val(new_step)


    def _on_key_press(self, event) -> None:
        if event.key in ("left", "down"):
            self._show_previous_step(event)
        elif event.key in ("right", "up"):
            self._show_next_step(event)
        elif event.key == "home":
            self.step_slider.set_val(1)
        elif event.key == "end":
            self.step_slider.set_val(self.n_steps)


    def update_display(self, step_number: int) -> None:
        """
        Display all poses from step one through the selected step.

        The heading arrow is shown only for the active/current pose.
        """
        step_number = int(np.clip(step_number, 1, self.n_steps))
        self.current_step = step_number

        visible_poses = self.trajectory[:step_number]
        current_pose = visible_poses[-1]

        self.trajectory_line.set_data(
            visible_poses[:, 0],
            visible_poses[:, 1],
        )

        current_x = float(current_pose[0])
        current_y = float(current_pose[1])
        current_theta = float(current_pose[2])

        self.current_pose_marker.set_data(
            [current_x],
            [current_y],
        )

        # Convert pose heading theta into a 2D vector.
        heading_end_x = (
            current_x
            + self.heading_vector_length * np.cos(current_theta)
        )
        heading_end_y = (
            current_y
            + self.heading_vector_length * np.sin(current_theta)
        )

        self.heading_arrow.set_positions(
            (current_x, current_y),
            (heading_end_x, heading_end_y),
        )

        self.ax.set_title(
            "Constant final occupancy grid with trajectory "
            "through step {}/{} | theta = {:.3f} deg".format(
                step_number,
                self.n_steps,
                np.rad2deg(current_theta),
            )
        )

        self.figure.canvas.draw_idle()


    def show(self) -> None:
        plt.show()



def test_robo_viewer_test_data():
    occupancy_grid = create_test_map()
    trajectory = create_test_trajectory()

    viewer = RoboViewer(
        ogm=occupancy_grid,
        trajectory=trajectory,
        resolution=0.5,
        origin_xy=(0.0, 0.0),
        heading_vector_length=1.0,
    )
    viewer.show()


def test_robo_viewer_actual_data():
    # Def storage path to laod data from
    storage_dir = "/home/smide/work/ros_workspaces/ros_ws/src/rbpf_slam/data/slam/optm_results_mult_part/"
    # sub_dir = "proposal_optm_test_3_map_storage/"
    sub_dir = "proposal_optm_1_13"
    map_sub = "maps/"
    map_suffix = "cafe_1783013816_7f47dcf1cbd1_22" 
    map_filename = "log_odds_map.npy"
    map_meta_filename = "log_odds_map_metadata.json"
    map_dir = os.path.join(storage_dir, sub_dir, map_sub, map_suffix)


    # Init map data handler 
    map_data_handler = MapDataHandler()

    # Load map data 
    map, map_meta = map_data_handler.load(
        input_dir=map_dir,
        map_filename=map_filename,
        metadata_filename=map_meta_filename,
    )

    # Init robo viewer
    origin = map_meta.get("origin")
    origin_x = origin.get("x")
    origin_y = origin.get("y")
    gird_res = map_meta.get("resolution")


    robo_viewer = RoboViewer(
        ogm=map,
        trajectory=create_test_trajectory(),
        resolution=gird_res,
        origin_xy=(origin_x, origin_y),
        heading_vector_length=1.0
    )

    robo_viewer.show()
    


def main():
    # test_robo_viewer_test_data()
    test_robo_viewer_actual_data()


if __name__ == "__main__":
    main()
