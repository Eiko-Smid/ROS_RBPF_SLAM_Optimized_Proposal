# RBPF SLAM with Optimized Proposal Distribution


A ROS1 implementation of **Rao-Blackwellized Particle Filter (RBPF) SLAM with an optimized proposal distribution** for a mobile robot.

The system estimates the robot pose while simultaneously building a **2D occupancy grid map (OGM)** of the environment. The project includes the complete SLAM algorithm, ROS/Gazebo integration, offline parameter-tuning pipelines, evaluation tools, and visualization utilities.

> **Status:** Version 1  
> The current implementation has primarily been developed and evaluated in simulated indoor environments using Gazebo.

---

## Overview

The main objective of this project is to implement an RBPF SLAM system in which each particle represents a possible robot trajectory and maintains its corresponding occupancy grid map.

Instead of sampling the next particle pose directly from the motion model, an **optimized proposal distribution** is estimated using odometry, scan matching, and laser measurements.

At a high level, the particle update follows:

```text
Wheel odometry
      ↓
Motion-model prediction
      ↓
Scan matching
      ↓
Deterministic proposal samples
      ↓
Gaussian proposal approximation
      ↓
Sample new particle pose
      ↓
Particle weighting
      ↓
Resampling if required
      ↓
Occupancy-grid map update
```

This allows the current sensor information to influence the proposal from which new particle poses are sampled.

---

## Main Features

- Rao-Blackwellized Particle Filter SLAM
- Optimized proposal distribution
- 2D occupancy grid mapping
- 2D laser scan matching
- Probabilistic motion and measurement models
- Particle weighting and adaptive resampling
- ROS1 integration
- Gazebo simulation support
- RViz visualization
- Offline dataset recording and playback
- Automated parameter evaluation and scoring
- Multiprocessing-based tuning pipelines
- Deterministic seeds for reproducible experiments
- Dedicated trajectory and map visualization using RoboViewer

---

## SLAM Algorithm

The RBPF factorizes the SLAM posterior such that the robot trajectory is represented by particles while the map can be estimated conditionally for each particle.

For every particle, the next state is estimated using an optimized proposal.

### 1. Motion prediction

Wheel odometry is used by the motion model to predict the next robot pose.

```text
previous particle pose
        +
wheel odometry
        ↓
predicted pose
```

### 2. Scan matching

The predicted pose is corrected using a 2D scan matcher and the particle's current occupancy grid map.

```text
predicted pose
      +
laser scan
      +
particle map
      ↓
scan-matched pose
```

The scan-matched pose provides the center around which the proposal distribution is estimated.

### 3. Optimized proposal estimation

A set of poses is sampled deterministically around the scan-matched pose.

The likelihood of these poses is evaluated and used to approximate the optimized proposal with a Gaussian distribution:

```text
q(x_t) ≈ N(μ, Σ)
```

The covariance can additionally be scaled and limited before sampling.

A new pose for the particle is then drawn from this proposal distribution.

### 4. Particle weighting

The new particle pose is evaluated using the measurement model. The resulting likelihood contributes to the particle weight.

Particle weights are normalized and the effective sample size is monitored.

### 5. Resampling

If the effective number of particles falls below the configured threshold, particles are resampled according to their weights.

### 6. Map update

Finally, the laser measurement is inserted into the occupancy grid associated with the particle using its newly estimated pose.

The process is repeated for every incoming SLAM step.

---

## Main Components

The SLAM implementation is composed of several reusable components.

### Occupancy Grid Mapping

The occupancy grid mapper maintains a 2D log-odds representation of the environment.

Laser beams are used to update free and occupied cells, while the map can dynamically grow when the robot approaches its current boundaries.

The internal log-odds representation can be converted into a ROS `nav_msgs/OccupancyGrid` for visualization and export.

### Scan Matcher

The scan matcher estimates a pose correction by aligning the current laser scan with surfaces extracted from the existing occupancy grid map.

It is used both as an independent component and as part of the optimized RBPF proposal.

### Motion Model

The motion model predicts robot motion from wheel-encoder increments and models uncertainty in the robot motion.

### Measurement Models

Probabilistic measurement models are used to evaluate how well a laser measurement agrees with a particle pose and its corresponding map.

### Proposal Estimator

The proposal estimator combines motion prediction, scan matching, deterministic pose samples, and measurement likelihoods to approximate the proposal distribution used by the RBPF.

### Particle Filter

The particle filter manages:

- particle states,
- particle maps,
- proposal sampling,
- particle weights,
- weight normalization,
- effective sample size,
- resampling,
- and trajectory information.

Several computationally expensive operations are accelerated using **Numba**.

---

## Parameter Tuning

A major part of the project is an offline tuning framework used to evaluate and optimize the different stages of the SLAM algorithm.

Three dedicated tuning pipelines are provided.

### 1. Scan Matcher Tuning

The first pipeline evaluates the scan matcher independently from the particle filter.

Because the simulated ground-truth pose is known, the estimated scan-matching correction can be compared directly against the true robot motion.

This pipeline is used to find stable scan-matcher parameters before integrating the matcher into the RBPF.

### 2. Single-Particle RBPF Tuning

The second pipeline runs the RBPF with a single particle.

Its main purpose is to tune the optimized proposal estimation without introducing the additional effects of a multi-particle filter.

This makes it possible to evaluate whether the proposal distribution improves the pose estimate obtained from motion prediction and scan matching.

### 3. Multi-Particle RBPF Tuning

The final pipeline evaluates the complete RBPF SLAM system using multiple particles.

This stage is used to tune parameters related to:

- proposal sampling,
- proposal covariance,
- particle weighting,
- resampling,
- measurement likelihoods,
- and overall trajectory accuracy.

### Evaluation Workflow

Each tuning pipeline follows approximately the same structure:

```text
Parameter sets
     ×
Datasets
     ×
Seeds
     ↓
Optimizer
     ↓
Runner
     ↓
Step-wise evaluation
     ↓
Run summary
     ↓
Scorer
     ↓
Aggregation across seeds/datasets
     ↓
Parameter ranking
     ↓
Result files
```

Each run produces evaluation metrics and a final score.

Results are first aggregated across repeated runs and seeds and are finally aggregated by parameter set. This produces a single comparable result for each tested parameter configuration across the selected datasets.

Parallel versions of the tuning pipelines use **multiprocessing** to reduce the runtime of larger parameter searches.

---

## Dataset Recording and Playback

Offline tuning is performed using previously recorded simulation data.

A playback/data-recording workflow stores the information required to reproduce SLAM runs, including:

- laser measurements,
- robot pose,
- odometry-related information,
- and timestamps.

The recorded datasets can then be processed repeatedly using different parameters and random seeds without having to rerun the complete Gazebo simulation.

This allows reproducible comparisons between parameter configurations.

---

## ROS Integration

The project contains ROS nodes for both testing individual components and running the complete SLAM system.

### RBPF Data Processor

The data processor prepares the input required by the SLAM system.

It synchronizes the relevant ROS data streams and provides:

- laser scans,
- the Gazebo ground-truth pose used for evaluation,
- and simulated differential-drive wheel increments `dl` and `dr`.

The synchronized information is published as a common RBPF input message.

### OGM Node

The OGM node can run the occupancy grid mapper independently of the SLAM algorithm.

It uses the known robot pose together with the laser measurements to test and visualize occupancy-grid mapping without introducing localization error.

### RBPF SLAM Node

The RBPF ROS node wraps the complete particle-filter SLAM implementation.

It processes the synchronized input data and publishes the estimated robot state, transforms, and generated occupancy grid for visualization in RViz.

---

## TF Structure

During SLAM operation, the relevant TF hierarchy is:

```text
map
└── odom_link
    └── base_link
        └── laser_scanner_link
```

The RBPF provides the localization-related transforms while the robot model provides the static transforms between the robot links and sensors.

---

## RoboViewer

A dedicated visualization tool called **RoboViewer** was implemented to analyze completed SLAM runs.

It can visualize the generated occupancy grid together with different robot trajectories, including:

- ground-truth trajectory,
- raw odometry,
- weighted-mean particle trajectory,
- best-particle trajectory,
- and the trajectory associated with the final map.

This makes it possible to inspect individual runs and identify locations where localization or mapping performance deteriorated.

---

## Technology

The project was developed using:

- Python
- ROS1 Noetic
- Gazebo
- RViz
- NumPy
- Numba
- Matplotlib
- Tkinter

The main development environment is Ubuntu 20.04 with ROS Noetic.

---

## Repository Structure

The most important directories are organized approximately as follows:

```text
rbpf_slam/
├── config/          # ROS and algorithm configuration files
├── data/            # Runtime / dataset-related data
├── launch/          # ROS launch files
├── msg/             # Custom ROS message definitions
├── rviz/            # RViz configurations
├── scripts/         # ROS nodes and executable tools
├── src/             # SLAM implementation
├── CMakeLists.txt
├── package.xml
└── README.md
```

The `src` directory contains the reusable SLAM components, while ROS executable nodes are located primarily in `scripts`.

---

## Building the ROS Package

The package is intended for a ROS1 Noetic catkin workspace.

Example:

```bash
cd ~/work/ros_workspaces/ros_ws
catkin_make
source devel/setup.bash
```

The workspace must be sourced in every terminal before starting the ROS nodes:

```bash
source ~/work/ros_workspaces/ros_ws/devel/setup.bash
```

Additional Python and ROS dependencies must be installed before running the project.

---

## Running the Project

The repository contains launch files for different use cases, including:

- occupancy-grid mapping,
- complete RBPF SLAM,
- dataset recording,
- and playback of previously recorded data.

The corresponding files can be found in:

```text
launch/
```

For example, the standalone occupancy-grid mapper can be started using:

```bash
roslaunch rbpf_slam ogm.launch
```

The exact launch files and configuration files can be adapted depending on the simulation environment and experiment.

---

## Visualization

The online SLAM state can be inspected using **RViz**.

Typical visualizations include:

- occupancy grid map,
- laser scan,
- robot model,
- estimated robot pose,
- TF tree,
- and robot trajectory.

For offline experiment analysis, RoboViewer provides a more detailed comparison between estimated and ground-truth trajectories.

---

## Reproducibility

The tuning framework is designed to make experiments reproducible.

Runs are identified using combinations of:

```text
dataset
parameter set
random seed
```

Relevant parameters and evaluation results are stored together with the corresponding run results.

This allows the same configuration to be evaluated repeatedly and compared across different maps and random seeds.

---

## Current Scope

Version 1 focuses on **2D indoor SLAM for a differential-drive mobile robot in simulation**.

The project includes the complete workflow from individual algorithm components to offline parameter optimization and ROS integration.

The current implementation should be considered an experimental SLAM system rather than a production-ready robotics framework.

Potential future work includes validation on physical hardware, further performance optimization, and additional automated parameter-search methods.