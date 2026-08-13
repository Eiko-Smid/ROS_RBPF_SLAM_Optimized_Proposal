# RBPF SLAM with Optimized Proposal Distribution

![RBPF SLAM mapping](src/rbpf_slam/data/slam/map_gifs/rbpf_slam_aws_indoor_final_small.gif)

This Project is a **ROS** implementation of the **Rao-Blackwellized Particle Filter (RBPF) SLAM with an optimized proposal distribution**
for a simulated Mobile Robot (**Gazebo**). The system estimates the pose of the robot while simultaneously building a **2D occupancy grid map (OGM)**
of the environment. Beside the ROS Nodes the project contains a **tuning pipeline** to efficiently tune the model parameter as well as the 
corresponding infrastructure.

## Features
The system consists of multiple components:
- **RBPF SLAM** algoirthm with optimized proposal consisting of:
    - Probabilistic Motion and Measurement model
    - **ICP** based **Scan Matcher** 
    - **Proposal Estimator** -> Approximates optimal proposal distribution by Gaussian
    - **Occupancy Grid Mapping** algorithm to build the 2D grid map
    - **Resampler** with adaptive resampling strategy
- **ROS Nodes** for sharing and processing information
- Robot Model of Differential Drive Mobile Robot and **Gazebo Simulation environment**
- Three staged **optimization pipeline** to find the best model parameters over different maps
- **Multiprocessing** implementation of pipelines to speedup the tunig process
- **Playback Node** to record **sychronized playback data** in order to feed the tuning pipeline
- **RoboViewer** to visualize the trajectory, particles and the map


## f