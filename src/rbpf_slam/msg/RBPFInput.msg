# timestamp of the selected laser scan
std_msgs/Header header

# Sensor messge for 2d laser scan
sensor_msgs/LaserScan laser_scan

# Distance traveled of left and right wheel from last timestamp
#float64 dl
#float64 dr

rbpf_slam/WheelEncoder wheel_encoder

# True pose of the robot for pose error computation and ground truth visualization
geometry_msgs/Pose2D true_pose