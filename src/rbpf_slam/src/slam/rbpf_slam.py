import rospy


class RbpfSlam:
    """Entry point for RBPF SLAM orchestration logic."""

    def __init__(self):
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        rospy.loginfo("RBPF SLAM core started")
