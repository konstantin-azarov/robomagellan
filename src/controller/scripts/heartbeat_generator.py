#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty

if __name__ == '__main__':
    rospy.init_node('heartbeat_generator')
    
    empty_msg = Empty()

    pub = rospy.Publisher('heartbeat', Empty, queue_size = 10)
    rate = rospy.Rate(20)
    
    while not rospy.is_shutdown():
        pub.publish(empty_msg)
        rate.sleep()
        
