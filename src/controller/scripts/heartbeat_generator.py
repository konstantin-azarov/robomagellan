#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty
from trex_dmc01.msg import Status

g_active = False

def status_callback(msg):
    global g_active
    g_active = (msg.state == 4 and msg.channels[1] > 4000)

if __name__ == '__main__':
    rospy.init_node('heartbeat_generator')

    force = rospy.get_param("~force", False)
    
    empty_msg = Empty()

    rospy.Subscriber("/trex_dmc01/status", Status, status_callback)
    pub = rospy.Publisher('heartbeat', Empty, queue_size = 10)
    rate = rospy.Rate(20)
    
    while not rospy.is_shutdown():
        if force or g_active:
            pub.publish(empty_msg)
        rate.sleep()
        
