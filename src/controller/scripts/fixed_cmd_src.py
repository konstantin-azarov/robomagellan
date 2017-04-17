#!/usr/bin/env python

import rospy
import math

from geometry_msgs.msg import Twist
from trex_dmc01.msg import Status

g_publish = False

def status_callback(msg):
    global g_publish
    g_publish = (msg.state == 4 and msg.channels[0] > 50) or cmd_force

if __name__ == '__main__':
    rospy.init_node('fixed_cmd_src')

    cmd_vel = rospy.get_param("~vel")
    cmd_turn = rospy.get_param("~turn")
    cmd_force = rospy.get_param("~force", False)

    g_publish = cmd_force
    
    rospy.Subscriber("trex_dmc01/status", Status, status_callback)
    g_pub = rospy.Publisher("/control", Twist, queue_size = 10)

    rospy.loginfo("Ready")
    print g_publish

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        cmd = Twist()
        if g_publish:
            cmd.linear.z = cmd_vel
            cmd.angular.y = cmd_turn
        else:
            cmd.linear.z = 0
            cmd.angular.y = 0
        g_pub.publish(cmd)
        rate.sleep()

