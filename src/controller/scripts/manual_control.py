#!/usr/bin/env python

import math

import rospy
from geometry_msgs.msg import Twist
from trex_dmc01.msg import Status

g_active = False
g_vel = 0
g_turn = 0

def axis(raw, center, rng):
    if math.fabs(raw - center) < 10:
        return 0

    return 2*(raw - center)/float(rng)

def status_callback(msg):
    global g_active, g_vel, g_turn
    g_active = msg.state == 4
    g_vel = axis(msg.channels[1], 3667, 1900)*1.2
    g_turn = -axis(msg.channels[0], 3656, 1900)*30.0*math.pi/180

if __name__ == '__main__':
    rospy.init_node('heartbeat_generator')

    rospy.Subscriber("/trex_dmc01/status", Status, status_callback)
    pub = rospy.Publisher('/control', Twist, queue_size = 10)
    rate = rospy.Rate(20)
    
    while not rospy.is_shutdown():
        cmd = Twist()
        if g_active:
            cmd.linear.z = g_vel
            cmd.angular.y = g_turn
        else:
            cmd.linear.z = 0
            cmd.angular.y = 0
        
        pub.publish(cmd)
        rate.sleep()

