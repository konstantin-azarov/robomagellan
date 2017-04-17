#!/usr/bin/env python

import rospy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import math
from xml.etree import ElementTree
import utm


def loadPath(filename):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    kml_doc = ElementTree.parse("/home/konstantin/robot_run.kml")

    placemark = kml_doc.find(".//kml:Placemark[kml:name='Robot run']", ns)

    line = placemark.find("./kml:LineString/kml:coordinates", ns)

    pts_latlon = map(
            lambda s: tuple(map(float, s.split(','))[0:2]), 
            line.text.strip().split(r' '))

    pts_utm = [utm.from_latlon(lat, lon) for (lon, lat) in pts_latlon]

    x0 = pts_utm[0][0]
    y0 = pts_utm[0][1]

    pts = [(x - x0, y - y0) for (x, y, _, _) in pts_utm]

    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]

    d = math.sqrt(dx*dx + dy*dy)

    cosf = dy / d
    sinf = dx / d

    path = [(0, 0)]

    for i in xrange(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]

        dx1 = cosf*dx - sinf*dy
        dy1 = sinf*dx + cosf*dy

        p = path[-1]
        path.append((p[0] + dx1, p[1] + dy1))

    msg = Path()
    msg.header.seq = 0
    msg.header.stamp = rospy.get_rostime()
    msg.header.frame_id = "/map"
    msg.poses = []

    rospy.loginfo("Path: ");
    for (x, y) in path:
        p = PoseStamped()
        p.header.seq = 0
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = "/map"
        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.position.z = 0
        p.pose.orientation.x = 0
        p.pose.orientation.y = 0
        p.pose.orientation.z = 0
        p.pose.orientation.w = 1
        msg.poses.append(p)

        rospy.loginfo("  %f, %f" % (x, y))

    return msg

if __name__ == '__main__':
    rospy.init_node('path_provider')

    path_msg = loadPath(rospy.get_param("~path_file"))

    pub = rospy.Publisher("/path", Path, queue_size = 1, latch=1)

    pub.publish(path_msg)

    rospy.spin()
