#!/usr/bin/env python

import socket
import threading

import genpy
import rosgraph
import roslib
import rospy
import rostopic

import yaml

from trex_dmc01.msg import SetMotors

class UnityGlueException(Exception):
    pass

class UnityClient(object):
    def __init__(self, port):
        self.port = port
        self.socket = None
        self.publishers = {}
    
    def reader_thread_main(self):
        f = self.socket.makefile()

        for l in f:
            topic, msg_class_str, msg_str = l.split(" ", 2)
            if topic not in self.publishers:
                self.publishers[topic] = self._create_publisher(topic, msg_class_str)

            msg_class, publisher = self.publishers[topic]
            
            msg = msg_class()
            genpy.message.fill_message_args(msg, [yaml.load(msg_str)], keys={})
            publisher.publish(msg)

        if not rospy.is_shutdown():
            rospy.signal_shutdown("Simulator connection lost")

    def _create_publisher(self, topic_name, msg_class_str):
        topic_name = rosgraph.names.script_resolve_name('unity_glue', topic_name)
        try:
            msg_class = roslib.message.get_message_class(msg_class_str)
        except:
            raise UnityGlueException("invalid topic type: %s" % msg_class_str)
        if msg_class is None:
            raise UnityGlueException("message type not found: %s" % topic_type)
        pub = rospy.Publisher(topic_name, msg_class, latch=False, queue_size=100)
        return msg_class, pub

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('127.0.0.1', self.port))

        self.reader_thread = threading.Thread(target=self.reader_thread_main)
        self.reader_thread.start();

    def send_command(self, target, command, message):
        self.socket.send("%s %s %s\n" % (target, command, message));

    def shutdown(self):
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()


def set_motors_callback(command):
    unity_client.send_command(
            "Robot", "SetSpeedCommand", 
            "%f %f" % (command.left / 127.0, command.right / 127.0))

def sigint_handler():
    unity_client.shutdown();

if __name__ == '__main__':
    rospy.init_node('unity_glue')

    unity_client = UnityClient(6668)
    unity_client.connect()

    rospy.on_shutdown(sigint_handler)
    rospy.Subscriber("/set_motors", SetMotors, set_motors_callback)
    print "Ready"
    rospy.spin()
