#!/usr/bin/env python3
"""A fake Scout: publishes roller_eye/frame on /CoreNode/jpg and logs whatever arrives on /cmd_vel.

This exists because the real robot is not always available - it is battery powered, it sleeps on
its dock, and when this was written there was no Scout on the network at all. Without a fixture,
the only way to find out whether the bridge works would be to buy time on the hardware, and every
mistake would look identical: a black camera card.

It verifies the two halves the bridge is responsible for. The camera half is a genuine end-to-end
test - a real JPEG goes in as message bytes and must come out of /mjpeg renderable. The motion half
prints the Twist it receives, which is how you confirm the axis mapping without watching a robot
drive into a wall: press "forward" and this must report y>0, not x>0.

Usage (see README.md for the full three-container recipe):
    roscore &
    python3 fake_scout.py
"""
import base64
import time

import rospy
from geometry_msgs.msg import Twist

from roller_eye.msg import frame

# A 160x120 JPEG, embedded so the fixture has no image library dependency - the ROS container has
# no Pillow and the bridge itself never decodes anything.
_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEEx"
    "NDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7"
    "Ozs7Ozs7Ozs7Ozs7Ozv/wAARCAB4AKADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAA"
    "AgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6"
    "Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG"
    "x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA"
    "AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5"
    "OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPE"
    "xcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDz2iiiukwCiiigAooooAKKKKACiiigAooo"
    "oAKKKKACiiigAooooAKKKKACiiigDZ0DQE1uO6kkvVtUtgpZmTcMHOSTkYxtq9/wimk/9DTZ/kv/AMXR4U/5AXiH/r1/9lkr"
    "mK8/99UrTjGdkrdF2Oj3Iwi3G9zp/wDhFNJ/6Gmz/Jf/AIuj/hFNJ/6Gmz/Jf/i65iitPYV/+fr+5f5E88P5fxZ0/wDwimk/"
    "9DTZ/kv/AMXR/wAIppP/AENNn+S//F1zFFHsK/8Az9f3L/IOeH8v4s6f/hFNJ/6Gmz/Jf/i6P+EU0n/oabP8l/8Ai65iij2F"
    "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"
)
JPEG = base64.b64decode("".join(_JPEG_B64.split()))

FRAME_TYPE_JPG = 1


def on_cmd_vel(msg: Twist) -> None:
    # On this robot +linear.y is forward and +linear.x is strafe. Printed in that order, with
    # names, so a wrong mapping is obvious here rather than on carpet.
    rospy.loginfo("cmd_vel  forward(y)=%+.3f  strafe(x)=%+.3f  yaw(z)=%+.3f",
                  msg.linear.y, msg.linear.x, msg.angular.z)


def main() -> None:
    rospy.init_node("fake_scout", anonymous=True)
    pub = rospy.Publisher("/CoreNode/jpg", frame, queue_size=1)
    rospy.Subscriber("/cmd_vel", Twist, on_cmd_vel, queue_size=10)
    rospy.loginfo("fake scout up: publishing %d-byte JPEGs on /CoreNode/jpg", len(JPEG))

    seq = 0
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        m = frame()
        m.seq = seq
        m.stamp = int(time.time() * 1000)
        m.type = FRAME_TYPE_JPG
        m.par1, m.par2 = 160, 120          # width, height, per the vendor's field comments
        # Vary the bytes per frame so the bridge's staleness watchdog sees a moving picture; a
        # constant image would be indistinguishable from the frozen-feed failure it watches for.
        # Appended AFTER the end-of-image marker, not written over it: everything from FFD9 on is
        # ignored by decoders, so the frame stays renderable while its digest changes.
        m.data = JPEG + bytes([seq % 251])
        pub.publish(m)
        seq += 1
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
