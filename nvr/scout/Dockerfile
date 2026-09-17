# Moorebot Scout -> MJPEG bridge for Frigate.
#
# Base is ros:noetic-ros-base, which is EOL (May 2025) and Ubuntu 20.04. Two things keep that from
# being a live risk. The image's ROS apt source is http://snapshots.ros.org/noetic/final, a frozen
# post-EOL snapshot rather than a rolling repo, so it is not going to shift under a rebuild; and
# ros-base already ships rospy, rostopic, catkin, message_generation and geometry_msgs, so the only
# thing this build installs is aiohttp. Verified working on this board in 2026.
#
# ROS Noetic is the only option, not a preference: the Scout's own firmware is ROS1 and speaks
# TCPROS. A ROS2 container could not talk to it without a bridge on the robot, which is exactly
# the thing the vendor's closed communication layer prevents.
FROM ros:noetic-ros-base

# aiohttp is the whole dependency list. Frames arrive already JPEG-encoded from the robot, so
# nothing here decodes an image and there is no numpy/OpenCV/Pillow in this container.
#
# Taken from Ubuntu (3.6.2), not pip: the base image has no pip at all, and installing one to
# fetch a newer aiohttp would add more to the image than it is worth. 3.6.2 has every API this
# bridge uses - StreamResponse, json_response, AppRunner, TCPSite - checked, not assumed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-aiohttp \
 && rm -rf /var/lib/apt/lists/*

# Build the message definitions. Without the generated roller_eye.msg Python module, subscribing
# to the camera topic fails at runtime with "Cannot load message class" - ROS1 resolves a topic's
# type by name from the master and then imports it locally.
WORKDIR /catkin_ws
COPY roller_eye/ src/roller_eye/
RUN . /opt/ros/noetic/setup.sh && catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5

COPY scout_mjpeg_bridge.py /opt/scout/scout_mjpeg_bridge.py

# Sourcing the devel overlay is what puts the generated messages on PYTHONPATH; running the script
# without it is the single most likely way to get a container that starts and then serves nothing.
ENTRYPOINT ["/bin/bash", "-lc", \
  ". /opt/ros/noetic/setup.bash && . /catkin_ws/devel/setup.bash && exec python3 -u /opt/scout/scout_mjpeg_bridge.py \"$@\"", "--"]
