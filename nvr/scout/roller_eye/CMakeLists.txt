cmake_minimum_required(VERSION 3.0.2)
project(roller_eye)

# Generates the Python class for roller_eye/frame and nothing else. There are no C++ sources here
# and nothing is built for the robot - see package.xml for why this package exists at all.
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)

add_message_files(FILES frame.msg status.msg)
generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs)
