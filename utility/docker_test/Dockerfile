FROM osrf/ros:melodic-desktop-full
#FROM ros:melodic-ros-base
# install ros tutorials packages
RUN apt-get update && apt-get install -y \
    ros-melodic-ros-tutorials \
    ros-melodic-common-tutorials \
    ros-melodic-turtlesim \
    libgl1-mesa-glx libgl1-mesa-dri mesa-utils \
    ros-melodic-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/

#RUN echo "source /ros_entrypoint.sh" >> /root/.bashrc 