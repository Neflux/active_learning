pkill -f ros2_eloquent_sources
pkill -f gazebo
pkill -f gzserver
pkill -f gzclient
kill -9 $(ps aux | grep gazebo | grep -v grep | awk '{ print $2 }') > /dev/null 2>&1

open -b com.apple.terminal utility/launch_gazebo.sh
#sleep 5
open -b com.apple.terminal utility/spawn_thymio.sh

#ros2 launch gazebo_ros gazebo.launch.py factory:=true verbose:=true paused:=true &
#ros2 launch thymioid_description spawn.launch name:=thymioX &

