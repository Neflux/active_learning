

pushd src
ros2 pkg create --build-type ament_python elohim --dependencies rclpy example_interfaces \
    --maintainer-name "Stefano Bonato" --maintainer-email "bonats@usi.ch" --license "Apache License 2.0" \
    --description "World editor"

popd


#whole, clean
alias full_rebuild=rm -rf install/ build/ && colcon build --merge-install && source install/setup.bash

#specific, clean
alias elohim_rebuild=rm -rf install/elohim/ build/elohim/ && colcon build --packages-select elohim --merge-install && source install/setup.bash






colcon build --packages-select elohim && ros2 run elohim client --texture snow

. install/setup.bash
rm -rf install/ build/ && colcon build --merge-install && source install/setup.bash && ros2 run elohim client --texture desert
colcon build --merge-install --packages-select elohim && source install/setup.bash && ros2 run elohim client --texture snow
rm -rf install/share/elohim/ build/elohim/ && colcon build --merge-install --packages-select elohim && source install/setup.bash && ros2 run elohim client --texture widmanstatten

ros2 topic pub --once /thymioX/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.05, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"


ros2 pkg create --build-type ament_cmake test_ground_plane --maintainer-name "Stefano Bonato" --maintainer-email "bonats@usi.ch" --license "Apache License 2.0" --description "Test World"

ros2 launch gazebo_ros gazebo.launch.py factory:=true world:=test_world.world
ros2 run elohim

ros2 launch thymioid_description spawn.launch name:=thymioX

ros2 interface show gazebo_msgs/srv/SpawnEntity.srv

string name                       # Name of the entity to be spawned (optional).
string xml                        # Entity XML description as a string, either URDF or SDF.
string robot_namespace            # Spawn robot and all ROS interfaces under this namespace
geometry_msgs/Pose initial_pose   # Initial entity pose.
string reference_frame            # initial_pose is defined relative to the frame of this entity.
                                  # If left empty or "world" or "map", then gazebo world frame is
                                  # used.
                                  # If non-existent entity is specified, an error is returned
                                  # and the entity is not spawned.
---
bool success                      # Return true if spawned successfully.
string status_message             # Comments if available.

#reset sim
ros2 service call /pause_physics std_srvs/srv/Empty
ros2 service call /reset_simulation std_srvs/srv/Empty


ros2 service call /reset_world std_srvs/srv/Empty

ros2 service call /spawn_entity gazebo_msgs/srv/SpawnEntity "{name: 'ciao', xml: '<?xml version=\"1.0\" ?>
<sdf version=\"1.6\">
  <model name=\"suv\">
    <static>true</static>
    <link name=\"link\">
      <collision name=\"collision\">
        <pose>0 0 0 0 0 -1.57079632679</pose>
        <geometry>
          <mesh>
            <scale>0.06 0.06 0.06</scale>
            <uri>model://suv/meshes/suv.obj</uri>
          </mesh>
        </geometry>
      </collision>
      <visual name=\"visual\">
        <pose>0 0 0 0 0 -1.57079632679</pose>
        <geometry>
          <mesh>
            <scale>0.06 0.06 0.06</scale>
            <uri>model://suv/meshes/suv.obj</uri>
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
', robot_namespace: 'thymioX'}"

craft_commit_link () {
    echo "https://raw.githubusercontent.com/Homebrew/homebrew-core/$1/Formula/$2.rb"
}
export -f craft_commit_link > /dev/null 2>&1

commits () {
    git -C "$(brew --repo homebrew/core)" log master -- Formula/$1.rb
}
export -f commits > /dev/null 2>&1

brew_deps_commit () {
    brew deps $(craft_commit_link $1 $2)
}
export -f brew_deps_commit > /dev/null 2>&1

brew_install_commit () {
    brew install $(craft_commit_link $1 $2)
}
export -f brew_install_commit > /dev/null 2>&1

nopy38 () { 
    commits=$(git --no-pager -C "$(brew --repo homebrew/core)" log master --pretty=format:"%H" --max-count=${2-5} -- Formula/$1.rb)
    while IFS= read -r sha; do
        echo "Trying $sha"
        deps=$(brew_deps_commit $sha $1 /dev/null 2>&1)
        if echo "$deps" | grep -q python@3.8
        then
            :
        else
            echo "Latest formula without python@3.8 found"
            break
        fi

    done <<< "$commits"
}
export -f nopy38 > /dev/null 2>&1



alias code="code --reuse-window"

#cd ~/ros2_eloquent_sources
#. ~/ros2_eloquent_sources/install/setup.bash > /dev/null 2>&1

#compile
export PATH="/usr/local/opt/opencv@3/bin:/usr/local/opt/python@3.8/bin:/usr/local/opt/qt/bin:/usr/local/opt/icu4c/bin:/usr/local/opt/icu4c/sbin:$PATH"
export LDFLAGS="-L/usr/local/opt/python@3.8/lib -L/usr/local/opt/qt/lib -L/usr/local/opt/icu4c/lib"
export PKG_CONFIG_PATH="/usr/local/opt/python@3.8/lib/pkgconfig:/usr/local/opt/qt/lib/pkgconfig:/usr/local/opt/icu4c/lib/pkgconfig"
export CPPFLAGS="-I/usr/local/opt/qt/include -I/usr/local/opt/icu4c/include"

export OpenCV_DIR="/usr/local/opt/opencv@3/share/OpenCV"

alias python3=python3.8

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/usr/local/opt/qt
export CPATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include

export ROS_DOMAIN_ID=137
export GAZEBO_MASTER_URI=http://localhost:11346

#source ~/ros2_eloquent_sources/install/setup.bash > /dev/null 2>&1
#echo Ros2 sourced

source ~/ws/install/setup.bash > /dev/null 2>&1
#echo Gazebo sourced

cd ~/code/active_learning
. install/setup.bash > /dev/null 2>&1
#echo Active Learning project sourced

#export GAZEBO_RESOURCE_PATH=/Users/ste/ws/src/ros-simulation/gazebo_ros_pkgs/gazebo_plugins/worlds/
#export GAZEBO_MODEL_PATH=/Users/ste/code/active_learning/models
#export GAZEBO_MODEL_PATH=/Users/ste/code/active_learning/install/share/elohim/models

#whole, clean
alias full_rebuild="rm -rf install/ build/ && colcon build --merge-install && source install/setup.bash > /dev/null 2>&1"

#specific, clean, only valid for --merge-install, doesnt delete points.json

elohim_rebuild () {
    cp install/share/elohim/points.json points.json
    rm -rf install/share/elohim/ build/share/elohim/
    colcon build --packages-select elohim --merge-install
    source install/setup.bash > /dev/null 2>&1
    mv points.json install/share/elohim/points.json
}
export -f elohim_rebuild > /dev/null 2>&1

alias pycharm="/Applications/PyCharm.app/Contents/MacOS/pycharm"

rebuild () {
    rm -rf install/share/$1 install/lib/$1 build/share/$1
    colcon build --packages-select $1 --merge-install
    source install/setup.bash  > /dev/null 2>&1
}
export -f rebuild > /dev/null 2>&1

export GAZEBO_RESOURCE_PATH="/usr/local/Cellar/gazebo9/9.13.0/share/gazebo-9/"
export IGN_IP=127.0.0.1

#echo $(dirname $(which gazebo))/$(dirname $(readlink $(which gazebo)))
#$(cd $(which gazebo)/../ && pwd))

