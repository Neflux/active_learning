

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