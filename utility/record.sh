#ros2 bag record /thymioX/virtual_sensor/signal /thymioX/ground_truth/odom /thymioX/odom /thymioX/head_camera/image_raw

ros2 bag record /thymioX/virtual_sensor/signal /thymioX/ground_truth/odom /thymioX/odom /thymioX/head_camera/image_raw -o my_bag --qos-profile-overrides-path durability_override.yaml
