alias build_roscore="docker build -t myros:roscore ."

alias create_network="docker network create ros_network"

alias roscore="docker run -it --rm \
    --net ros_network \
    --name master \
    myros:roscore \
    roscore"

rosrun () {
    if [ "$#" -ne 2 ]; then
        echo "Usage: rosrun <package> <node>"
        return
    fi

    ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
    xhost + $ip > /dev/null 2>&1

    docker run -it --rm \
        --name "$2" \
        --net ros_network \
        -e DISPLAY=$ip:0 -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --env ROS_HOSTNAME="$2" --env ROS_MASTER_URI=http://master:11311 \
        myros:roscore \
        rosrun $1 $2
}
export -f rosrun > /dev/null 2>&1

graphical_app() {
    xhost + > /dev/null 2>&1

    ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
    xhost + $ip > /dev/null 2>&1

    sudo docker run -it --rm \
        --name "$1" \
        -e DISPLAY=$ip:0 -v /tmp/.X11-unix:/tmp/.X11-unix:rw --env="QT_X11_NO_MITSHM=1" \
        --env ROS_HOSTNAME="$1" --env ROS_MASTER_URI=http://master:11311 --net ros_network \
        myros:roscore \
        $1
}

alias rqt="graphical_app rqt"
alias rviz="graphical_app rviz"
#alias gazebo="graphical_app gazebo"

master () {
    docker exec -it master bash -c "source /ros_entrypoint.sh;bash"
}
export -f master > /dev/null 2>&1

dkillall () {
    if [ "$#" -eq 0 ]; then
        docker rm -f $(docker ps -a -q)
    else
        for var in "$@"
        do
            docker stop $var && docker rm $var
        done
    fi
}
export -f dkillall > /dev/null 2>&1
