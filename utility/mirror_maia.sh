pushd /Users/ste/code/active_learning/src/elohim/elohim/
#cp -fR src/elohim/elohim/* /Volumes/home/code
#cp -fR src/elohim/elohim/training.py /Volumes/home/code/training.py
rsync -uav --progress --exclude bnn --exclude __pycache__ ./* /Volumes/home/code

popd