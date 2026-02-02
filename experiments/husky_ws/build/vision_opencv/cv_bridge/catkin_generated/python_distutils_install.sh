#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/src/vision_opencv/cv_bridge"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/install/lib/python3/dist-packages:/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/build" \
    "/home/robotixx/anaconda3/envs/nvcahsor_nightly/bin/python" \
    "/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/src/vision_opencv/cv_bridge/setup.py" \
    egg_info --egg-base /mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/build/vision_opencv/cv_bridge \
    build --build-base "/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/build/vision_opencv/cv_bridge" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/install" --install-scripts="/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/install/bin"
