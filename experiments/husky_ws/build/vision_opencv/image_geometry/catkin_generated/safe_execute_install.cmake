execute_process(COMMAND "/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/build/vision_opencv/image_geometry/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/mnt/sbackup/Server_3/harshr/sbt_depth_experiments/husky_ws/build/vision_opencv/image_geometry/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
