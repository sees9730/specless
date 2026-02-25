#!/usr/bin/env bash
export PATH="$HOME/miniconda3/envs/ros2/bin:$PATH"
export AMENT_PREFIX_PATH="$HOME/miniconda3/envs/ros2"
export PYTHONPATH="$HOME/miniconda3/envs/ros2/lib/python3.11/site-packages:$PYTHONPATH"
export DYLD_FALLBACK_LIBRARY_PATH="$HOME/miniconda3/envs/ros2/lib:$DYLD_FALLBACK_LIBRARY_PATH"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CYCLONEDDS_URI="file://$SCRIPT_DIR/cyclonedds.xml"

rviz2 -d "$SCRIPT_DIR/visualization/mars_rover.rviz"
