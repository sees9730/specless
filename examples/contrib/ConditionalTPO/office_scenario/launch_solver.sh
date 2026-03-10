#!/usr/bin/env bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CYCLONEDDS_URI="file://$SCRIPT_DIR/cyclonedds.xml"

source "$HOME/Desktop/specless/.venv/bin/activate"
cd "$SCRIPT_DIR"
python main.py --rviz "$@"
