#!/bin/bash
# deploy.sh - run on the Mac: copy agv_ws to the Pi and build it there.
#   ./deploy.sh            copy + build
#   ./deploy.sh flash      copy + build + compile and upload the GIGA firmware
set -e
PI=${PI:-controlheist@172.20.10.2}
cd "$(dirname "$0")"

rsync -a --delete --exclude build/ --exclude install/ --exclude log/ ./ "$PI:agv_ws/"
ssh "$PI" 'bash -lc "source /opt/ros/jazzy/setup.bash && source ~/robot_ws/install/setup.bash \
  && cd ~/agv_ws && colcon build --symlink-install 2>&1 | tail -5"'

if [ "$1" = "flash" ]; then
  ssh "$PI" 'bash ~/agv_ws/flash_firmware.sh'
fi
