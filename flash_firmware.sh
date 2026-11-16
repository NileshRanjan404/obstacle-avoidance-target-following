#!/bin/bash
# flash_firmware.sh - run on the Pi: compile and upload firmware/agv_base to the GIGA.
set -e
cd "$(dirname "$0")"
FQBN=arduino:mbed_giga:giga
PORT=$(ls /dev/serial/by-id/*Arduino* 2>/dev/null | head -1)
PORT=${PORT:-/dev/ttyACM0}

if pgrep -f "agv_follower/base_driver" >/dev/null; then
  echo "base_driver is running and holds the serial port - stop the launch first."; exit 1
fi
arduino-cli compile --fqbn $FQBN firmware/agv_base
# the GIGA is slow to switch into its bootloader; a later attempt usually works
for i in 1 2 3 4; do
  if arduino-cli upload --fqbn $FQBN -p "$PORT" firmware/agv_base; then
    echo "firmware uploaded"; exit 0
  fi
  echo "upload attempt $i failed, retrying..."; sleep 8   # the bootloader can take several seconds to appear
done
echo "UPLOAD FAILED - double-tap the GIGA reset button and run this again"; exit 1
