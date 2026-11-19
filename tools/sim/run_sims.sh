#!/bin/bash
# run_sims.sh - run the follower against every simulated scenario, on the Pi.
#   bash ~/agv_ws/tools/sim/run_sims.sh              all scenarios
#   bash ~/agv_ws/tools/sim/run_sims.sh box hidden   only these
#
# Everything runs in its own ROS domain (ROS_DOMAIN_ID=77), so the simulated /cmd_vel can never
# reach the real robot's base_driver, even if the robot is running at the same time.
source ~/agv_ws/env.sh
export ROS_DOMAIN_ID=77
DIR=$(cd "$(dirname "$0")" && pwd)
CFG=~/agv_ws/src/agv_follower/config/agv.yaml
SCENARIOS=${*:-clear box offset hidden moving wall near}
declare -A DUR=([hidden]=70 [moving]=40 [wall]=60)

pass=0; fail=0
for sc in $SCENARIOS; do
  ~/agv_ws/install/agv_follower/lib/agv_follower/follower --ros-args --params-file "$CFG" \
      > /tmp/sim_follower_$sc.log 2>&1 &
  FP=$!
  if python3 "$DIR/sim_world.py" "$sc" "${DUR[$sc]:-40}"; then pass=$((pass+1)); else fail=$((fail+1)); fi
  kill -TERM $FP; wait $FP 2>/dev/null
done
echo "== $pass passed, $fail failed =="
echo DONE
