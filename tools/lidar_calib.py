#!/usr/bin/env python3
"""lidar_calib.py - find which way the lidar's 0 deg points and which way its angles turn.

Needs the lidar running, e.g.
    ros2 launch agv_follower bringup.launch.py base:=false camera:=false follow:=false
then, in a second terminal:
    source ~/agv_ws/env.sh && python3 ~/agv_ws/tools/lidar_calib.py

Steps
  1. clear the area around the robot (about 1 m), press Enter        -> baseline scan
  2. put a box about 40 cm straight AHEAD, press Enter  -> angle of 'front'
  3. move it about 40 cm to the robot's LEFT, press Enter             -> angle direction
It also lists returns that never change and sit close to the robot (cables, camera mast ...),
because the planner would treat those as obstacles.
"""
import math

import numpy as np
np.seterr(invalid="ignore")
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def grab(node, n=8):
    """Median of n scans -> (angles_deg, ranges) with invalid = inf."""
    got = []
    sub = node.create_subscription(LaserScan, '/scan', got.append, qos_profile_sensor_data)
    while len(got) < n:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_subscription(sub)
    # Beam count changes from scan to scan, so put every scan into fixed 0.5 deg bins
    # (nearest return per bin), then take the median over the scans.
    nb = 720
    R = np.full((len(got), nb), np.inf)
    for j, s in enumerate(got):
        r = np.asarray(s.ranges, dtype=float)
        ang = np.degrees(s.angle_min + np.arange(len(r)) * s.angle_increment)
        ok = np.isfinite(r) & (r >= 0.03)
        b = (((ang[ok] + 180.0) % 360.0) / 0.5).astype(int) % nb
        np.minimum.at(R[j], b, r[ok])
    r = np.median(R, axis=0)
    a = -180.0 + 0.5 * np.arange(nb) + 0.25
    return a, r


def new_object(a, base, now, max_range=0.6):
    """Angle and range of the CLOSEST group of beams (>= 3) that got >= 10 cm closer.
    Only looks within max_range, so a person standing further away is ignored."""
    closer = np.isfinite(now) & (now < max_range) & ((~np.isfinite(base)) | (base - now > 0.10))
    idx = np.where(closer)[0]
    if len(idx) < 3:
        return None
    # group neighbouring beams (wrapping around 360)
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= 3:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    if len(groups) > 1 and groups[0][0] < 5 and groups[-1][-1] > len(a) - 6:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    groups = [g for g in groups if len(g) >= 3]
    if not groups:
        return None
    g = min(groups, key=lambda g: float(np.median(now[g])))
    ang = np.radians(a[g])
    centre = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean()))
    return centre, float(np.median(now[g])), len(g)


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def main():
    rclpy.init()
    node = rclpy.create_node('lidar_calib')
    print(__doc__.split('Steps')[1])

    input('1. Clear the area around the robot, then press Enter ')
    a, base = grab(node)
    near = np.isfinite(base) & (base < 0.35)
    if near.any():
        print(f'   {near.sum()} beams see something closer than 35 cm even with the area clear:')
        idx = np.where(near)[0]
        runs, cur = [], [idx[0]]       # contiguous runs of beams
        for i in idx[1:]:
            if i - cur[-1] <= 3:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        for run in runs:
            print(f'     {a[run[0]]:+7.1f} .. {a[run[-1]]:+7.1f} deg   at {np.min(base[run]):.2f} m')
        print('   -> if these are parts of the robot, tell Claude so they can be masked out.')

    input('\n2. Put the box about 40 cm straight AHEAD of the robot, then press Enter ')
    a2, ahead = grab(node)
    f = new_object(a, base, ahead)
    if f is None:
        print('   No new object found - is it within 60 cm and at the lidar\'s height?')
        return
    print(f'   object ahead seen at {f[0]:+.1f} deg, {f[1]:.2f} m ({f[2]} beams)')

    input('\n3. Move the box about 40 cm to the robot\'s LEFT, then press Enter ')
    a3, left = grab(node)
    l = new_object(a, base, left)
    if l is None:
        print('   No object found on the left - try again.')
        return
    print(f'   object left seen at {l[0]:+.1f} deg, {l[1]:.2f} m ({l[2]} beams)')

    turn = wrap(l[0] - f[0])
    sign = 1.0 if turn > 0 else -1.0
    if abs(abs(turn) - 90) > 35:
        print(f'\n   Left is {turn:+.0f} deg from ahead, but it should be about +-90.')
        print('   Probably the tool saw you instead of the box. Keep your body more than 60 cm')
        print('   from the robot when pressing Enter, and run again. No values printed.')
        return
    # follower: robot_angle = sign * laser_angle + yaw  ->  0 = sign * f + yaw
    yaw = wrap(-sign * f[0])
    print('\nPut these in src/agv_follower/config/agv.yaml (shared section):')
    print(f'    laser_yaw_deg: {yaw:.1f}')
    print(f'    laser_angle_sign: {sign:.1f}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
