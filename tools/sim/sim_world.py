#!/usr/bin/env python3
"""sim_world.py - closed-loop test of the follower without the robot.

A kinematic Ackermann robot is driven by /cmd_vel. It publishes /odom, a lidar /scan of box
obstacles and a camera /ball detection, using the SAME geometry as the real robot (read from
config/agv.yaml): wheelbase, steering limit, footprint, the lidar's rotated/mirrored mounting and
the camera's field of view. So the follower runs with the real config, unchanged.

Run it through run_sims.sh (it isolates everything in its own ROS domain), or by hand:
    ROS_DOMAIN_ID=77 python3 sim_world.py box 40

Prints one result line and exits 0 when the scenario passed, 1 when it failed.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

CONFIG = Path.home() / 'agv_ws/src/agv_follower/config/agv.yaml'

# boxes: (centre x, centre y, half length x, half width y) in the world (robot starts at 0,0 facing +x)
# expect: 'arrive' = must end stopped at the ball; 'safe' = only must not collide
SCENARIOS = {
    'clear':  dict(ball=(3.0, 0.0), boxes=[], expect='arrive'),
    'box':    dict(ball=(3.0, 0.0), boxes=[(1.5, 0.0, 0.15, 0.15)], expect='arrive'),
    'offset': dict(ball=(3.0, 0.6), boxes=[(1.5, 0.3, 0.2, 0.2), (2.2, -0.4, 0.2, 0.2)], expect='arrive'),
    'hidden': dict(ball=(-1.5, 2.5), boxes=[(1.2, 0.6, 0.15, 0.15)], expect='arrive'),
    'moving': dict(ball=(2.0, 0.0), boxes=[(2.5, 0.8, 0.2, 0.2)], ball_v=(0.06, 0.03), expect='follow'),
    'wall':   dict(ball=(3.0, 0.0), boxes=[(1.2, 0.0, 0.05, 0.9)], expect='arrive'),
    'near':   dict(ball=(2.0, 0.0), boxes=[(0.75, 0.0, 0.15, 0.15)], expect='safe'),
}


def ray_box(ox, oy, dx, dy, b):
    cx, cy, hx, hy = b
    tmin, tmax = -1e9, 1e9
    for o, d, lo, hi in ((ox, dx, cx - hx, cx + hx), (oy, dy, cy - hy, cy + hy)):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
    return tmin if tmax >= tmin and tmin > 0 else None


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'box'
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
    sc = SCENARIOS[name]
    P = yaml.safe_load(open(CONFIG))
    sh, fo, bd = P['/**']['ros__parameters'], P['follower']['ros__parameters'], P['ball_detector']['ros__parameters']
    L, maxd = sh['wheelbase'], sh['max_steer_deg']
    F, R, HW = sh['footprint_front'], sh['footprint_rear'], sh['footprint_width'] / 2
    lx0, lyaw, lsign = sh['laser_x'], math.radians(sh['laser_yaw_deg']), sh['laser_angle_sign']
    cam_x, cam_yaw = sh['camera_x'], math.radians(sh['camera_yaw_deg'])
    half_fov = math.radians(bd['hfov_deg']) / 2
    ball_r = sh['ball_diameter'] / 2
    stop = fo['stop_distance']

    rclpy.init()
    n = rclpy.create_node('sim_world')
    po = n.create_publisher(Odometry, '/odom', 10)
    ps = n.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
    pb = n.create_publisher(PointStamped, '/ball', 10)
    cmd = [Twist()]
    states = []
    n.create_subscription(Twist, '/cmd_vel', lambda m: cmd.__setitem__(0, m), 10)
    n.create_subscription(String, '/follower/state',
                          lambda m: states.append(m.data) if not states or states[-1] != m.data else None, 10)

    x = y = yaw = steer = 0.0
    ball = list(sc['ball'])
    bv = sc.get('ball_v', (0.0, 0.0))
    dt, t = 0.05, 0.0
    collided, min_gap, gap = False, 9.0, 9.0
    fx, fy = np.meshgrid(np.linspace(-R, F, 10), np.linspace(-HW, HW, 5))
    N = 720
    while t < t_end:
        t0 = time.time()
        for _ in range(10):
            rclpy.spin_once(n, timeout_sec=0.0)
        v, wz = cmd[0].linear.x, cmd[0].angular.z
        target = math.degrees(math.atan(L * wz / v)) if abs(v) > 0.02 else maxd * max(-1, min(1, wz))
        target = max(-maxd, min(maxd, target))
        steer += max(-200 * dt, min(200 * dt, target - steer))
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += v * math.tan(math.radians(steer)) / L * dt
        ball[0] += bv[0] * dt
        ball[1] += bv[1] * dt
        px = x + math.cos(yaw) * fx - math.sin(yaw) * fy
        py = y + math.sin(yaw) * fx + math.cos(yaw) * fy
        for (cx, cy, hx, hy) in sc['boxes']:
            if ((abs(px - cx) <= hx) & (abs(py - cy) <= hy)).any():
                collided = True

        now = n.get_clock().now().to_msg()
        o = Odometry()
        o.header.stamp, o.header.frame_id = now, 'odom'
        o.pose.pose.position.x, o.pose.pose.position.y = x, y
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        po.publish(o)

        # lidar, in the real lidar's own angle convention: robot_angle = sign * a + yaw_offset
        s = LaserScan()
        s.header.stamp, s.header.frame_id = now, 'laser_frame'
        s.angle_min, s.angle_increment = -math.pi, 2 * math.pi / N
        s.angle_max = s.angle_min + (N - 1) * s.angle_increment
        s.range_min, s.range_max = 0.01, 64.0
        lxw, lyw = x + lx0 * math.cos(yaw), y + lx0 * math.sin(yaw)
        ranges = []
        for i in range(N):
            a = s.angle_min + i * s.angle_increment
            wa = yaw + lsign * a + lyaw
            best = float('inf')
            for b in sc['boxes']:
                r = ray_box(lxw, lyw, math.cos(wa), math.sin(wa), b)
                if r is not None and r < best:
                    best = r
            ranges.append(best)
        s.ranges = ranges
        ps.publish(s)

        # camera: ball visible inside the real horizontal field of view, 0.1 .. 4 m
        dx, dy = ball[0] - x, ball[1] - y
        bx = math.cos(yaw) * dx + math.sin(yaw) * dy
        by = -math.sin(yaw) * dx + math.cos(yaw) * dy
        cxr, cyr = bx - cam_x, by
        bearing = math.atan2(cyr, cxr) - cam_yaw
        if cxr > 0.1 and abs(bearing) < half_fov and math.hypot(cxr, cyr) < 4.0:
            m = PointStamped()
            m.header.stamp, m.header.frame_id = now, 'base_link'
            m.point.x, m.point.y = bx, by
            pb.publish(m)
        gap = math.hypot(max(bx - F, -R - bx, 0), max(abs(by) - HW, 0)) - ball_r
        min_gap = min(min_gap, gap)
        t += dt
        time.sleep(max(0.0, dt - (time.time() - t0)))

    expect = sc['expect']
    if collided:
        ok, why = False, 'hit an obstacle'
    elif min_gap < stop - 0.10:
        ok, why = False, f'got too close to the ball ({min_gap:.2f} m)'
    elif expect == 'arrive' and not (stop - 0.07 <= gap <= stop + 0.10):
        ok, why = False, f'did not end at the ball (gap {gap:.2f} m)'
    elif expect == 'follow' and gap > 1.0:
        ok, why = False, f'fell behind the moving ball (gap {gap:.2f} m)'
    else:
        ok, why = True, 'ok'
    print(f'{"PASS" if ok else "FAIL"}  {name:7s} gap {gap:5.2f} m (min {min_gap:5.2f})  '
          f'{why:32s} states: {" > ".join(states[-8:])}')
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
