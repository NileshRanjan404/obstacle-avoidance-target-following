#!/usr/bin/env python3
"""camera_calib.py - check and correct the ball distance and direction from the webcam.

Needs the robot running WITHOUT the follower:
    ros2 launch agv_follower bringup.launch.py follow:=false
Then in a second terminal:
    source ~/agv_ws/env.sh && python3 ~/agv_ws/tools/camera_calib.py

Measure every distance from the CAMERA LENS to the FRONT of the ball, along the floor.
Keep yourself out of the camera view while it measures (3 s each).
"""
import math
import statistics
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rcl_interfaces.srv import GetParameters

DISTANCES = (0.5, 1.0, 1.5)


class Cal:
    def __init__(self):
        self.node = rclpy.create_node('camera_calib')
        self.pts = []
        self.node.create_subscription(PointStamped, '/ball', lambda m: self.pts.append((m.point.x, m.point.y)), 10)
        self.params = self.get_params(['camera_x', 'camera_y', 'camera_yaw_deg', 'hfov_deg', 'focal_px',
                                       'width', 'ball_diameter'])

    def get_params(self, names):
        cli = self.node.create_client(GetParameters, '/ball_detector/get_parameters')
        if not cli.wait_for_service(timeout_sec=5.0):
            raise SystemExit('ball_detector is not running - start the bringup with follow:=false first.')
        fut = cli.call_async(GetParameters.Request(names=names))
        rclpy.spin_until_future_complete(self.node, fut)
        out = {}
        for n, v in zip(names, fut.result().values):
            out[n] = v.double_value if v.type == 3 else (v.integer_value if v.type == 2 else v.string_value)
        return out

    def measure(self, seconds=3.0):
        """Median (range from camera to ball centre, bearing deg) over `seconds`, or None."""
        self.pts = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if len(self.pts) < 5:
            return None
        cx, cy = self.params['camera_x'], self.params['camera_y']
        rng = [math.hypot(x - cx, y - cy) for x, y in self.pts]
        brg = [math.degrees(math.atan2(y - cy, x - cx)) for x, y in self.pts]
        return statistics.median(rng), statistics.median(brg), len(self.pts)


def main():
    rclpy.init()
    c = Cal()
    p = c.params
    w = p['width']
    f_now = p['focal_px'] if p['focal_px'] > 0 else (w / 2.0) / math.tan(math.radians(p['hfov_deg']) / 2.0)
    r_ball = p['ball_diameter'] / 2.0
    print('Measure from the CAMERA LENS to the FRONT of the ball. Stay out of view while it measures.')
    print(f'current: focal {f_now:.0f} px (hfov {p["hfov_deg"]:.1f} deg), ball {p["ball_diameter"] * 100:.1f} cm, '
          f'camera yaw {p["camera_yaw_deg"]:+.1f} deg\n')

    # ---- distance ----
    ratios, bearings = [], []
    for d in DISTANCES:
        input(f'Put the ball STRAIGHT AHEAD, {d * 100:.0f} cm from the lens to the front of the ball. Enter ')
        m = c.measure()
        if m is None:
            print('   ball not detected - check the camera view (rqt_image_view) and try again later.\n')
            continue
        true_centre = d + r_ball
        ratios.append(true_centre / m[0])
        bearings.append(m[1])
        print(f'   measured {m[0] - r_ball:.2f} m (true {d:.2f})   direction {m[1]:+.1f} deg   ({m[2]} detections)\n')

    # ---- left check ----
    input('Put the ball about 1 m ahead and 30 cm to the robot\'s LEFT. Enter ')
    m = c.measure()
    left_ok = None
    if m is None:
        print('   ball not detected.\n')
    else:
        left_ok = m[1] > 5
        print(f'   direction {m[1]:+.1f} deg (expected about +{math.degrees(math.atan2(0.3, 1.0)):.0f})  '
              f'-> {"OK" if left_ok else "WRONG SIGN - the image may be mirrored"}\n')

    print('================ results ================')
    if ratios:
        k = statistics.median(ratios)
        f_new = f_now / k
        hfov_new = 2 * math.degrees(math.atan((w / 2.0) / f_new))
        spread = (max(ratios) - min(ratios)) / k * 100 if len(ratios) > 1 else 0
        print(f'distance error factor {k:.3f} (spread {spread:.0f} %)')
        if spread > 15:
            print('  readings disagree a lot between distances - the ball may be partly out of colour range')
        print(f'    focal_px: {f_new:.1f}          # (ball_detector section; hfov equivalent {hfov_new:.1f} deg)')
    if bearings:
        b = statistics.median(bearings)
        print(f'    camera_yaw_deg: {p["camera_yaw_deg"] - b:.1f}    # (shared section; straight-ahead read {b:+.1f} deg)')
    if left_ok is False:
        print('  LEFT/RIGHT is mirrored - tell Claude.')
    print('Send a screenshot of this to Claude.')
    c.node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
