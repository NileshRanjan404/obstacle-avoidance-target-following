#!/usr/bin/env python3
"""ball_detector - finds the yellow smiley ball in the webcam image.

Publishes
  /ball                      geometry_msgs/PointStamped  ball centre in base_link (x fwd, y left),
                                                         only when the ball is seen
  /ball/debug_image/compressed  sensor_msgs/CompressedImage  annotated view (rqt_image_view)

How it works
  1. HSV threshold for yellow.
  2. Morphological close with a large kernel fills the smiley's dark eyes and mouth,
     so the ball becomes one solid disc again.
  3. Every blob is scored by size and how well it fills its enclosing circle (roundness).
  4. Distance comes from the apparent size:  range = focal_px * ball_diameter / diameter_px.
     Bearing comes from the horizontal pixel position.
"""
import math
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')
        for name, default in [
            ('device', '/dev/video0'), ('width', 640), ('height', 480), ('fps', 30),
            ('v4l2_controls', 'exposure_dynamic_framerate=0,power_line_frequency=1'),
            ('hfov_deg', 60.0), ('focal_px', 0.0), ('center_offset_px', 0.0),
            ('hsv_low', [20, 100, 100]), ('hsv_high', [35, 255, 255]),
            ('min_radius_px', 6.0), ('min_fill', 0.55), ('close_kernel_px', 15),
            ('debug_rate', 10.0), ('debug_view', 'overlay'), ('stream_port', 8080),
            ('ball_diameter', 0.10), ('camera_x', 0.30), ('camera_y', 0.0),
            ('camera_yaw_deg', 0.0), ('base_frame', 'base_link'),
        ]:
            self.declare_parameter(name, default)
        self.cfg = {}
        self.load_params()
        self.add_on_set_parameters_callback(self.on_params)

        self.pub_ball = self.create_publisher(PointStamped, '/ball', 10)
        self.pub_dbg = self.create_publisher(CompressedImage, '/ball/debug_image/compressed', 2)

        self.cap = None
        self.prev = None            # (u, v, r) of the last detection, used to prefer the same blob
        self.last_dbg = 0.0
        self.frames = 0
        self.fps_t = time.monotonic()
        self.running = True
        self.latest_jpg = None
        self.jpg_cond = threading.Condition()
        port = int(self.get_parameter('stream_port').value)
        if port > 0:
            self.start_stream(port)
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    # ---------------- web view: http://<pi-ip>:<stream_port> ----------------
    def start_stream(self, port):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path != '/stream':
                    page = (b'<html><head><title>AGV camera</title></head><body style="margin:0;background:#111">'
                            b'<img src="/stream" style="width:100%;max-width:960px;display:block;margin:auto">'
                            b'</body></html>')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(page)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while node.running:
                        with node.jpg_cond:
                            node.jpg_cond.wait(timeout=1.0)
                            jpg = node.latest_jpg
                        if jpg is None:
                            continue
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                         str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass

        try:
            server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        except OSError as e:
            self.get_logger().warn(f'web view not started on port {port}: {e}')
            return
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.get_logger().info(f'camera web view on http://<pi-ip>:{port}')

    # ---------------- parameters ----------------
    def load_params(self, overrides=None):
        names = ['hfov_deg', 'focal_px', 'center_offset_px', 'hsv_low', 'hsv_high',
                 'min_radius_px', 'min_fill', 'close_kernel_px', 'debug_rate', 'debug_view',
                 'ball_diameter', 'camera_x', 'camera_y', 'camera_yaw_deg', 'base_frame', 'width']
        cfg = {n: self.get_parameter(n).value for n in names}
        if overrides:
            cfg.update(overrides)
        k = max(1, int(cfg['close_kernel_px']))
        cfg['kernel'] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.cfg = cfg

    def on_params(self, params):
        self.load_params({p.name: p.value for p in params})
        return SetParametersResult(successful=True)

    # ---------------- camera ----------------
    def apply_v4l2(self, dev):
        """Camera controls, e.g. 'exposure_dynamic_framerate=0,power_line_frequency=1'."""
        ctrls = self.get_parameter('v4l2_controls').value
        if not ctrls:
            return
        r = subprocess.run(['v4l2-ctl', '-d', dev, '-c', ctrls], capture_output=True, text=True)
        if r.returncode != 0:
            self.get_logger().warn(f'v4l2-ctl: {r.stderr.strip()}')

    def open_camera(self):
        dev = self.get_parameter('device').value
        self.apply_v4l2(dev)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.get_logger().warn(f'cannot open camera {dev}, retrying...', throttle_duration_sec=5.0)
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.get_parameter('width').value)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.get_parameter('height').value)
        cap.set(cv2.CAP_PROP_FPS, self.get_parameter('fps').value)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'camera {dev} open at {w}x{h}')
        self.apply_v4l2(dev)       # some controls only stick once streaming is configured
        return cap

    def loop(self):
        while self.running and rclpy.ok():
            if self.cap is None:
                self.cap = self.open_camera()
                if self.cap is None:
                    time.sleep(2.0)
                    continue
            ok, frame = self.cap.read()
            if not ok:
                self.get_logger().warn('camera read failed, reopening')
                self.cap.release()
                self.cap = None
                continue
            stamp = self.get_clock().now().to_msg()
            self.process(frame, stamp)
            self.frames += 1
            now = time.monotonic()
            if now - self.fps_t > 10.0:
                self.get_logger().info(f'camera {self.frames / (now - self.fps_t):.1f} fps')
                self.frames, self.fps_t = 0, now

    # ---------------- detection ----------------
    def detect(self, frame):
        c = self.cfg
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(c['hsv_low'], np.uint8), np.array(c['hsv_high'], np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, c['kernel'])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, best_score = None, 0.0
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            (u, v), r = cv2.minEnclosingCircle(hull)
            if r < c['min_radius_px']:
                continue
            fill = cv2.contourArea(hull) / (math.pi * r * r)
            if fill < c['min_fill']:
                continue
            score = fill * r * r
            if self.prev is not None:            # small bonus for staying on the same blob
                d = math.hypot(u - self.prev[0], v - self.prev[1])
                score *= 1.0 + 0.5 * math.exp(-d / max(self.prev[2], 1.0))
            if score > best_score:
                best, best_score = (u, v, r, fill), score
        return best, mask

    def process(self, frame, stamp):
        c = self.cfg
        h, w = frame.shape[:2]
        best, mask = self.detect(frame)
        f = c['focal_px'] if c['focal_px'] > 0 else (w / 2.0) / math.tan(math.radians(c['hfov_deg']) / 2.0)
        cx0 = w / 2.0 + c['center_offset_px']

        info = None
        if best is not None:
            u, v, r, fill = best
            self.prev = (u, v, r)
            depth = f * c['ball_diameter'] / (2.0 * r)        # along the optical axis
            lateral = -(u - cx0) * depth / f                  # + = left
            yaw = math.radians(c['camera_yaw_deg'])
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = c['base_frame']
            msg.point.x = c['camera_x'] + depth * math.cos(yaw) - lateral * math.sin(yaw)
            msg.point.y = c['camera_y'] + depth * math.sin(yaw) + lateral * math.cos(yaw)
            msg.point.z = 0.0
            self.pub_ball.publish(msg)
            clipped = u - r < 2 or u + r > w - 2 or v - r < 2 or v + r > h - 2
            info = (u, v, r, fill, depth, math.degrees(math.atan2(lateral, depth)), clipped)
        else:
            self.prev = None

        if c['debug_rate'] > 0 and time.monotonic() - self.last_dbg > 1.0 / c['debug_rate']:
            self.last_dbg = time.monotonic()
            self.publish_debug(frame, mask, info, cx0, stamp)

    def publish_debug(self, frame, mask, info, cx0, stamp):
        if self.cfg['debug_view'] == 'mask':
            img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            img = frame.copy()
        h = img.shape[0]
        cv2.line(img, (int(cx0), 0), (int(cx0), h), (255, 255, 255), 1)
        if info:
            u, v, r, fill, depth, bearing, clipped = info
            col = (0, 165, 255) if clipped else (0, 255, 0)
            cv2.circle(img, (int(u), int(v)), int(r), col, 2)
            cv2.putText(img, f'{depth:.2f} m  {bearing:+.1f} deg  r={r:.0f}px  fill={fill:.2f}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        else:
            cv2.putText(img, 'no ball', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.jpg_cond:                  # hand the frame to the web stream
                self.latest_jpg = jpg.tobytes()
                self.jpg_cond.notify_all()
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = 'camera_link'
            msg.format = 'jpeg'
            msg.data = jpg.tobytes()
            self.pub_dbg.publish(msg)

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


def main():
    rclpy.init()
    node = BallDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
