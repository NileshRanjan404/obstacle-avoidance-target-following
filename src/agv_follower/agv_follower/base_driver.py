#!/usr/bin/env python3
"""base_driver - the only node that talks to the Arduino GIGA.

Subscribes
  /cmd_vel        geometry_msgs/Twist   linear.x m/s, angular.z rad/s (converted to steering)
  /base/raw_cmd   std_msgs/Float32MultiArray [duty_pct, steer_deg]  open-loop, for calibration
Publishes
  /odom           nav_msgs/Odometry     wheel-encoder + steering odometry (bicycle model)
  TF odom -> base_link
  /base/encoder_count  std_msgs/Int64
  /base/speed          std_msgs/Float32  measured m/s
  /base/duty           std_msgs/Float32  duty % applied by the GIGA
  /base/steer_deg      std_msgs/Float32  steering angle applied (+ = left)
  /base/fault          std_msgs/Bool     stall fault active
Services
  /base/clear_fault    std_srvs/Trigger
  /base/reset_odom     std_srvs/Trigger
"""
import glob
import math
import time

import rclpy
import serial
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int64
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster


def find_port():
    for pattern in ('/dev/serial/by-id/*Arduino*', '/dev/ttyACM*'):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class BaseDriver(Node):
    def __init__(self):
        super().__init__('base_driver')
        for name, default in [
            ('port', ''), ('meters_per_count', 0.0001), ('wheelbase', 0.25),
            ('max_steer_deg', 25.0), ('max_speed', 0.6), ('cmd_timeout', 0.3),
            ('servo_center_deg', 90.0), ('servo_dir', 1.0), ('servo_per_wheel_deg', 1.0),
            ('kp', 0.01), ('ki', 0.02), ('kff', 0.015), ('u0', 40.0), ('accel', 0.8),
            ('publish_tf', True), ('allow_reverse', False), ('odom_frame', 'odom'), ('base_frame', 'base_link'),
        ]:
            self.declare_parameter(name, default)

        self.ser = None
        self.rx_buf = b''
        self.last_open_try = 0.0

        self.cmd = (0.0, 0.0)          # (v m/s, steer deg)
        self.cmd_time = 0.0
        self.raw = (0.0, 0.0)          # (duty %, steer deg)
        self.raw_time = 0.0

        # odometry state
        self.x = self.y = self.yaw = 0.0
        self.last_count = None
        self.fault = False

        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.pub_count = self.create_publisher(Int64, '/base/encoder_count', 10)
        self.pub_speed = self.create_publisher(Float32, '/base/speed', 10)
        self.pub_duty = self.create_publisher(Float32, '/base/duty', 10)
        self.pub_steer = self.create_publisher(Float32, '/base/steer_deg', 10)
        self.pub_fault = self.create_publisher(Bool, '/base/fault', 10)
        self.tf = TransformBroadcaster(self)

        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.create_subscription(Float32MultiArray, '/base/raw_cmd', self.on_raw, 10)
        self.create_service(Trigger, '/base/clear_fault', self.on_clear_fault)
        self.create_service(Trigger, '/base/reset_odom', self.on_reset_odom)
        self.add_on_set_parameters_callback(self.on_params)

        self.create_timer(0.02, self.send_command)   # 50 Hz
        self.create_timer(0.005, self.read_serial)   # drain serial quickly
        self.open_serial()

    # ---------------- helpers ----------------
    def p(self, name):
        return self.get_parameter(name).value

    def open_serial(self):
        self.last_open_try = time.monotonic()
        port = self.p('port') or find_port()
        if not port:
            self.get_logger().warn('GIGA not found, retrying...', throttle_duration_sec=5.0)
            return
        try:
            self.ser = serial.Serial(port, 115200, timeout=0, write_timeout=0.1)
        except serial.SerialException as e:
            self.get_logger().warn(f'cannot open {port}: {e}', throttle_duration_sec=5.0)
            self.ser = None
            return
        time.sleep(0.5)
        self.ser.reset_input_buffer()
        self.rx_buf = b''
        self.last_count = None
        self.get_logger().info(f'connected to GIGA on {port}')
        self.send_config()
        self.write('R')

    def write(self, line):
        if self.ser is None:
            return
        try:
            self.ser.write((line + '\n').encode())
        except (serial.SerialException, OSError) as e:
            self.get_logger().error(f'serial write failed: {e}')
            self.close_serial()

    def close_serial(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def send_config(self, overrides=None):
        q = dict((n, self.p(n)) for n in (
            'servo_center_deg', 'servo_dir', 'servo_per_wheel_deg', 'max_steer_deg',
            'kp', 'ki', 'kff', 'u0', 'accel', 'meters_per_count'))
        if overrides:
            q.update(overrides)
        accel_cps2 = q['accel'] / q['meters_per_count']
        self.write('T %.2f %d %.2f %.3f' % (q['servo_center_deg'], 1 if q['servo_dir'] >= 0 else -1,
                                            q['max_steer_deg'], q['servo_per_wheel_deg']))
        self.write('G %.6f %.6f %.6f %.2f %.1f' % (q['kp'], q['ki'], q['kff'], q['u0'], accel_cps2))

    # ---------------- callbacks ----------------
    def on_params(self, params):
        changed = {p.name: p.value for p in params}
        if set(changed) & {'servo_center_deg', 'servo_dir', 'servo_per_wheel_deg', 'max_steer_deg',
                           'kp', 'ki', 'kff', 'u0', 'accel', 'meters_per_count'}:
            self.send_config(changed)
        return SetParametersResult(successful=True)

    def on_cmd_vel(self, msg):
        v_min = -self.p('max_speed') if self.p('allow_reverse') else 0.0
        v = max(v_min, min(self.p('max_speed'), msg.linear.x))
        wz = msg.angular.z
        max_steer = self.p('max_steer_deg')
        if abs(v) > 0.02:
            steer = math.degrees(math.atan(self.p('wheelbase') * wz / v))
        else:
            steer = max_steer * max(-1.0, min(1.0, wz))   # standing still: turn the wheels only
        steer = max(-max_steer, min(max_steer, steer))
        self.cmd = (v, steer)
        self.cmd_time = time.monotonic()

    def on_raw(self, msg):
        if len(msg.data) >= 2:
            self.raw = (float(msg.data[0]), float(msg.data[1]))
            self.raw_time = time.monotonic()

    def on_clear_fault(self, _req, resp):
        self.write('X')
        resp.success = True
        resp.message = 'clear sent'
        return resp

    def on_reset_odom(self, _req, resp):
        self.x = self.y = self.yaw = 0.0
        resp.success = True
        resp.message = 'odometry reset'
        return resp

    # ---------------- periodic ----------------
    def send_command(self):
        if self.ser is None:
            if time.monotonic() - self.last_open_try > 2.0:
                self.open_serial()
            return
        now = time.monotonic()
        timeout = self.p('cmd_timeout')
        if now - self.raw_time < timeout:
            duty, steer = self.raw
            self.write('D %.1f %.1f' % (duty, steer))
        elif now - self.cmd_time < timeout:
            v, steer = self.cmd
            self.write('C %.1f %.1f' % (v / self.p('meters_per_count'), steer))
        else:
            self.write('C 0 0')

    def read_serial(self):
        if self.ser is None:
            return
        try:
            data = self.ser.read(4096)
        except (serial.SerialException, OSError) as e:
            self.get_logger().error(f'serial read failed: {e}')
            self.close_serial()
            return
        if not data:
            return
        self.rx_buf += data
        *lines, self.rx_buf = self.rx_buf.split(b'\n')
        for raw in lines:
            line = raw.decode(errors='ignore').strip()
            if line.startswith('E,'):
                self.handle_telemetry(line)
            elif line.startswith('#'):
                self.get_logger().info('GIGA ' + line)

    def handle_telemetry(self, line):
        parts = line.split(',')
        if len(parts) != 7:
            return
        try:
            count = int(parts[2])
            cps = float(parts[3])
            duty = float(parts[4])
            steer_deg = float(parts[5])
            flags = int(parts[6])
        except ValueError:
            return

        mpc = self.p('meters_per_count')
        L = self.p('wheelbase')
        if self.last_count is None:
            self.last_count = count
        dcount = count - self.last_count
        self.last_count = count
        if abs(dcount) > 100000:      # GIGA reset or garbage
            dcount = 0

        ds = dcount * mpc
        steer = math.radians(steer_deg)
        dyaw = ds * math.tan(steer) / L
        mid = self.yaw + dyaw / 2.0
        self.x += ds * math.cos(mid)
        self.y += ds * math.sin(mid)
        self.yaw = math.atan2(math.sin(self.yaw + dyaw), math.cos(self.yaw + dyaw))

        v = cps * mpc
        wz = v * math.tan(steer) / L
        stamp = self.get_clock().now().to_msg()
        q = yaw_to_quat(self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.p('odom_frame')
        odom.child_frame_id = self.p('base_frame')
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = q[0], q[1]
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = q[2], q[3]
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = wz
        self.pub_odom.publish(odom)

        if self.p('publish_tf'):
            t = TransformStamped()
            t.header = odom.header
            t.child_frame_id = odom.child_frame_id
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation = odom.pose.pose.orientation
            self.tf.sendTransform(t)

        self.pub_count.publish(Int64(data=count))
        self.pub_speed.publish(Float32(data=v))
        self.pub_duty.publish(Float32(data=duty))
        self.pub_steer.publish(Float32(data=steer_deg))
        fault = bool(flags & 1)
        if fault != self.fault:
            self.fault = fault
            if fault:
                self.get_logger().error('STALL fault: call  ros2 service call /base/clear_fault '
                                        'std_srvs/srv/Trigger  after checking the robot')
        self.pub_fault.publish(Bool(data=fault))

    def stop(self):
        for _ in range(3):
            self.write('C 0 0')
            time.sleep(0.03)
        self.close_serial()


def main():
    rclpy.init()
    node = BaseDriver()
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
