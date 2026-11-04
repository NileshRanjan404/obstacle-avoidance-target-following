#!/usr/bin/env python3
"""follower - decides how the AGV drives.

Subscribes
  /ball   geometry_msgs/PointStamped   ball position in base_link (from ball_detector)
  /scan   sensor_msgs/LaserScan        lidar (obstacles only; the ball is below the scan plane)
  /odom   nav_msgs/Odometry            used to remember where the ball was when it is not visible
Publishes
  /cmd_vel          geometry_msgs/Twist
  /follower/state   std_msgs/String     current state (see below)
  /follower/path    nav_msgs/Path       the arc that was chosen (view in rviz2, frame base_link)

States
  WAIT_SENSORS  lidar or odometry not arriving -> stopped
  FOLLOW        ball visible, driving toward it
  MEMORY        ball just lost, driving to where it was last seen
  ARRIVED       within stop_distance of the ball -> stopped, wheels pointed at the ball
  REPOSITION    ball is inside the minimum turning circle (beside the robot): swing away at full
                lock to make room, then approach it
  AVOID_TIGHT   no arc keeps safety_margin, but one keeps squeeze_margin -> crawling around it
  BLOCKED       every arc toward the ball hits an obstacle even with the thin margin -> stopped
  RECOVER       blocked for too long -> reversing a short way with opposite steering
  WAIT_CLOSE    ball lost while within stop_distance + close_wait_margin (it is probably just
                below the camera's view) -> stay stopped until it is seen again
  SEARCH        ball not seen -> explore: turn a full circle on the spot, then drive ~1 m (steering
                round obstacles), circle the other way, and so on, until the ball is spotted
  IDLE          nothing to do (search timed out, or never saw the ball)
  DISABLED      parameter enabled = false

Planner
  The car can only drive arcs (Ackermann). Each cycle n_arcs steering angles (plus the exact
  pure-pursuit angle to the ball) are simulated for `horizon` metres. The robot footprint
  (inflated by safety_margin) is swept along each arc and checked against lidar points.
  Arcs that cannot keep stop_buffer + braking distance of free space are rejected.
  The rest are scored:
     w_goal   * (distance from the arc end to the ball + w_block * fraction of the arc that is blocked)
   + w_trap   * how trapped the robot would be escape_check_m along the arc (no turn-away room)
   + w_clear  * how close the arc passes to obstacles
   + w_smooth * change of steering from last cycle
  The cheapest arc wins and its speed is limited so it can always brake before the first obstacle.
"""
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from agv_follower.tracker import BallTracker


def arc_poses(s, steer_rad, wheelbase):
    """Poses (x, y, yaw) after driving distances s along an arc from the origin."""
    k = math.tan(steer_rad) / wheelbase
    if abs(k) < 1e-6:
        return s, np.zeros_like(s), np.zeros_like(s)
    th = k * s
    return np.sin(th) / k, (1.0 - np.cos(th)) / k, th


class Follower(Node):
    def __init__(self):
        super().__init__('follower')
        for name, default in [
            ('enabled', True), ('rate', 20.0),
            ('wheelbase', 0.25), ('max_steer_deg', 25.0),
            ('footprint_front', 0.35), ('footprint_rear', 0.10), ('footprint_width', 0.30),
            ('laser_x', 0.20), ('laser_y', 0.0), ('laser_yaw_deg', 0.0), ('laser_angle_sign', 1.0),
            ('ball_diameter', 0.10),
            ('stop_distance', 0.30), ('stop_hysteresis', 0.05),
            ('max_speed', 0.45), ('min_speed', 0.10), ('speed_gain', 0.8), ('decel', 0.8),
            ('safety_margin', 0.06), ('stop_buffer', 0.10), ('horizon', 1.5), ('arc_step', 0.05),
            ('n_arcs', 15), ('w_goal', 1.0), ('w_block', 1.5),
            ('squeeze_margin', 0.02), ('squeeze_buffer', 0.05), ('w_clear', 0.6), ('w_smooth', 0.2), ('clear_ref', 0.40),
            ('ball_ignore_radius', 0.25), ('memory_timeout', 3.0), ('memory_timeout_still', 8.0),
            ('max_extrapolation', 2.0), ('lead_time', 0.4), ('still_speed', 0.05), ('tracker_accel_noise', 0.6),
            ('search_enabled', True), ('search_speed', 0.15), ('search_timeout', 20.0),
            ('close_wait_margin', 0.25), ('close_wait_timeout', 5.0), ('search_circle_deg', 360.0), ('search_roam_m', 1.0), ('search_circle_room', 1.2), ('reposition_factor', 0.0), ('keepout_fresh_s', 2.0), ('escape_len', 0.8), ('escape_check_m', 0.3), ('w_trap', 3.0), ('search_look_m', 2.5), ('ball_keepout', 0.15),
            ('allow_reverse', False),
            ('blocked_timeout', 1.5), ('recover_distance', 0.30), ('recover_speed', 0.15),
            ('laser_min_range', 0.05), ('self_filter_margin', 0.05), ('laser_max_range', 4.0), ('scan_timeout', 0.5),
        ]:
            self.declare_parameter(name, default)

        self.scan = None
        self.scan_time = 0.0
        self.pose = None                 # (x, y, yaw) in odom
        self.odom_time = 0.0

        self.tracker = BallTracker(accel_noise=self.p('tracker_accel_noise'))   # ball in odom frame
        self.ball_side = 1.0             # +1 = last seen on the left

        self.state = 'WAIT_SENSORS'
        self.prev_steer = 0.0
        self.blocked_since = None
        self.recover_start = None
        self.recover_steer = 0.0
        self.search_start = None
        self.search_dir = 1.0
        self.search_phase = 'CIRCLE'
        self.search_turned = 0.0
        self.search_last_yaw = 0.0
        self.roam_start = (0.0, 0.0, 0.0)
        self.roam_goal = (0.0, 0.0)
        self.last_gap = float('inf')     # gap to the ball when it was last tracked
        self.was_enabled = True

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_state = self.create_publisher(String, '/follower/state', 10)
        self.pub_path = self.create_publisher(Path, '/follower/path', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 20)
        self.create_subscription(PointStamped, '/ball', self.on_ball, 10)
        self.create_timer(1.0 / self.p('rate'), self.tick)

    def p(self, name):
        return self.get_parameter(name).value

    # ---------------- inputs ----------------
    def on_scan(self, msg):
        self.scan = msg
        self.scan_time = time.monotonic()

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self.odom_time = time.monotonic()

    def on_ball(self, msg):
        if self.pose is None:
            return
        x, y, yaw = self.pose
        bx, by = msg.point.x, msg.point.y
        gx = x + math.cos(yaw) * bx - math.sin(yaw) * by
        gy = y + math.sin(yaw) * bx + math.cos(yaw) * by
        now = time.monotonic()
        if not self.tracker.update(gx, gy, now, rng=math.hypot(bx, by)):
            return                               # rejected as an outlier
        if abs(by) > 0.02:
            self.ball_side = 1.0 if by > 0 else -1.0

    def ball_valid(self, now):
        """Is the tracked ball position still worth driving to? A still ball is trusted longer
        (it is probably just out of view), a moving one only briefly."""
        if not self.tracker.active:
            return False
        still = self.tracker.speed() < self.p('still_speed')
        limit = self.p('memory_timeout_still') if still else self.p('memory_timeout')
        return self.tracker.age(now) < limit

    def ball_local(self, now=None, lead=0.0):
        """Tracked (predicted) ball position in the current base_link frame."""
        now = time.monotonic() if now is None else now
        bx, by = self.tracker.position(now, self.p('max_extrapolation'), lead)
        x, y, yaw = self.pose
        dx, dy = bx - x, by - y
        return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)

    def obstacles(self):
        """Lidar points in base_link as an (N, 2) array, with the robot's own body removed."""
        m = self.scan
        r = np.asarray(m.ranges, dtype=np.float32)
        a = m.angle_min + np.arange(len(r), dtype=np.float32) * m.angle_increment
        ok = np.isfinite(r) & (r > max(m.range_min, self.p('laser_min_range'))) & \
            (r < self.p('laser_max_range'))
        r, a = r[ok], a[ok]
        ang = self.p('laser_angle_sign') * a + math.radians(self.p('laser_yaw_deg'))
        pts = np.stack([self.p('laser_x') + r * np.cos(ang), self.p('laser_y') + r * np.sin(ang)], axis=1)
        sm = self.p('self_filter_margin')       # lidar hits on the robot's own parts
        body = (pts[:, 0] < self.p('footprint_front') + sm) & (pts[:, 0] > -self.p('footprint_rear') - sm) & \
            (np.abs(pts[:, 1]) < self.p('footprint_width') / 2.0 + sm)
        return pts[~body]

    def with_ball_ring(self, obs, bx, by):
        """Add a ring of virtual obstacle points around the ball (its RC car is below the lidar)."""
        rr = self.p('ball_diameter') / 2.0 + self.p('ball_keepout')
        ang = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        ring = np.stack([bx + rr * np.cos(ang), by + rr * np.sin(ang)], axis=1)
        return np.concatenate([obs, ring]) if len(obs) else ring

    # ---------------- planning ----------------
    def first_hit(self, xs, ys, ths, obs, margin=None):
        """Index of the first pose whose inflated footprint contains an obstacle, or None."""
        if len(obs) == 0:
            return None
        f0, r0, w0 = self.p('footprint_front'), self.p('footprint_rear'), self.p('footprint_width') / 2.0
        mg = self.p('safety_margin') if margin is None else margin
        # Points already inside the safety margin right now (e.g. a box corner beside the robot)
        # would block every arc, even ones steering away. For those, only the real body
        # (+1 cm) must stay clear; all other points must stay outside the full margin.
        inside_now = (obs[:, 0] <= f0 + mg) & (obs[:, 0] >= -r0 - mg) & (np.abs(obs[:, 1]) <= w0 + mg)
        m = np.where(inside_now, 0.01, mg)[None, :]
        dx = obs[None, :, 0] - xs[:, None]
        dy = obs[None, :, 1] - ys[:, None]
        c, s = np.cos(ths)[:, None], np.sin(ths)[:, None]
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        hit = ((lx <= f0 + m) & (lx >= -r0 - m) & (np.abs(ly) <= w0 + m)).any(axis=1)
        return int(np.argmax(hit)) if hit.any() else None

    def free_length(self, steer_deg, obs, direction=1.0, length=None, margin=None):
        """How far the robot can drive along this arc (direction -1 = reversing)."""
        H = length if length is not None else self.p('horizon')
        step = self.p('arc_step')
        s = np.arange(step, H + 1e-9, step)
        xs, ys, ths = arc_poses(direction * s, math.radians(steer_deg), self.p('wheelbase'))
        i = self.first_hit(xs, ys, ths, obs, margin)
        free = H if i is None else float(s[i] - step)
        return free, (xs, ys, ths)

    def escape_room(self, pose, obs, margin=None):
        """From a future pose (x, y, yaw in the current robot frame): the most free distance the robot
        would have along full-left, straight or full-right. Small = it would be trapped there
        (it cannot reverse and turns wide)."""
        if len(obs) == 0:
            return self.p('escape_len')
        x0, y0, th0 = pose
        c, s = math.cos(th0), math.sin(th0)
        dx, dy = obs[:, 0] - x0, obs[:, 1] - y0
        local = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)
        local = local[np.hypot(local[:, 0], local[:, 1]) < self.p('escape_len') + 0.6]
        maxd = self.p('max_steer_deg')
        return max(self.free_length(d, local, length=self.p('escape_len'), margin=margin)[0]
                   for d in (-maxd, 0.0, maxd))

    def plan(self, obs, goal, v_des, margin=None, buf=None):
        """Choose (steer_deg, speed, poses) toward goal, or None if every arc is blocked.
        margin / buf override safety_margin / stop_buffer (used for the tight 'squeeze' retry)."""
        L, maxd = self.p('wheelbase'), self.p('max_steer_deg')
        gx, gy = goal
        dist = math.hypot(gx, gy)
        pp = math.degrees(math.atan2(2.0 * L * math.sin(math.atan2(gy, gx)), max(dist, 1e-3)))
        pp = max(-maxd, min(maxd, pp))
        cands = list(np.linspace(-maxd, maxd, int(self.p('n_arcs')))) + [pp]

        decel = self.p('decel')
        buf = self.p('stop_buffer') if buf is None else buf
        need_min = self.p('min_speed') ** 2 / (2 * decel) + buf
        near = obs[np.hypot(obs[:, 0], obs[:, 1]) < self.p('horizon') + 1.0] if len(obs) else obs

        best, best_cost = None, float('inf')
        for d in cands:
            # Judge each arc only until the heading has turned 90 deg: a full-lock arc curls back
            # on itself after that, and would otherwise always look worse than going straight.
            k = abs(math.tan(math.radians(d))) / L
            h_arc = self.p('horizon') if k < 1e-6 else min(self.p('horizon'), (math.pi / 2.0) / k)
            free, (xs, ys, ths) = self.free_length(d, near, length=h_arc, margin=margin)
            if free < need_min:
                continue
            h_eff = min(dist, h_arc)
            s_eval = min(free, h_eff)
            ex, ey, _ = arc_poses(np.array([s_eval]), math.radians(d), L)
            # distance still left to the ball, plus a penalty for arcs that run into something
            cost_goal = math.hypot(ex[0] - gx, ey[0] - gy) + \
                self.p('w_block') * max(0.0, h_eff - free) / max(h_eff, 1e-3)
            n_eval = max(1, int(s_eval / self.p('arc_step')))
            if len(near):
                cd = np.hypot(near[None, :, 0] - xs[:n_eval, None], near[None, :, 1] - ys[:n_eval, None])
                clear = float(cd.min()) - self.p('footprint_width') / 2.0
            else:
                clear = self.p('clear_ref')
            cost_clear = max(0.0, self.p('clear_ref') - clear) / self.p('clear_ref')
            cost_smooth = abs(d - self.prev_steer) / (2.0 * maxd)
            # trap check: a little way along this arc, could the robot still turn away?
            s_chk = min(free, self.p('escape_check_m'))
            i_chk = max(0, min(len(xs) - 1, int(s_chk / self.p('arc_step')) - 1))
            room = self.escape_room((xs[i_chk], ys[i_chk], ths[i_chk]), near, margin)
            cost_trap = max(0.0, 1.0 - room / self.p('escape_len'))
            cost = self.p('w_goal') * cost_goal + self.p('w_clear') * cost_clear + \
                self.p('w_smooth') * cost_smooth + self.p('w_trap') * cost_trap
            if cost < best_cost:
                v_allow = math.sqrt(max(0.0, 2.0 * decel * (free - buf)))
                v = min(v_des, v_allow) * (1.0 - 0.3 * abs(d) / maxd)
                v = max(v, self.p('min_speed'))
                n_free = max(1, int(free / self.p('arc_step')))
                best, best_cost = (float(d), v, (xs[:n_free], ys[:n_free], ths[:n_free])), cost
        return best

    # ---------------- output ----------------
    def drive(self, v, steer_deg, poses=None):
        L, maxd = self.p('wheelbase'), self.p('max_steer_deg')
        t = Twist()
        t.linear.x = float(v)
        if abs(v) > 0.02:
            t.angular.z = float(v * math.tan(math.radians(steer_deg)) / L)
        else:
            t.angular.z = float(steer_deg / maxd)     # base_driver: steer wheels while stopped
        self.pub_cmd.publish(t)
        self.prev_steer = steer_deg
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'base_link'
        if poses is not None:
            for x, y, th in zip(*poses):
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x, ps.pose.position.y = float(x), float(y)
                ps.pose.orientation.z, ps.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
                path.poses.append(ps)
        self.pub_path.publish(path)

    def set_state(self, s):
        if s != self.state:
            self.get_logger().info(f'{self.state} -> {s}')
            self.state = s
        self.pub_state.publish(String(data=s))

    # ---------------- main loop ----------------
    def tick(self):
        now = time.monotonic()
        if not self.p('enabled'):
            if self.was_enabled:
                self.drive(0.0, 0.0)
            self.was_enabled = False
            self.set_state('DISABLED')
            return
        self.was_enabled = True

        if self.scan is None or self.pose is None or now - self.scan_time > self.p('scan_timeout') \
                or now - self.odom_time > 0.5:
            self.set_state('WAIT_SENSORS')
            self.drive(0.0, 0.0)
            return

        obs = self.obstacles()

        # ---- recovering: reverse until done, then plan normally again ----
        if self.state == 'RECOVER':
            travelled = math.hypot(self.pose[0] - self.recover_start[0], self.pose[1] - self.recover_start[1])
            free, poses = self.free_length(self.recover_steer, obs, direction=-1.0,
                                           length=self.p('recover_distance'))
            if travelled < self.p('recover_distance') and free > self.p('stop_buffer'):
                self.drive(-self.p('recover_speed'), self.recover_steer, poses)
                return
            self.blocked_since = None

        age = self.tracker.age(now)
        if self.ball_valid(now):
            self.search_start = None
            moving = self.tracker.speed() >= self.p('still_speed')
            gx, gy = self.ball_local(now)                       # where the ball is now
            ax, ay = self.ball_local(now, self.p('lead_time')) if moving else (gx, gy)   # aim point
            radius = self.p('ball_diameter') / 2.0
            if len(obs):    # lidar hits near the ball are its carrier: replace them with a clean ring
                obs = obs[np.hypot(obs[:, 0] - gx, obs[:, 1] - gy) > self.p('ball_ignore_radius')]
            obs = self.with_ball_ring(obs, gx, gy)     # never swipe the (lidar-invisible) RC car
            front, half_w = self.p('footprint_front'), self.p('footprint_width') / 2.0
            # true distance from the robot outline to the ball surface, in every direction
            # (the ball can be beside or behind the robot after a big swerve)
            rear = self.p('footprint_rear')
            gap = math.hypot(max(gx - front, -rear - gx, 0.0), max(abs(gy) - half_w, 0.0)) - radius
            self.last_gap = gap
            stop_at = self.p('stop_distance') + (self.p('stop_hysteresis') if self.state == 'ARRIVED' else 0.0)
            L, maxd = self.p('wheelbase'), self.p('max_steer_deg')
            if gap <= stop_at and gx > front - 0.05:
                pp = math.degrees(math.atan2(2.0 * L * math.sin(math.atan2(gy, gx)), max(math.hypot(gx, gy), 1e-3)))
                self.set_state('ARRIVED')
                self.drive(0.0, max(-maxd, min(maxd, pp)))
                self.blocked_since = None
                return
            # Ball inside the robot's minimum turning circle (beside it): no forward arc can reach
            # it. Swing away at full lock until it is outside the circle, then approach normally.
            R_min = L / max(math.tan(math.radians(maxd)), 1e-3)
            side = 1.0 if ay >= 0 else -1.0
            inside = math.hypot(ax, ay - side * R_min) < R_min * self.p('reposition_factor')
            if inside or (self.state == 'REPOSITION' and
                          math.hypot(ax, ay - side * R_min) < R_min * (self.p('reposition_factor') + 0.15)):
                steer = -side * maxd
                free, poses = self.free_length(steer, obs)
                need = self.p('min_speed') ** 2 / (2 * self.p('decel')) + self.p('stop_buffer')
                if free >= need:
                    self.set_state('REPOSITION')
                    self.drive(self.p('min_speed'), steer, poses)
                    return
            v_des = min(self.p('max_speed'),
                        self.p('min_speed') + self.p('speed_gain') * (gap - self.p('stop_distance')))
            choice = self.plan(obs, (ax, ay), v_des)
            state = 'FOLLOW' if age < 0.3 else 'MEMORY'
            if choice is None:
                # No arc keeps the full margin: try again with a thin margin and crawl around it.
                choice = self.plan(obs, (ax, ay), self.p('min_speed'),
                                   margin=self.p('squeeze_margin'), buf=self.p('squeeze_buffer'))
                state = 'AVOID_TIGHT'
            if choice is None:
                if gap < self.p('stop_distance') + 0.30:
                    # blocked right next to the ball (usually its RC car): that is arriving
                    self.set_state('ARRIVED')
                    self.drive(0.0, self.prev_steer)
                    return
                self.handle_blocked(now, obs, gy)
                return
            self.blocked_since = None
            steer, v, poses = choice
            self.set_state(state)
            self.drive(v, steer, poses)
            return

        # ---- ball lost ----
        # Lost while already close: it most likely dropped below the camera's view.
        # Searching would drive away from it, so stay stopped until it is seen again.
        # After close_wait_timeout it searches anyway (the ball car may have driven off sideways).
        if self.last_gap < self.p('stop_distance') + self.p('close_wait_margin') and \
                self.tracker.speed() < self.p('still_speed') and \
                age < self.p('memory_timeout_still') + self.p('close_wait_timeout'):
            self.set_state('WAIT_CLOSE')
            self.drive(0.0, 0.0)
            return
        if not self.p('search_enabled'):        # (searches even if the ball was never seen)
            self.set_state('IDLE')
            self.drive(0.0, 0.0)
            return
        # ---- SEARCH: explore the room ----
        #   CIRCLE: full lock until the heading has turned search_circle_deg (a full look around)
        #   ROAM:   drive about search_roam_m straight ahead (arc planner, so it steers round
        #           obstacles), then circle again the other way. Repeats until the ball is seen.
        yaw = self.pose[2]
        if self.search_start is None:
            self.search_start = now
            self.search_dir = self.ball_side
            self.search_phase = 'CIRCLE'
            self.search_turned = 0.0
            self.search_last_yaw = yaw
        if now - self.search_start > self.p('search_timeout'):
            self.set_state('IDLE')
            self.drive(0.0, 0.0)
            return
        dyaw = math.atan2(math.sin(yaw - self.search_last_yaw), math.cos(yaw - self.search_last_yaw))
        self.search_last_yaw = yaw
        self.search_turned += abs(math.degrees(dyaw))

        # The ball's RC car is too low for the lidar: keep out of the spot where the ball was last seen.
        # Only a FRESH ball position gets a keep-out ring: the ball keeps moving, and a stale ring
        # around where it used to be would block the search for no reason.
        if self.tracker.active and self.tracker.age(now) < self.p('keepout_fresh_s'):
            obs = self.with_ball_ring(obs, *self.ball_local(now))

        speed = self.p('search_speed')
        maxd = self.p('max_steer_deg')
        L = self.p('wheelbase')

        if self.search_phase == 'ROAM':
            x0, y0, _ = self.roam_start
            travelled = math.hypot(self.pose[0] - x0, self.pose[1] - y0)
            gx_o, gy_o = self.roam_goal
            dx, dy = gx_o - self.pose[0], gy_o - self.pose[1]
            goal = (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)
            choice = None
            if travelled < self.p('search_roam_m') and math.hypot(*goal) > 0.3:
                choice = self.plan(obs, goal, speed)
            if choice is not None:
                steer, v, poses = choice
                self.set_state('SEARCH')
                self.drive(min(v, speed), steer, poses)
                return
            # roam leg finished (or blocked): look around again, turning the other way
            self.search_phase = 'CIRCLE'
            self.search_turned = 0.0
            self.search_dir = -self.search_dir

        # Circle only where a full-lock turn has room; otherwise roam to open space first.
        # (No reverse and wide turns: driving at a wall is how the robot gets stuck.)
        circle_len = math.pi * L / max(math.tan(math.radians(maxd)), 1e-3)      # half a circle
        need = speed ** 2 / (2 * self.p('decel')) + self.p('stop_buffer') + 0.15
        if self.search_turned < self.p('search_circle_deg'):
            for side in (self.search_dir, -self.search_dir):
                free, poses = self.free_length(side * maxd, obs, length=circle_len)
                if free >= min(circle_len, self.p('search_circle_room')):
                    self.search_dir = side
                    self.set_state('SEARCH')
                    self.drive(speed, side * maxd, poses)
                    return
        # Start a roam leg toward the most open direction.
        best = None
        for d in np.linspace(-maxd, maxd, int(self.p('n_arcs'))):
            free, _ = self.free_length(d, obs, length=self.p('search_look_m'))
            score = free - 0.05 * abs(d) / maxd                 # prefer straighter when equal
            if free >= need and (best is None or score > best[0]):
                best = (score, d, free)
        if best is not None:
            _, d, free = best
            s_goal = min(free, self.p('search_look_m'))
            ex, ey, _ = arc_poses(np.array([s_goal]), math.radians(d), L)
            x, y, _ = self.pose
            self.roam_goal = (x + math.cos(yaw) * ex[0] - math.sin(yaw) * ey[0],
                              y + math.sin(yaw) * ex[0] + math.cos(yaw) * ey[0])
            self.roam_start = self.pose
            self.search_phase = 'ROAM'
            self.search_turned = self.p('search_circle_deg')    # circle again after this leg
            choice = self.plan(obs, (ex[0], ey[0]), speed)
            if choice is not None:
                steer, v, poses = choice
                self.set_state('SEARCH')
                self.drive(min(v, speed), steer, poses)
                return
        self.handle_blocked(now, obs, self.ball_side)

    def handle_blocked(self, now, obs, goal_y):
        if self.blocked_since is None:
            self.blocked_since = now
        if self.p('allow_reverse') and now - self.blocked_since > self.p('blocked_timeout'):
            # Preferred: reverse with the wheels turned away from the goal (the nose swings toward it).
            # If the front would clip something, try straight back, then the other way.
            pref = -math.copysign(self.p('max_steer_deg'), goal_y if goal_y != 0 else 1.0)
            for steer in (pref, 0.0, -pref):
                free, _ = self.free_length(steer, obs, direction=-1.0, length=self.p('recover_distance'))
                if free > self.p('stop_buffer'):
                    self.recover_start = self.pose
                    self.recover_steer = steer
                    self.set_state('RECOVER')
                    self.drive(-self.p('recover_speed'), steer)
                    return
        self.set_state('BLOCKED')
        self.drive(0.0, self.prev_steer)


def main():
    rclpy.init()
    node = Follower()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.drive(0.0, 0.0)     # base_driver also stops on its own after cmd_timeout
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
