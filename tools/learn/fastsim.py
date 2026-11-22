"""fastsim.py - fast, ROS-free simulator that runs the REAL follower.py code.

ROS is replaced by tiny stand-ins (fake rclpy / message classes) so the actual Follower class -
its planner, tracker, states and search - runs exactly as on the robot, but in simulated time and
far faster than real time. Used by tune.py to learn the best parameters.

Realism included: wide turns (11 deg wheels), steering slew, motor lag and minimum speed, lidar
at ~7 Hz with noise and its rotated/mirrored mounting, camera at ~15 Hz with the real 37.7 deg
field of view, distance noise and missed detections, random rooms with boxes and walls, and
moving balls.
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]            # agv_ws
CONFIG = ROOT / 'src/agv_follower/config/agv.yaml'


# ---------------------------------------------------------------- ROS stand-ins
class Msg:
    """Message stand-in: unknown attributes become nested messages; `poses` is a list."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        v = [] if name == 'poses' else Msg()
        self.__dict__[name] = v
        return v


class _Param:
    def __init__(self, v):
        self.value = v


class _Pub:
    def __init__(self, sink, topic):
        self.sink, self.topic = sink, topic

    def publish(self, m):
        self.sink[self.topic] = m


class _Logger:
    def info(self, *a, **k): pass
    warn = error = info


class _Clock:
    def now(self):
        return types.SimpleNamespace(to_msg=lambda: Msg())


class FakeNode:
    OVERRIDES = {}

    def __init__(self, name):
        self._params = {}
        self.out = {}

    def declare_parameter(self, name, default):
        self._params[name] = self.OVERRIDES.get(name, default)

    def get_parameter(self, name):
        return _Param(self._params[name])

    def create_publisher(self, _type, topic, _qos):
        return _Pub(self.out, topic)

    def create_subscription(self, *a, **k): pass
    def create_timer(self, *a, **k): pass

    def get_logger(self):
        return _Logger()

    def get_clock(self):
        return _Clock()


def _install_stubs():
    mods = {
        'rclpy': types.ModuleType('rclpy'),
        'rclpy.node': types.ModuleType('rclpy.node'),
        'rclpy.executors': types.ModuleType('rclpy.executors'),
        'rclpy.qos': types.ModuleType('rclpy.qos'),
        'geometry_msgs': types.ModuleType('geometry_msgs'),
        'geometry_msgs.msg': types.ModuleType('geometry_msgs.msg'),
        'nav_msgs': types.ModuleType('nav_msgs'),
        'nav_msgs.msg': types.ModuleType('nav_msgs.msg'),
        'sensor_msgs': types.ModuleType('sensor_msgs'),
        'sensor_msgs.msg': types.ModuleType('sensor_msgs.msg'),
        'std_msgs': types.ModuleType('std_msgs'),
        'std_msgs.msg': types.ModuleType('std_msgs.msg'),
    }
    mods['rclpy.node'].Node = FakeNode
    mods['rclpy.executors'].ExternalShutdownException = type('ExternalShutdownException', (Exception,), {})
    mods['rclpy.qos'].qos_profile_sensor_data = None
    for m in ('PointStamped', 'PoseStamped', 'Twist'):
        setattr(mods['geometry_msgs.msg'], m, Msg)
    for m in ('Odometry', 'Path'):
        setattr(mods['nav_msgs.msg'], m, Msg)
    mods['sensor_msgs.msg'].LaserScan = Msg
    mods['std_msgs.msg'].String = Msg
    sys.modules.update(mods)
    sys.path.insert(0, str(ROOT / 'src/agv_follower'))


_install_stubs()
import agv_follower.follower as follower_mod          # noqa: E402  (the real code)


class SimClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


# ---------------------------------------------------------------- world
def load_params(overrides=None):
    P = yaml.safe_load(open(CONFIG))
    params = dict(P['/**']['ros__parameters'])
    params.update(P['follower']['ros__parameters'])
    cam = P['ball_detector']['ros__parameters']
    params['_hfov_deg'] = cam['hfov_deg']
    if overrides:
        params.update(overrides)
    return params


def random_world(rng, moving_prob=1.0):
    """A deliberately hard random room. Returns a dict.

    kind 'easy'    (30 %): open floor, or 1-2 obstacles well off the path
    kind 'block'   (31 %): 1-2 obstacles, up to 80 cm wide, ON the line between robot and ball
    kind 'gap'     (18 %): a wall across the room between them with one 60-90 cm opening
    kind 'clutter' (21 %): 3-6 obstacles scattered around
    plus 0-2 extra random obstacles in the first two kinds. The ball starts 2-4 m away.
    """
    for _attempt in range(100):
        W, H = rng.uniform(4.5, 6.5), rng.uniform(4.5, 6.5)
        t = 0.05
        walls = [(W / 2, -t, W / 2, t), (W / 2, H + t, W / 2, t), (-t, H / 2, t, H / 2), (W + t, H / 2, t, H / 2)]
        start = (rng.uniform(0.8, W - 0.8), rng.uniform(0.8, H - 0.8), rng.uniform(-math.pi, math.pi))
        for _ in range(100):
            ball = (rng.uniform(0.6, W - 0.6), rng.uniform(0.6, H - 0.6))
            if 2.0 <= math.hypot(ball[0] - start[0], ball[1] - start[1]) <= 4.0:
                break
        else:
            continue
        sx, sy = start[0], start[1]
        dx, dy = ball[0] - sx, ball[1] - sy
        dist = math.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        along_x = abs(ux) >= abs(uy)               # which way the path mostly runs
        boxes = []
        r = rng.random()
        if r < 0.30:
            kind = 'easy'                               # open floor, or 1-2 boxes off the path
        elif r < 0.30 + 0.70 * 0.45:
            kind = 'block'
            for f in sorted(rng.uniform(0.35, 0.7, size=rng.integers(1, 3))):
                off = rng.uniform(-0.15, 0.15)
                cx, cy = sx + ux * dist * f - uy * off, sy + uy * dist * f + ux * off
                across, depth = rng.uniform(0.15, 0.4), rng.uniform(0.1, 0.2)
                boxes.append((cx, cy, depth, across) if along_x else (cx, cy, across, depth))
        elif r < 0.30 + 0.70 * 0.70:
            kind = 'gap'
            gap = rng.uniform(0.6, 0.9)
            if along_x:
                wx = sx + dx / 2
                g = rng.uniform(0.6 + gap / 2, H - 0.6 - gap / 2)
                lo, hi = g - gap / 2, g + gap / 2
                boxes += [(wx, lo / 2, 0.05, lo / 2), (wx, (hi + H) / 2, 0.05, (H - hi) / 2)]
            else:
                wy = sy + dy / 2
                g = rng.uniform(0.6 + gap / 2, W - 0.6 - gap / 2)
                lo, hi = g - gap / 2, g + gap / 2
                boxes += [(lo / 2, wy, lo / 2, 0.05), ((hi + W) / 2, wy, (W - hi) / 2, 0.05)]
        else:
            kind = 'clutter'
        n_extra = {'clutter': rng.integers(3, 7), 'easy': rng.integers(0, 3)}.get(kind, rng.integers(0, 3))
        for _ in range(n_extra):
            hx, hy = rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)
            for _try in range(20):
                cx, cy = rng.uniform(0.6, W - 0.6), rng.uniform(0.6, H - 0.6)
                # easy rooms keep obstacles at least 0.8 m off the straight robot-ball line
                t_ = max(0.0, min(1.0, ((cx - sx) * ux + (cy - sy) * uy) / dist))
                off_line = math.hypot(cx - (sx + ux * dist * t_), cy - (sy + uy * dist * t_))
                if kind != 'easy' or off_line > 0.8 + max(hx, hy):
                    boxes.append((cx, cy, hx, hy))
                    break

        def clear(x, y, c):
            return all(not (abs(x - cx) < hx + c and abs(y - cy) < hy + c) for (cx, cy, hx, hy) in boxes)

        # robot must fit at the start, the ball must not be inside anything
        boxes = [b for b in boxes if math.hypot(b[0] - sx, b[1] - sy) > 0.5 + max(b[2], b[3]) or b[2] < 0.06 or b[3] < 0.06]
        if not clear(sx, sy, 0.45) or not clear(ball[0], ball[1], 0.35):
            continue
        moving = rng.random() < moving_prob
        return dict(W=W, H=H, boxes=walls + boxes, start=start, ball=ball, kind=kind, moving=moving)
    raise RuntimeError('could not build a room')


def raycast(ox, oy, angles, boxes, max_r=8.0):
    """Vectorised ray / axis-aligned-box intersection. Returns ranges (inf = no hit)."""
    dx, dy = np.cos(angles), np.sin(angles)
    best = np.full(len(angles), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        for (cx, cy, hx, hy) in boxes:
            t1x, t2x = (cx - hx - ox) / dx, (cx + hx - ox) / dx
            t1y, t2y = (cy - hy - oy) / dy, (cy + hy - oy) / dy
            tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
            tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
            hit = (tmax >= tmin) & (tmin > 0)
            best = np.where(hit & (tmin < best), tmin, best)
    best[best > max_r] = np.inf
    return best


# ---------------------------------------------------------------- one episode
def run_episode(world, params, seed=0, t_max=90.0, record=False):
    """Simulate one episode. Returns a dict with the reward and what happened."""
    rng = np.random.default_rng(seed)
    clock = SimClock()
    follower_mod.time = clock                               # follower uses time.monotonic()
    FakeNode.OVERRIDES = {k: v for k, v in params.items() if not k.startswith('_')}
    f = follower_mod.Follower()

    L, maxd = params['wheelbase'], params['max_steer_deg']
    F, R, HW = params['footprint_front'], params['footprint_rear'], params['footprint_width'] / 2
    lx0, lyaw, lsign = params['laser_x'], math.radians(params['laser_yaw_deg']), params['laser_angle_sign']
    cam_x, cam_yaw = params['camera_x'], math.radians(params['camera_yaw_deg'])
    half_fov = math.radians(params['_hfov_deg']) / 2
    ball_r, stop = params['ball_diameter'] / 2, params['stop_distance']

    x, y, yaw = world['start']
    bx, by = world['ball']
    moving = world['moving']
    b_head = rng.uniform(-math.pi, math.pi)          # RC car: heading, speed, time to next change
    b_speed, b_timer = 0.0, 0.0
    v = steer = 0.0
    dt = 0.05
    n_rays = 360
    lid_a = -math.pi + np.arange(n_rays) * (2 * math.pi / n_rays)
    fx, fy = np.meshgrid(np.linspace(-R, F, 8), np.linspace(-HW, HW, 5))
    collided = False
    min_gap, gap = 9.0, 9.0
    settled = 0.0
    success_t = None
    near_time = 0.0
    follow_time = 0.0
    catch_t = None
    last_seen_t = -9.0
    fault_bumps = 0
    in_bump = False
    tick = 0
    trace = []

    while clock.t < t_max:
        tick += 1
        # --- ball motion: an RC car someone is driving around ---
        if moving:
            b_timer -= dt
            if b_timer <= 0:                                  # driver changes speed / direction
                b_timer = rng.uniform(2.0, 5.0)
                b_head += rng.uniform(-1.2, 1.2)
                b_speed = 0.0 if rng.random() < 0.15 else rng.uniform(0.05, 0.25)

            def blocked(px, py):
                if not (0.4 < px < world['W'] - 0.4 and 0.4 < py < world['H'] - 0.4):
                    return True
                return any(abs(px - cx) < hx + 0.15 and abs(py - cy) < hy + 0.15
                           for (cx, cy, hx, hy) in world['boxes'][4:])
            if b_speed > 0:
                for _turn in range(8):                        # steer away from walls and obstacles
                    if not blocked(bx + 0.3 * math.cos(b_head), by + 0.3 * math.sin(b_head)):
                        break
                    b_head += rng.choice([-1, 1]) * rng.uniform(0.6, 1.6)
                rx, ry = bx - x, by - y                       # ...and does not ram the robot
                dist = math.hypot(rx, ry)
                if dist < 0.7 and (rx * math.cos(b_head) + ry * math.sin(b_head)) < 0.5 * dist:
                    b_head = math.atan2(ry, rx)
                nbx, nby = bx + b_speed * math.cos(b_head) * dt, by + b_speed * math.sin(b_head) * dt
                if not blocked(nbx, nby):
                    bx, by = nbx, nby

        # --- sensors ---
        f.pose = (x, y, yaw)
        f.odom_time = clock.t
        if tick % 3 == 1:                                    # lidar ~7 Hz
            ox, oy = x + lx0 * math.cos(yaw), y + lx0 * math.sin(yaw)
            world_ang = yaw + lsign * lid_a + lyaw
            r = raycast(ox, oy, world_ang, world['boxes'])
            r = r + rng.normal(0, 0.01, n_rays)
            f.scan = Msg(ranges=list(r), angle_min=-math.pi, angle_increment=2 * math.pi / n_rays, range_min=0.01)
            f.scan_time = clock.t
        rel_x = math.cos(yaw) * (bx - x) + math.sin(yaw) * (by - y)
        rel_y = -math.sin(yaw) * (bx - x) + math.cos(yaw) * (by - y)
        if tick % 4 != 0:                                    # camera ~15 Hz
            cxr, cyr = rel_x - cam_x, rel_y
            rng_c = math.hypot(cxr, cyr)
            bearing = math.atan2(cyr, cxr) - cam_yaw
            if cxr > 0.1 and abs(bearing) < half_fov * 0.95 and rng_c < 3.5 and rng.random() > 0.1:
                k = 1.0 + rng.normal(0, 0.05)               # size-based distance: ~5 % error
                b = bearing + rng.normal(0, math.radians(1.0)) + cam_yaw
                mx, my = cam_x + rng_c * k * math.cos(b), rng_c * k * math.sin(b)
                f.on_ball(Msg(point=Msg(x=mx, y=my)))
                last_seen_t = clock.t

        # --- follower decides ---
        f.tick()
        cmd = f.out.get('/cmd_vel')
        v_cmd, wz = (cmd.linear.x, cmd.angular.z) if cmd is not None else (0.0, 0.0)
        target = math.degrees(math.atan(L * wz / v_cmd)) if abs(v_cmd) > 0.02 else maxd * max(-1, min(1, wz))
        target = max(-maxd, min(maxd, target))
        steer += max(-120 * dt, min(120 * dt, target - steer))        # servo slew
        v_tgt = v_cmd if abs(v_cmd) >= 0.03 else 0.0                   # motor cannot crawl slower
        v += (v_tgt - v) * min(1.0, dt / 0.25)                         # motor lag
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += v * math.tan(math.radians(steer)) / L * dt
        clock.t += dt

        # --- outcome bookkeeping ---
        px = x + math.cos(yaw) * fx - math.sin(yaw) * fy
        py = y + math.sin(yaw) * fx + math.cos(yaw) * fy
        for (cx, cy, hx, hy) in world['boxes']:
            if ((np.abs(px - cx) <= hx) & (np.abs(py - cy) <= hy)).any():
                collided = True
        if collided:
            break
        gx = math.cos(yaw) * (bx - x) + math.sin(yaw) * (by - y)
        gy = -math.sin(yaw) * (bx - x) + math.cos(yaw) * (by - y)
        gap = math.hypot(max(gx - F, -R - gx, 0), max(abs(gy) - HW, 0)) - ball_r
        min_gap = min(min_gap, gap)
        # a bump counts only if it is the robot's fault: driving forward at a ball it could see
        bumping = gap < stop - 0.12
        if bumping and not in_bump and v > 0.05 and gx > 0 and clock.t - last_seen_t < 1.0:
            fault_bumps += 1
        in_bump = bumping
        if record:
            trace.append((clock.t, x, y, yaw, bx, by, f.state))
        if gap < stop + 0.25:
            near_time += dt
        if 0.15 <= gap <= 0.60:                               # close behind the ball
            follow_time += dt
            if catch_t is None:
                catch_t = clock.t
        in_front = gx > F - 0.05
        if not moving and abs(v) < 0.02 and in_front and stop - 0.08 <= gap <= stop + 0.10:
            settled += dt
            if settled >= 1.0 and success_t is None:
                success_t = clock.t
                break
        else:
            settled = 0.0

    # --- reward ---
    if collided:
        reward = -100.0
    elif fault_bumps:
        reward = -50.0                                        # drove into the ball's RC car
    elif moving:
        # share of the time spent close behind the ball, and catching it quickly
        reward = 100.0 * follow_time / max(clock.t, 1e-6) - 0.2 * (catch_t if catch_t is not None else t_max)
    elif success_t is not None:
        reward = 100.0 - 0.5 * success_t                      # faster = better
    else:
        reward = -10.0 - 5.0 * min(gap, 6.0)                  # never got there
    follow = follow_time / max(clock.t, 1e-6)
    success = (follow >= 0.5) if moving else success_t is not None
    out = dict(reward=reward, collided=collided, success=success, moving=moving, follow=follow,
               fault_bumps=fault_bumps,
               catch_t=catch_t,
               time=clock.t, gap=gap, min_gap=min_gap)
    if record:
        out['trace'] = trace
    return out
