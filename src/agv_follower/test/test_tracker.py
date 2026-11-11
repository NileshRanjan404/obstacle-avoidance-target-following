"""Unit tests for the ball tracker: run with  python3 -m pytest -q  in src/agv_follower."""
import math
import random

from agv_follower.tracker import BallTracker


def feed(tr, path, dt=0.066, noise=0.0, t0=0.0):
    t = t0
    for (x, y) in path:
        tr.update(x + random.gauss(0, noise), y + random.gauss(0, noise), t, rng=1.0)
        t += dt
    return t - dt


def test_still_ball_converges_and_has_no_velocity():
    random.seed(1)
    tr = BallTracker()
    t = feed(tr, [(2.0, 0.5)] * 40, noise=0.03)
    x, y = tr.position(t)
    assert abs(x - 2.0) < 0.03 and abs(y - 0.5) < 0.03
    assert tr.speed() < 0.05


def test_moving_ball_velocity_is_estimated():
    random.seed(2)
    tr = BallTracker()
    dt = 0.066
    path = [(1.0 + 0.2 * i * dt, 0.1 * i * dt) for i in range(60)]   # 0.2, 0.1 m/s
    t = feed(tr, path, dt=dt, noise=0.02)
    vx, vy = tr.velocity()
    assert abs(vx - 0.2) < 0.05 and abs(vy - 0.1) < 0.05


def test_prediction_continues_while_unseen_but_is_capped():
    tr = BallTracker()
    dt = 0.066
    t = feed(tr, [(0.25 * i * dt, 0.0) for i in range(60)], dt=dt)
    x_last = tr.position(t)[0]
    x_1s = tr.position(t + 1.0, max_extrapolation=2.0)[0]
    x_10s = tr.position(t + 10.0, max_extrapolation=2.0)[0]
    assert 0.18 < x_1s - x_last < 0.32          # about 0.25 m in 1 s
    assert x_10s - x_last < 0.6                 # capped at ~2 s worth


def test_single_outlier_is_rejected_then_jump_is_accepted():
    tr = BallTracker()
    t = feed(tr, [(2.0, 0.0)] * 30)
    assert tr.update(5.0, 3.0, t + 0.07) is False        # one wild detection: ignored
    assert abs(tr.position(t + 0.07)[0] - 2.0) < 0.05
    tr.update(5.0, 3.0, t + 0.14)
    tr.update(5.0, 3.0, t + 0.21)                         # third in a row: the ball really moved
    x, y = tr.position(t + 0.21)
    assert abs(x - 5.0) < 0.05 and abs(y - 3.0) < 0.05


def test_age_and_reset():
    tr = BallTracker()
    assert tr.position(0.0) is None and math.isinf(tr.age(0.0))
    tr.update(1.0, 1.0, 5.0)
    assert abs(tr.age(7.5) - 2.5) < 1e-9
    tr.reset()
    assert not tr.active
