"""tracker.py - constant-velocity Kalman filter for the ball, in the odom (world) frame.

No ROS in here, so it can be unit-tested on its own (test/test_tracker.py).

State: [x, y, vx, vy]. Each camera detection updates it; between detections (or when the ball is
out of view) the position is predicted forward with the estimated velocity. That lets the robot
  - keep driving toward a MOVING ball that left the camera's narrow field of view, and
  - trust a STILL ball's position for much longer (e.g. while going around an obstacle).
Detections that are far from the prediction (Mahalanobis gate) are treated as outliers; if they
keep coming, the filter assumes the ball really jumped and restarts there.
"""
import math

import numpy as np


class BallTracker:
    def __init__(self, accel_noise=0.6, meas_noise=0.04, meas_noise_per_m=0.05,
                 max_speed=0.6, gate=13.8, outliers_to_reset=3):
        self.q = accel_noise                # m/s^2, how quickly the ball may change velocity
        self.r0 = meas_noise                # m, detection noise at 0 m
        self.r1 = meas_noise_per_m          # m per m of range (size-based distance gets worse far away)
        self.max_speed = max_speed          # m/s, cap on the velocity estimate
        self.gate = gate                    # chi-square 99.9 % for 2 dof
        self.outliers_to_reset = outliers_to_reset
        self.reset()

    def reset(self):
        self.x = None                       # state vector, None = nothing tracked yet
        self.P = None
        self.t = None                       # time of the state
        self.t_seen = None                  # time of the last accepted detection
        self.n_updates = 0
        self.outliers = 0

    @property
    def active(self):
        return self.x is not None

    # ---------------- filter ----------------
    def _predict_to(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        g = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        Q = g @ g.T * self.q ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, zx, zy, t, rng=1.0):
        """Feed one detection (odom frame) taken at time t, rng = camera range (for the noise).
        Returns True if it was used, False if it was rejected as an outlier."""
        r = self.r0 + self.r1 * max(rng, 0.0)
        R = np.eye(2) * r * r
        z = np.array([zx, zy], float)
        if self.x is None:
            self._start(z, t, R)
            return True
        self._predict_to(t)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > self.gate:
            self.outliers += 1
            if self.outliers >= self.outliers_to_reset:
                self._start(z, t, R)        # it keeps appearing there: the ball really moved
                return True
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        sp = math.hypot(self.x[2], self.x[3])
        if sp > self.max_speed:
            self.x[2:] *= self.max_speed / sp
        self.t_seen = t
        self.n_updates += 1
        self.outliers = 0
        return True

    def _start(self, z, t, R):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([R[0, 0], R[1, 1], 0.3 ** 2, 0.3 ** 2])
        self.t = self.t_seen = t
        self.n_updates = 1
        self.outliers = 0

    # ---------------- queries ----------------
    def age(self, t):
        return float('inf') if self.t_seen is None else t - self.t_seen

    def velocity(self):
        """Estimated (vx, vy); zero until a few detections have been seen."""
        if self.x is None or self.n_updates < 5:
            return 0.0, 0.0
        return float(self.x[2]), float(self.x[3])

    def speed(self):
        vx, vy = self.velocity()
        return math.hypot(vx, vy)

    def position(self, t, max_extrapolation=2.0, lead=0.0):
        """Predicted (x, y) at time t + lead. Prediction beyond the last detection is limited to
        max_extrapolation seconds (a ball out of view should not be extrapolated forever)."""
        if self.x is None:
            return None
        vx, vy = self.velocity()
        dt = min(max(t + lead - self.t, 0.0), max(max_extrapolation - (self.t - self.t_seen), 0.0))
        return float(self.x[0] + vx * dt), float(self.x[1] + vy * dt)
