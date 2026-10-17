#!/usr/bin/env python3
"""motor_calib.py - check the encoder and tune the GIGA speed loop, with the DRIVE WHEELS LIFTED.

Run on the Pi with the ROS launch STOPPED:
    python3 ~/agv_ws/tools/motor_calib.py

Counts per wheel turn (2340) come from the original RPM.ino.
Steps (each can be skipped with 's'):
  1. start   duty ramps up slowly until the wheel moves -> static duty u0
  2. sweep   fixed duties forward and reverse -> speed vs duty, direction check, kff
  3. loop    closed-loop speed steps with the fitted gains -> tracking check

Ctrl+C at any time stops the motor.
"""
import glob
import math
import os
import sys
import time

import serial

WHEEL_D = 0.07
CPR_EXPECTED = 2340
MAX_DUTY = 70.0


def find_port():
    for pattern in ('/dev/serial/by-id/*Arduino*', '/dev/ttyACM*'):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    sys.exit('GIGA not found')


def base_driver_running():
    me = os.getpid()
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                if b'lib/agv_follower/base_driver' in f.read():
                    return True
        except OSError:
            pass
    return False


class Giga:
    def __init__(self):
        self.s = serial.Serial(find_port(), 115200, timeout=0.05)
        time.sleep(0.5)
        self.s.reset_input_buffer()
        self.buf = b''
        self.count = 0
        self.cps = 0.0
        self.duty = 0.0
        self.flags = 0

    def send(self, line):
        self.s.write((line + '\n').encode())

    def poll(self):
        self.buf += self.s.read(4096)
        *lines, self.buf = self.buf.split(b'\n')
        for raw in lines:
            line = raw.decode(errors='ignore').strip()
            if line.startswith('E,'):
                p = line.split(',')
                if len(p) == 7:
                    self.count, self.cps = int(p[2]), float(p[3])
                    self.duty, self.flags = float(p[4]), int(p[6])
            elif line.startswith('#'):
                print('  GIGA ' + line)

    def hold(self, cmd, seconds, sample_from=None):
        """Repeat cmd for `seconds` (keeps the watchdog fed); return cps samples after sample_from s."""
        t0 = time.monotonic()
        samples = []
        while True:
            t = time.monotonic() - t0
            if t >= seconds:
                return samples
            self.send(cmd)
            end = time.monotonic() + 0.05
            while time.monotonic() < end:
                self.poll()
            if sample_from is not None and t >= sample_from:
                samples.append(self.cps)

    def stop(self):
        for _ in range(5):
            self.send('D 0 0')
            time.sleep(0.03)


def ask(prompt):
    return input(prompt).strip().lower()


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if xs else 0.0


def main():
    if base_driver_running():
        sys.exit('base_driver is running - stop the ROS launch first.')
    print(__doc__)
    if ask('Are the drive wheels OFF the ground? [y/N] ') != 'y':
        sys.exit('Lift the robot first.')
    g = Giga()
    g.send('T 85 1 25 1.0')         # steering straight while testing
    g.send('X')
    res = {}
    try:
        # counts per wheel turn come from the original RPM.ino (2340, both edges) - no hand step

        # ---------- 2. start ----------
        if ask('\n1. START: find the duty where the wheel starts to turn? [Enter/s] ') != 's':
            u_start = None
            duty = 5.0
            while duty <= MAX_DUTY:
                g.hold('D %.1f 0' % duty, 0.25)
                if abs(g.cps) > 30:
                    u_start = duty
                    break
                duty += 1.0
            g.stop()
            g.hold('D 0 0', 1.0)
            if u_start is None:
                print(f'   wheel never moved up to {MAX_DUTY:.0f}% - check the motor power / BTS7960.')
            else:
                print(f'   starts turning at {u_start:.0f}% duty')
                res['u_start'] = u_start

        # ---------- 3. sweep ----------
        if ask('\n2. SWEEP: run fixed duties forward and reverse (about 30 s)? [Enter/s] ') != 's':
            lo = max(20.0, res.get('u_start', 30.0))
            duties = [d for d in (lo, lo + 10, lo + 20, lo + 30, MAX_DUTY) if d <= MAX_DUTY]
            duties = sorted(set(round(d) for d in duties))
            pts = []
            print('   duty %   counts/s     m/s     (std)')
            for sign in (1, -1):
                for d in duties:
                    s = g.hold('D %.1f 0' % (sign * d), 2.0, sample_from=1.2)
                    c = mean(s)
                    print(f'   {sign * d:+6.0f}   {c:+9.0f}   {c * math.pi * WHEEL_D / CPR_EXPECTED:+6.2f}   ({std(s):.0f})')
                    pts.append((sign * d, c))
                g.stop()
                g.hold('D 0 0', 1.5)
                if sign == 1:
                    fwd = ask('   Did the wheel turn so the robot would drive FORWARD? [y/n] ')
                    res['fwd_ok'] = fwd == 'y'
            # direction of counts
            fwd_c = [c for d, c in pts if d > 0]
            rev_c = [c for d, c in pts if d < 0]
            if fwd_c and rev_c and mean(fwd_c) > 0 > mean(rev_c):
                print('   counts go up forward and down in reverse: OK')
            # linear fit duty = u0 + kff * |cps| over all points
            xs = [abs(c) for d, c in pts if abs(c) > 30]
            ys = [abs(d) for d, c in pts if abs(c) > 30]
            if len(xs) >= 3:
                mx, my = mean(xs), mean(ys)
                kff = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
                u0 = my - kff * mx
                res['kff'], res['u0'] = kff, max(0.0, u0)
                res['max_cps'] = max(xs)
                print(f'   fit: duty = {u0:.1f} + {kff:.5f} * counts/s   (top speed {max(xs) * math.pi * WHEEL_D / CPR_EXPECTED:.2f} m/s)')

        # ---------- 4. loop ----------
        if 'kff' in res and ask('\n3. LOOP: test the speed controller (about 15 s)? [Enter/s] ') != 's':
            kff, u0 = res['kff'], res['u0']
            kp, ki = 0.5 * kff, 2.0 * kff
            res['kp'], res['ki'] = kp, ki
            g.send('G %.6f %.6f %.6f %.2f %.1f' % (kp, ki, kff, u0, 0.8 / (math.pi * WHEEL_D / CPR_EXPECTED)))
            mpc = math.pi * WHEEL_D / CPR_EXPECTED
            print('   target m/s   measured m/s   (std)')
            for v in (0.15, 0.30, 0.45, -0.20):
                if abs(v) / mpc > res['max_cps'] * 0.9:
                    continue
                s = g.hold('C %.1f 0' % (v / mpc), 3.0, sample_from=1.5)
                print(f'   {v:+8.2f}     {mean(s) * mpc:+8.3f}      ({std(s) * mpc:.3f})')
                if abs(mean(s) * mpc - v) > 0.3 * abs(v):
                    print('     tracking is off by more than 30% - tell Claude the numbers above')
            g.send('C 0 0')
            g.hold('C 0 0', 1.0)
    except KeyboardInterrupt:
        print('\n   stopped')
    finally:
        g.stop()

    print('\n================ results ================')
    if 'u_start' in res:
        print(f'start duty: {res["u_start"]:.0f}%')
    if 'fwd_ok' in res:
        print('forward direction: ' + ('OK' if res['fwd_ok'] else 'WRONG - swap the motor wires on the BTS7960'))
    if 'kff' in res:
        print(f'base_driver gains:  u0: {res["u0"]:.1f}   kff: {res["kff"]:.6f}', end='')
        if 'kp' in res:
            print(f'   kp: {res["kp"]:.6f}   ki: {res["ki"]:.6f}', end='')
        print(f'\ntop speed (no load): {res["max_cps"] * math.pi * WHEEL_D / CPR_EXPECTED:.2f} m/s')
    print('Send a screenshot of this to Claude.')


if __name__ == '__main__':
    main()
