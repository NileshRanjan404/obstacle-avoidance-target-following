#!/usr/bin/env python3
"""steer_calib.py - find the servo angle for straight wheels and the steering limits.

Run on the Pi with the ROS launch STOPPED (this script needs the GIGA's serial port):
    python3 ~/agv_ws/tools/steer_calib.py

The drive motor is never commanded, so the robot will not move. Lift the front anyway.

Keys
    a / d     servo angle -1 / +1 deg          A / D   -5 / +5 deg
    g         type an exact servo angle
    c         mark CENTRE  (front wheels pointing straight ahead)
    l         mark full LEFT  (wheels turned left as far as they go WITHOUT the servo straining)
    r         mark full RIGHT
    q         finish and print the values for agv.yaml

Stop turning as soon as the linkage stops moving or the servo buzzes - holding a servo
against its end stop heats it up and can strip its gears.
"""
import glob
import os
import sys
import termios
import threading
import time
import tty

import serial


def find_port():
    for pattern in ('/dev/serial/by-id/*Arduino*', '/dev/ttyACM*'):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    sys.exit('GIGA not found')


def reader(ser, stop):
    while not stop.is_set():
        try:
            line = ser.readline().decode(errors='ignore').strip()
        except Exception:
            return
        if line.startswith('#') and 'servo raw' not in line:
            print('\r  GIGA ' + line + '\r')


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def ask(prompt):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return sys.stdin.readline().strip()


def base_driver_running():
    me = os.getpid()
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read().replace(b'\0', b' ')
        except OSError:
            continue
        if b'lib/agv_follower/base_driver' in cmd:
            return True
    return False


def main():
    if base_driver_running():
        sys.exit('base_driver is running - stop the ROS launch first.')
    port = find_port()
    ser = serial.Serial(port, 115200, timeout=0.2)
    time.sleep(0.5)
    stop = threading.Event()
    threading.Thread(target=reader, args=(ser, stop), daemon=True).start()

    angle = 90.0
    marks = {}

    def send():
        ser.write(('P %.1f\n' % angle).encode())

    print(__doc__.split('Keys')[1])
    send()
    try:
        while True:
            label = '  '.join(f'{k}={v:.0f}' for k, v in marks.items())
            sys.stdout.write(f'\r  servo angle {angle:5.1f} deg   [{label}]        ')
            sys.stdout.flush()
            k = getch()
            if k in ('q', '\x03'):
                break
            step = {'a': -1, 'd': 1, 'A': -5, 'D': 5}.get(k)
            if step:
                angle = min(180.0, max(0.0, angle + step))
                send()
            elif k == 'g':
                print()
                try:
                    angle = min(180.0, max(0.0, float(ask('  servo angle: '))))
                    send()
                except ValueError:
                    pass
            elif k == 'c':
                marks['centre'] = angle
            elif k == 'l':
                marks['left'] = angle
            elif k == 'r':
                marks['right'] = angle
    finally:
        print()
        c = marks.get('centre', 90.0)
        ser.write(('P %.1f\n' % c).encode())     # leave the wheels straight
        time.sleep(0.2)
        stop.set()
        ser.close()

    if not {'centre', 'left', 'right'} <= set(marks):
        print('Need centre, left and right marks - nothing computed.')
        return
    c, L, R = marks['centre'], marks['left'], marks['right']
    if (L - c) * (R - c) >= 0:
        print('Left and right are on the same side of centre - please redo.')
        return
    servo_dir = 1 if L > c else -1
    left_span, right_span = abs(L - c), abs(R - c)
    usable = min(left_span, right_span) - 2.0      # 2 deg away from the end stops

    print(f'\nservo: centre {c:.0f}, left {L:.0f} ({left_span:.0f} deg), right {R:.0f} ({right_span:.0f} deg)')
    ratio = 1.0
    w = ask('Wheel angle you measured at full LEFT, in degrees (Enter to skip): ')
    if w:
        try:
            ratio = left_span / float(w)
        except (ValueError, ZeroDivisionError):
            pass
    print('\nPut these in src/agv_follower/config/agv.yaml:')
    print(f'    servo_center_deg: {c:.1f}')
    print(f'    servo_dir: {servo_dir:.1f}')
    print(f'    servo_per_wheel_deg: {ratio:.3f}' + ('' if w else '   # unchanged - set by the turning-circle test'))
    print(f'    max_steer_deg: {usable / ratio:.1f}   # (shared section, wheel degrees)')
    if abs(left_span - right_span) > 5:
        print(f'\nNote: left and right travel differ by {abs(left_span - right_span):.0f} deg - '
              'the servo horn may be one spline off centre.')


if __name__ == '__main__':
    main()
