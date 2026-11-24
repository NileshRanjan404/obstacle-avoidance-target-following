#!/usr/bin/env python3
"""export_traces.py - record simulated episodes (current config) as JSON for the replay page."""
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

import fastsim as fs


def run(seed):
    P = fs.load_params()
    w = fs.random_world(np.random.default_rng(seed))
    r = fs.run_episode(w, P, seed=seed, record=True)
    tr = r['trace'][::2]                      # 10 Hz is plenty for drawing
    return dict(
        seed=seed, success=r['success'], moving=r['moving'], collided=r['collided'],
        reward=round(r['reward'], 1), time=round(r['time'], 1), gap=round(r['gap'], 3),
        room=[round(w['W'], 2), round(w['H'], 2)],
        boxes=[[round(v, 3) for v in b] for b in w['boxes'][4:]],
        frames=[[round(t, 2), round(x, 3), round(y, 3), round(yaw, 3), round(bx, 3), round(by, 3), s]
                for (t, x, y, yaw, bx, by, s) in tr],
        n_boxes=len(w['boxes']) - 4,
        searched=any(f[6] == 'SEARCH' for f in tr))


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'traces.json'
    with Pool() as p:
        res = p.map(run, range(3000, 3040))
    pick = []
    # successful runs with obstacles, a search that found the ball, moving balls, and one failure
    pick += [r for r in res if r['success'] and r['n_boxes'] >= 2][:2]
    pick += [r for r in res if r['success'] and r['searched'] and r not in pick][:1]
    pick += [r for r in res if r['moving'] and not r['collided']][:2]
    pick += [r for r in res if not r['moving'] and not r['success']][:1]
    json.dump(dict(params={'wheelbase': 0.14, 'front': 0.205, 'rear': 0.045, 'width': 0.28,
                           'stop': 0.30, 'ball_d': 0.07}, episodes=pick),
              open(out, 'w'), separators=(',', ':'))
    for r in pick:
        print(r['seed'], 'success' if r['success'] else ('moving' if r['moving'] else 'fail'),
              r['n_boxes'], 'boxes', r['time'], 's', len(r['frames']), 'frames')
