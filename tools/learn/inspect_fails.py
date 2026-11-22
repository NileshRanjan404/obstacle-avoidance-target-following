#!/usr/bin/env python3
"""inspect_fails.py - show what happened in failed episodes (state time shares, end situation)."""
import collections
import math
import sys
from multiprocessing import Pool

import numpy as np

import fastsim as fs


def run(seed):
    P = fs.load_params()
    w = fs.random_world(np.random.default_rng(seed))
    r = fs.run_episode(w, P, seed=seed, record=True)
    tr = r['trace']
    st = collections.Counter(s for *_, s in tr)
    n = max(len(tr), 1)
    t, x, y, yaw, bx, by, s_end = tr[-1]
    rel = math.degrees(math.atan2(by - y, bx - x) - yaw)
    rel = (rel + 180) % 360 - 180
    return seed, r, ', '.join(f'{k}:{100 * v // n}' for k, v in st.most_common(4)), s_end, rel


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    with Pool() as p:
        res = p.map(run, range(1000, 1000 + n))
    for seed, r, states, s_end, rel in res:
        bad = (not r['moving'] and not r['success']) or r['min_gap'] < 0.18
        if bad:
            print(f'{seed} {"MOVING" if r["moving"] else "still "} gap {r["gap"]:5.2f} min {r["min_gap"]:5.2f} '
                  f'end {s_end:11s} ball at {rel:+5.0f} deg | {states}')
