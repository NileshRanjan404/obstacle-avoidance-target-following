#!/usr/bin/env python3
"""evaluate.py - score the follower in the fast simulator over many random rooms.

    python3 evaluate.py                       # current config, 60 rooms
    python3 evaluate.py --rooms 200
    python3 evaluate.py --params best_params.yaml   # with learned overrides
"""
import argparse
import collections
from multiprocessing import Pool

import numpy as np
import yaml

import fastsim as fs

_OVR = {}


def _init(ovr):
    global _OVR
    _OVR = ovr


def run(seed):
    P = fs.load_params(_OVR)
    w = fs.random_world(np.random.default_rng(seed))
    r = fs.run_episode(w, P, seed=seed, record=True)
    st = collections.Counter(s for *_, s in r['trace'])
    n = max(len(r['trace']), 1)
    return dict(seed=seed, kind=w['kind'], success=r['success'], moving=r['moving'], collided=r['collided'],
                follow=r.get('follow', 0.0), catch_t=r.get('catch_t'),
                bump=r.get('fault_bumps', 0) > 0, reward=r['reward'], blocked=st['BLOCKED'] / n,
                never_seen=not any(s in ('FOLLOW', 'MEMORY') for *_, s in r['trace']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rooms', type=int, default=60)
    ap.add_argument('--start', type=int, default=1000)
    ap.add_argument('--params', default=None)
    a = ap.parse_args()
    ovr = yaml.safe_load(open(a.params)) if a.params else {}
    with Pool(initializer=_init, initargs=(ovr,)) as p:
        res = p.map(run, range(a.start, a.start + a.rooms))
    still = [r for r in res if not r['moving']]
    fails = [r for r in still if not r['success']]
    print(f'rooms {len(res)}  ({len(still)} still ball, {len(res) - len(still)} moving)')
    print(f'  still-ball success   {sum(r["success"] for r in still)}/{len(still)}')
    print(f'  collisions           {sum(r["collided"] for r in res)}')
    print(f'  ball bumps           {sum(r["bump"] for r in res)}')
    print(f'  mean reward          {np.mean([r["reward"] for r in res]):.1f}')
    print(f'  time following close (15-60 cm)  {100 * np.mean([r["follow"] for r in res]):.0f} %')
    caught = [r["catch_t"] for r in res if r["catch_t"] is not None]
    print(f'  caught up with the ball in {len(caught)}/{len(res)} rooms, median after {np.median(caught) if caught else float("nan"):.0f} s')
    print(f'  mean time BLOCKED    {100 * np.mean([r["blocked"] for r in res]):.0f} %')
    print('  by room type:            rooms  following  caught  robot-fault bumps  collisions')
    for k in ('easy', 'block', 'gap', 'clutter'):
        g = [r for r in res if r['kind'] == k]
        if g:
            print(f'    {k:8s}               {len(g):5d}  {100 * np.mean([r["follow"] for r in g]):8.0f} %'
                  f'  {sum(r["catch_t"] is not None for r in g):3d}/{len(g):<3d}'
                  f'  {sum(r["bump"] for r in g):17d}  {sum(r["collided"] for r in g):10d}')
    print(f'  failed still rooms where the ball was never seen: {sum(r["never_seen"] for r in fails)}/{len(fails)}')


if __name__ == '__main__':
    main()
