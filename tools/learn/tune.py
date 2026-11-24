#!/usr/bin/env python3
"""tune.py - learn the follower's parameters in the fast simulator (cross-entropy method).

Each candidate parameter set is scored by its average reward over the same set of random rooms
(reward: reach the ball and stop at 30 cm fast, never collide, keep up with moving balls).
The best candidates shape the next generation. At the end the winner is validated against the
current config on rooms it never trained on.

    python3 tune.py                 # full run (~15-20 min on 10 cores)
    python3 tune.py --quick         # small smoke run

Results: tools/learn/results.json (progress) and tools/learn/best_params.yaml (winner).
"""
import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

import fastsim as fs

HERE = Path(__file__).resolve().parent

# name: (low, high, is_int)
SPACE = {
    'w_block':              (0.5, 8.0, False),
    'w_clear':              (0.0, 2.0, False),
    'w_smooth':             (0.0, 1.0, False),
    'clear_ref':            (0.10, 0.60, False),
    'safety_margin':        (0.03, 0.12, False),
    'horizon':              (1.0, 3.0, False),
    'n_arcs':               (9, 25, True),
    'speed_gain':           (0.3, 2.0, False),
    'max_speed':            (0.20, 0.45, False),
    'search_speed':         (0.10, 0.40, False),
    'search_roam_m':        (0.5, 2.5, False),
    'search_circle_deg':    (180.0, 450.0, False),
    'memory_timeout':       (1.0, 5.0, False),
    'memory_timeout_still': (3.0, 15.0, False),
    'lead_time':            (0.0, 1.0, False),
    'ball_keepout':         (0.05, 0.30, False),
    'ball_ignore_radius':   (0.15, 0.40, False),
    'search_circle_room':   (0.5, 2.2, False),
    'search_look_m':        (1.0, 3.5, False),
    'reposition_factor':    (0.0, 1.1, False),
    'keepout_fresh_s':      (0.5, 4.0, False),
    'escape_len':           (0.4, 1.4, False),
    'escape_check_m':       (0.1, 0.6, False),
    'w_trap':               (0.0, 8.0, False),
}
NAMES = list(SPACE)


def to_params(u):
    """unit vector [0,1]^n -> parameter dict"""
    out = {}
    for k, ui in zip(NAMES, u):
        lo, hi, is_int = SPACE[k]
        v = lo + float(np.clip(ui, 0, 1)) * (hi - lo)
        out[k] = int(round(v)) if is_int else round(v, 4)
    return out


def from_params(p):
    return np.array([(p[k] - SPACE[k][0]) / (SPACE[k][1] - SPACE[k][0]) for k in NAMES])


_BASE = None


def _init():
    global _BASE
    _BASE = fs.load_params()


def _episode(job):
    overrides, world_seed = job
    world = fs.random_world(np.random.default_rng(world_seed))
    params = dict(_BASE)
    params.update(overrides)
    try:
        return fs.run_episode(world, params, seed=world_seed)
    except Exception as e:                                   # a broken candidate scores badly
        return dict(reward=-100.0, collided=False, success=False, moving=False, error=str(e))


def evaluate(pool, candidates, seeds):
    jobs = [(c, s) for c in candidates for s in seeds]
    res = pool.map(_episode, jobs, chunksize=4)
    n = len(seeds)
    return [res[i * n:(i + 1) * n] for i in range(len(candidates))]


def summary(results):
    r = np.array([x['reward'] for x in results])
    return dict(reward=float(r.mean()),
                success=float(np.mean([x['success'] for x in results])),
                follow=float(np.mean([x.get('follow', 0.0) for x in results])),
                collisions=int(sum(x['collided'] for x in results)),
                ball_bumps=int(sum(x.get('fault_bumps', 0) > 0 for x in results)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--iters', type=int, default=12)
    ap.add_argument('--pop', type=int, default=24)
    ap.add_argument('--elite', type=int, default=6)
    ap.add_argument('--episodes', type=int, default=40)
    ap.add_argument('--validate', type=int, default=200)
    a = ap.parse_args()
    if a.quick:
        a.iters, a.pop, a.elite, a.episodes, a.validate = 2, 6, 2, 6, 12

    base = fs.load_params()
    current = {k: base[k] for k in NAMES}
    mean = from_params(current)
    std = np.full(len(NAMES), 0.25)
    train_seeds = list(range(1000, 1000 + a.episodes))
    rng = np.random.default_rng(0)
    log = dict(space={k: SPACE[k][:2] for k in NAMES}, iterations=[])
    best_u, best_score = mean.copy(), -1e9
    t0 = time.time()

    with Pool(initializer=_init) as pool:
        for it in range(a.iters):
            cands_u = [mean.copy(), best_u.copy()] + \
                [np.clip(mean + std * rng.standard_normal(len(NAMES)), 0, 1) for _ in range(a.pop - 2)]
            cands = [to_params(u) for u in cands_u]
            results = evaluate(pool, cands, train_seeds)
            scores = [summary(r) for r in results]
            order = np.argsort([-s['reward'] for s in scores])
            elite = [cands_u[i] for i in order[:a.elite]]
            mean = np.mean(elite, axis=0)
            std = np.maximum(np.std(elite, axis=0), 0.03)
            top = scores[order[0]]
            if top['reward'] > best_score:
                best_score, best_u = top['reward'], cands_u[order[0]].copy()
            log['iterations'].append(dict(iter=it, best=top, best_params=cands[order[0]],
                                          mean_reward=float(np.mean([s['reward'] for s in scores]))))
            json.dump(log, open(HERE / 'results.json', 'w'), indent=1)
            print(f'iter {it:2d}  best reward {top["reward"]:6.1f}  following {top["follow"]:.0%} of the time  '
                  f'collisions {top["collisions"]}  bumps {top["ball_bumps"]}  '
                  f'pop mean {np.mean([s["reward"] for s in scores]):6.1f}  ({time.time() - t0:.0f} s)', flush=True)

        # ---- validation on unseen rooms ----
        val_seeds = list(range(50000, 50000 + a.validate))
        best = to_params(best_u)
        res_cur, res_best = evaluate(pool, [current, best], val_seeds)
        s_cur, s_best = summary(res_cur), summary(res_best)
    log['validation'] = dict(current=s_cur, learned=s_best, learned_params=best, rooms=a.validate)
    json.dump(log, open(HERE / 'results.json', 'w'), indent=1)
    yaml.safe_dump(best, open(HERE / 'best_params.yaml', 'w'), sort_keys=False)
    print(f'\nVALIDATION on {a.validate} unseen rooms')
    for name, s in (('current', s_cur), ('learned', s_best)):
        print(f'  {name:8s} reward {s["reward"]:6.1f}  following {s["follow"]:.0%} of the time  '
              f'collisions {s["collisions"]}  ball bumps {s["ball_bumps"]}')
    print('learned params:', best)
    print(f'DONE in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
