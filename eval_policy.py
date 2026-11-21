#!/usr/bin/env python3
"""Unified evaluation for all baselines on 50 fixed seeds.

Policies: random | rule | model (checkpoint).
Usage:
  python eval_policy.py --policy random --name random
  python eval_policy.py --policy rule --name rule
  python eval_policy.py --policy model --name ddqn --ckpt results/dino_trained_best.pt --algo ddqn --obs vision
  python eval_policy.py --policy model --name c51 --ckpt results/exp/c51_vision/best.pt --algo c51 --obs vision
"""
import argparse, random, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from dino_env import (DEVICE, N_ACTIONS, FEAT_DIM, GameEnv, make_net,
                      batched_to_tensor, append_log, log_status, shutdown_asset_server,
                      MAX_GAME_FRAMES)

N_EVAL = 50
EVAL_SEED_BASE = 900_000


class RulePolicy:
    """Hand-crafted jump heuristic over engine features ({noop, jump} only).

    Game runs with ACCELERATION=0 (constant speed 6), so only cacti spawn
    (pterodactyls need speed>=8.5). Tight multi-cactus clusters are cleared
    with one jump: airborne window is ~200px at speed 6.
    """

    def __init__(self, k_single=12, k_cluster=7, cluster_gap=60, horizon=32):
        self.k_single = k_single
        self.k_cluster = k_cluster
        self.cluster_gap = cluster_gap
        self.horizon = horizon

    def act(self, feat):
        if feat[2] > 0.5:  # airborne: cannot jump again
            return 0
        speed = float(feat[4]) * 7.0 + 6.0
        upcoming = []
        for i in range(3):
            s = feat[5 + 7 * i: 5 + 7 * (i + 1)]
            dist = float(s[0]) * 300.0
            if dist < -60:
                continue
            upcoming.append((dist, float(s[1]) * 80.0))
        if not upcoming:
            return 0
        d0, w0 = upcoming[0]
        if d0 > speed * self.horizon:
            return 0
        span_end = d0 + w0
        for d, w in upcoming[1:]:
            if d - span_end < self.cluster_gap:
                span_end = d + w
            else:
                break
        span = span_end - d0
        k = self.k_single if span <= 45 else self.k_cluster
        return 1 if d0 < speed * k else 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', choices=['random', 'rule', 'model'], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--algo', default='ddqn', choices=['dqn', 'ddqn', 'c51', 'ppo', 'bc'])
    p.add_argument('--obs', default='vision', choices=['vision', 'state'])
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--episodes', type=int, default=N_EVAL)
    p.add_argument('--seed-base', type=int, default=EVAL_SEED_BASE)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path('results/exp') / args.name
    out.mkdir(parents=True, exist_ok=True)
    status = out / 'eval_status.txt'
    log_path = out / 'eval_episodes.csv'

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    net = None
    rule = None
    meta_stack, meta_res, meta_mask = 4, 80, None
    if args.policy == 'model':
        assert args.ckpt, '--ckpt required for model policy'
        ckpt = torch.load(args.ckpt, map_location=DEVICE)
        meta_algo = ckpt.get('algo', args.algo)
        meta_obs = ckpt.get('obs', args.obs)
        meta_atoms = ckpt.get('atoms', 0)
        meta_stack = int(ckpt.get('frame_stack', 4))
        meta_res = int(ckpt.get('obs_res', 80))
        meta_mask = ckpt.get('feat_mask', None)
        if meta_algo == 'ppo':
            from dino_env import PPOVisionNet
            net = PPOVisionNet(in_channels=meta_stack).to(DEVICE)
            net.load_state_dict(ckpt['model'])
        else:
            out_dim = 2 * 51 if (meta_atoms or 0) > 0 else N_ACTIONS
            net = make_net(meta_algo, meta_obs, out_dim, in_channels=meta_stack).to(DEVICE)
            net.load_state_dict(ckpt['model'])
        net.eval()
        args.obs = meta_obs
        args.algo = meta_algo
        log_status(status, f'loaded {args.ckpt} algo={meta_algo} obs={meta_obs} '
                           f'atoms={meta_atoms} frame_stack={meta_stack} obs_res={meta_res} '
                           f'feat_mask={meta_mask} train_best={ckpt.get("score")}')
    elif args.policy == 'rule':
        rule = RulePolicy()
        args.obs = 'state'
    else:
        args.obs = 'state'  # random walks through the cheap state env

    log_status(status, f'eval policy={args.policy} name={args.name} obs={args.obs} '
                       f'episodes={args.episodes} n_envs={args.n_envs}')

    envs = [GameEnv(i, mode=args.obs, seed_base=args.seed_base,
                    frame_stack=meta_stack, obs_res=meta_res, feat_mask=meta_mask)
            for i in range(args.n_envs)]
    pool = ThreadPoolExecutor(max_workers=args.n_envs)

    seed_queue = [args.seed_base + i for i in range(args.episodes)]
    next_seed = [0]  # closure counter
    results_rows = []

    def take_seed():
        i = next_seed[0]
        next_seed[0] += 1
        return seed_queue[i] if i < len(seed_queue) else None

    states = {}
    for env in envs:
        s = take_seed()
        if s is None:
            break
        states[env.env_id] = env.reset(seed=s)

    def batch_actions(env_list):
        if args.policy == 'random':
            return [random.randrange(N_ACTIONS) for _ in env_list]
        if args.policy == 'rule':
            return [rule.act(states[e.env_id]) for e in env_list]
        x = batched_to_tensor([states[e.env_id] for e in env_list], args.obs)
        with torch.inference_mode():
            outp = net(x)
            if args.algo == 'ppo':
                logits, _ = outp
                return logits.argmax(1).cpu().tolist()
            if args.algo == 'c51':
                logits = outp.view(-1, N_ACTIONS, 51)
                atoms = torch.linspace(-20, 20, 51, device=DEVICE)
                q = (logits.softmax(-1) * atoms[None, None, :]).sum(-1)
                return q.argmax(1).cpu().tolist()
            return outp.argmax(1).cpu().tolist()

    def step_env(env, action):
        nxt, reward, done, score, discount, advanced = env.step(action)
        return env, nxt, done, score, advanced

    started = time.monotonic()
    done_count = 0
    try:
        while done_count < args.episodes:
            actions = batch_actions(envs)
            futures = [pool.submit(step_env, env, int(a))
                       for env, a in zip(envs, actions)]
            for f in futures:
                env, nxt, done, score, advanced = f.result()
                if env.game_frames >= MAX_GAME_FRAMES:
                    done = True  # truncation cap (~score 3000); perfect policies can't loop
                if done:
                    done_count += 1
                    results_rows.append(dict(episode=done_count, seed=env.seed,
                                             score=round(score, 2),
                                             frames=env.game_frames,
                                             policy=args.policy))
                    append_log(log_path, results_rows[-1])
                    if done_count % 10 == 0:
                        log_status(status, f'done {done_count}/{args.episodes} '
                                           f'last={score:.0f}')
                    s = take_seed()
                    if s is not None:
                        states[env.env_id] = env.reset(seed=s)
                    else:
                        states[env.env_id] = None
                else:
                    states[env.env_id] = nxt
            envs = [e for e in envs if states.get(e.env_id) is not None]
            if not envs:
                break

        scores = [r['score'] for r in results_rows]
        arr = np.array(scores, dtype=np.float64)
        summary = dict(
            policy=args.policy, name=args.name, n=len(arr),
            mean=round(arr.mean(), 2), median=round(float(np.median(arr)), 2),
            std=round(arr.std(), 2), min=round(arr.min(), 2), max=round(arr.max(), 2),
            p25=round(float(np.percentile(arr, 25)), 2),
            p75=round(float(np.percentile(arr, 75)), 2),
            seconds=round(time.monotonic() - started, 1))
        append_log(out / 'eval_summary.csv', summary)
        log_status(status, f'EVAL DONE {summary}')
    except Exception:
        log_status(status, 'ERROR:\n' + traceback.format_exc())
        raise
    finally:
        pool.shutdown(wait=False)
        for e in envs:
            e.close()
        shutdown_asset_server()
        log_status(status, 'cleaned up')


if __name__ == '__main__':
    main()
