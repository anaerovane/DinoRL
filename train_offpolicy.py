#!/usr/bin/env python3
"""Off-policy trainer: vanilla DQN, Double DQN, or C51 (vision|state obs).

Usage: python train_offpolicy.py --algo ddqn --obs vision --name ddqn_vision
"""
import argparse, copy, random, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dino_env import (DEVICE, GAMMA, N_ACTIONS, TARGET_SCORE, FEAT_DIM, GameEnv,
                      ReplayBuffer, make_net, batched_to_tensor, append_log,
                      log_status, shutdown_asset_server, FEAT_MASKS)

V_MIN, V_MAX, N_ATOMS = -20.0, 20.0, 51
ATOMS = torch.linspace(V_MIN, V_MAX, N_ATOMS, device=DEVICE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=['dqn', 'ddqn', 'c51'], required=True)
    p.add_argument('--obs', choices=['vision', 'state'], default='vision')
    p.add_argument('--name', default=None)
    p.add_argument('--frames', type=int, default=2_230_000)
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--hours', type=float, default=10.0, help='hard wall-clock cap')
    p.add_argument('--feat-mask', choices=sorted(FEAT_MASKS), default=None)
    p.add_argument('--frame-stack', type=int, default=4)
    p.add_argument('--obs-res', type=int, default=80)
    p.add_argument('--step-reward', type=float, default=0.1)
    p.add_argument('--crash-penalty', type=float, default=1.0)
    p.add_argument('--early-penalty', type=float, default=5.0)
    return p.parse_args()


def c51_project(rewards, discounts, terminals, next_probs):
    """Project the Bellman target distribution onto the fixed support.
    rewards/discounts/terminals: (B,), next_probs: (B, N_ATOMS). Returns (B, N_ATOMS)."""
    b = rewards[:, None] + (discounts * (1 - terminals))[:, None] * ATOMS[None, :]
    b = b.clamp(V_MIN, V_MAX)
    idx = (b - V_MIN) / ((V_MAX - V_MIN) / (N_ATOMS - 1))
    l = idx.floor().long().clamp(0, N_ATOMS - 1)
    u = idx.ceil().long().clamp(0, N_ATOMS - 1)
    eq = l == u
    l = torch.where(eq & (u > 0), l - 1, l)
    u = torch.where(eq & (u < N_ATOMS - 1), u + 1, u)
    w_l = next_probs * (u.float() - idx)
    w_u = next_probs * (idx - l.float())
    proj = torch.zeros_like(next_probs)
    proj.scatter_add_(1, l, w_l)
    proj.scatter_add_(1, u, w_u)
    return proj


def main():
    args = parse_args()
    name = args.name or f'{args.algo}_{args.obs}'
    out = Path('results/exp') / name
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    status = out / 'status.txt'
    log_path = out / 'training_log.csv'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    is_c51 = args.algo == 'c51'
    out_dim = N_ACTIONS * N_ATOMS if is_c51 else N_ACTIONS
    obs_shape = ((args.frame_stack, args.obs_res, args.obs_res) if args.obs == 'vision'
                 else (FEAT_DIM,))
    obs_dtype = np.uint8 if args.obs == 'vision' else np.float32

    lr, batch_size = 3e-4, 256
    updates_per_round, replay_cap, warmup = 8, 50_000, 1000
    eps_start, eps_end, eps_decay = 0.25, 0.02, 1_000_000
    target_tau, grad_clip = 0.002, 10.0

    learner = make_net(args.algo, args.obs, out_dim, in_channels=args.frame_stack).to(DEVICE)
    target_net = copy.deepcopy(learner).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    replay = ReplayBuffer(replay_cap, obs_shape, obs_dtype)

    log_status(status, f'start algo={args.algo} obs={args.obs} name={name} '
                       f'frames_budget={args.frames} n_envs={args.n_envs} '
                       f'feat_mask={args.feat_mask} frame_stack={args.frame_stack} '
                       f'obs_res={args.obs_res} step_reward={args.step_reward} '
                       f'crash_penalty={args.crash_penalty} '
                       f'early_penalty={args.early_penalty} device={DEVICE}')

    log_status(status, 'spawning environments...')
    envs = [GameEnv(i, mode=args.obs, seed_base=args.seed, frame_stack=args.frame_stack,
                    obs_res=args.obs_res, step_reward=args.step_reward,
                    crash_penalty=args.crash_penalty, early_penalty=args.early_penalty,
                    feat_mask=args.feat_mask)
            for i in range(args.n_envs)]
    log_status(status, f'{len(envs)} envs ready')

    states = {}
    for env in envs:
        states[env.env_id] = env.reset()
    log_status(status, 'all envs reset')

    pool = ThreadPoolExecutor(max_workers=args.n_envs)

    def step_env(env):
        sid = states[env.env_id]
        nxt, reward, done, score, discount, advanced = env.step(actions_map[env.env_id])
        return env, sid, actions_map[env.env_id], reward, nxt, done, score, discount, advanced

    started = time.monotonic()
    deadline = started + args.hours * 3600
    total_frames, updates, best_score, episode_counter, round_idx = 0, 0, -1.0, 0, 0

    try:
        while total_frames < args.frames and time.monotonic() < deadline:
            round_idx += 1
            # batched GPU action selection (epsilon-greedy)
            sid_list = [states[e.env_id] for e in envs]
            x = batched_to_tensor(sid_list, args.obs)
            eps = max(eps_end, eps_start - (eps_start - eps_end) * total_frames / eps_decay)
            with torch.inference_mode():
                if is_c51:
                    logits = learner(x).view(-1, N_ACTIONS, N_ATOMS)
                    q = (logits.softmax(-1) * ATOMS[None, None, :]).sum(-1)
                else:
                    q = learner(x)
                greedy = q.argmax(1).cpu().numpy()
            actions_map = {}
            for env, g in zip(envs, greedy):
                actions_map[env.env_id] = random.randrange(N_ACTIONS) if random.random() < eps else int(g)

            futures = [pool.submit(step_env, env) for env in envs]
            results = [f.result() for f in futures]

            for env, sid, action, reward, nxt, done, score, discount, advanced in results:
                replay.add(sid, action, reward, nxt, done, discount)
                total_frames += advanced
                if done:
                    episode_counter += 1
                    is_best = score > best_score
                    if is_best:
                        best_score = score
                        torch.save({'model': learner.state_dict(), 'score': score,
                                    'episode': episode_counter, 'algo': args.algo,
                                    'obs': args.obs, 'atoms': N_ATOMS if is_c51 else 0,
                                    'frame_stack': args.frame_stack,
                                    'obs_res': args.obs_res, 'feat_mask': args.feat_mask},
                                   out / 'best.pt')
                    append_log(log_path, dict(
                        episode=episode_counter, seed=env.seed, score=round(score, 2),
                        frames=env.game_frames, total_frames=total_frames,
                        epsilon=round(eps, 4), mean_loss='', updates=updates,
                        terminal='collision',
                        episode_seconds=round(time.monotonic() - started, 2),
                        elapsed_seconds=round(time.monotonic() - started, 1),
                        env=env.env_id))
                    if episode_counter % 10 == 0 or is_best or score >= TARGET_SCORE:
                        log_status(status, f'ep={episode_counter} score={score:.0f} '
                                           f'best={best_score:.0f} frames={total_frames} '
                                           f'upd={updates} sp={total_frames / max(1e-9, time.monotonic() - started):.0f}f/s')
                    if best_score >= TARGET_SCORE:
                        torch.save({'model': learner.state_dict(), 'score': best_score,
                                    'updates': updates, 'algo': args.algo, 'obs': args.obs,
                                    'atoms': N_ATOMS if is_c51 else 0,
                                    'frame_stack': args.frame_stack,
                                    'obs_res': args.obs_res, 'feat_mask': args.feat_mask},
                                   out / 'final.pt')
                        log_status(status, f'TARGET REACHED best={best_score:.0f}')
                        return
                    states[env.env_id] = env.reset()
                else:
                    states[env.env_id] = nxt

            if replay.size >= max(warmup, batch_size):
                for _ in range(updates_per_round):
                    obs_b, act_b, rew_b, next_b, done_b, disc_b = replay.sample(batch_size)
                    st = batched_to_tensor(list(obs_b), args.obs)
                    ns = batched_to_tensor(list(next_b), args.obs)
                    actions = torch.from_numpy(act_b).to(DEVICE, non_blocking=True)
                    rewards = torch.from_numpy(rew_b).to(DEVICE, non_blocking=True)
                    terminals = torch.from_numpy(done_b).to(DEVICE, non_blocking=True)
                    discounts = torch.from_numpy(disc_b).to(DEVICE, non_blocking=True)

                    if is_c51:
                        with torch.amp.autocast('cuda'):
                            logits = learner(st).view(-1, N_ACTIONS, N_ATOMS)
                            log_p = F.log_softmax(logits, dim=-1)
                            qa = log_p.gather(1, actions[:, None, None]
                                              .expand(-1, 1, N_ATOMS)).squeeze(1)
                            with torch.no_grad():
                                next_logits = learner(ns).view(-1, N_ACTIONS, N_ATOMS)
                                next_q = (next_logits.softmax(-1) * ATOMS[None, None, :]).sum(-1)
                                next_a = next_q.argmax(1)
                                tgt_logits = target_net(ns).view(-1, N_ACTIONS, N_ATOMS)
                                next_p = tgt_logits.softmax(-1).gather(
                                    1, next_a[:, None, None].expand(-1, 1, N_ATOMS)).squeeze(1)
                                proj = c51_project(rewards, discounts, terminals, next_p)
                                proj = proj.clamp_min(1e-8)
                            loss = -(proj * qa).sum(-1).mean()
                    else:
                        with torch.amp.autocast('cuda'):
                            q = learner(st).gather(1, actions[:, None]).squeeze(1)
                            with torch.no_grad():
                                if args.algo == 'ddqn':
                                    next_actions = learner(ns).argmax(1)
                                    expected = rewards + discounts * (1 - terminals) * \
                                        target_net(ns).gather(1, next_actions[:, None]).squeeze(1)
                                else:  # vanilla DQN: target-net max
                                    expected = rewards + discounts * (1 - terminals) * \
                                        target_net(ns).max(1).values
                            loss = F.smooth_l1_loss(q, expected)

                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite loss')
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(learner.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    with torch.no_grad():
                        for tp, lp in zip(target_net.parameters(), learner.parameters()):
                            tp.lerp_(lp, target_tau)
                    updates += 1

            if round_idx % 200 == 0:
                torch.save({'model': learner.state_dict(), 'updates': updates,
                            'total_frames': total_frames, 'best': best_score,
                            'algo': args.algo, 'obs': args.obs,
                            'atoms': N_ATOMS if is_c51 else 0,
                            'frame_stack': args.frame_stack,
                            'obs_res': args.obs_res, 'feat_mask': args.feat_mask},
                           ckpt_dir / 'learner_latest.pt')
                log_status(status, f'round={round_idx} frames={total_frames} upd={updates} '
                                   f'best={best_score:.0f} replay={replay.size} '
                                   f'speed={total_frames / max(1e-9, time.monotonic() - started):.0f} f/s')

        torch.save({'model': learner.state_dict(), 'score': best_score,
                    'updates': updates, 'total_frames': total_frames,
                    'algo': args.algo, 'obs': args.obs,
                    'atoms': N_ATOMS if is_c51 else 0,
                    'frame_stack': args.frame_stack,
                    'obs_res': args.obs_res, 'feat_mask': args.feat_mask},
                   out / 'final.pt')
        log_status(status, f'finished best={best_score:.0f} frames={total_frames} upd={updates}')
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
