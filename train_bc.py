#!/usr/bin/env python3
"""Behavioral cloning: privileged rule teacher -> pure-vision student.

The rule policy sees engine state and is near-optimal in this constant-speed
cactus-only environment. We collect (pixel stack, rule action) pairs and train
a CNN classifier so the student reaches high scores from pixels alone.

Usage: python train_bc.py --name bc_vision --episodes 48 --n-envs 4
"""
import argparse, random, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dino_env import (DEVICE, N_ACTIONS, GameEnv, GitHubDinoNet, append_log,
                      log_status, shutdown_asset_server, MAX_GAME_FRAMES)
from eval_policy import RulePolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='bc_vision')
    p.add_argument('--episodes', type=int, default=48)
    p.add_argument('--n-envs', type=int, default=4)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--max-frames-ep', type=int, default=10_000)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path('results/exp') / args.name
    out.mkdir(parents=True, exist_ok=True)
    status = out / 'status.txt'
    log_path = out / 'training_log.csv'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_status(status, f'bc start teacher=rule episodes={args.episodes} '
                       f'n_envs={args.n_envs} max_frames_ep={args.max_frames_ep}')

    rule = RulePolicy()
    envs = [GameEnv(i, mode='vision', seed_base=args.seed) for i in range(args.n_envs)]
    pool = ThreadPoolExecutor(max_workers=args.n_envs)
    states = {e.env_id: e.reset() for e in envs}

    obs_list, act_list = [], []

    def step_env(env, action):
        sid = states[env.env_id]
        nxt, reward, done, score, discount, advanced = env.step(action)
        return env, sid, action, nxt, done, score

    started = time.monotonic()
    done_count = 0
    try:
        # ---- collect ----
        while done_count < args.episodes:
            futures = []
            for env in envs:
                feats = env.feats()
                action = rule.act(feats)
                futures.append(pool.submit(step_env, env, action))
            for f in futures:
                env, sid, action, nxt, done, score = f.result()
                obs_list.append(sid)
                act_list.append(action)
                if env.game_frames >= args.max_frames_ep:
                    done = True
                if done:
                    done_count += 1
                    append_log(log_path, dict(episode=done_count, seed=env.seed,
                                              score=round(score, 2),
                                              frames=env.game_frames,
                                              samples=len(act_list)))
                    if done_count % 4 == 0:
                        log_status(status, f'collect {done_count}/{args.episodes} '
                                           f'last={score:.0f} samples={len(act_list)}')
                    if done_count < args.episodes:
                        states[env.env_id] = env.reset()
                    else:
                        states[env.env_id] = None
                else:
                    states[env.env_id] = nxt
            envs = [e for e in envs if states.get(e.env_id) is not None]
            if not envs:
                break

        X = np.stack(obs_list)
        y = np.asarray(act_list, dtype=np.int64)
        counts = np.bincount(y, minlength=N_ACTIONS).astype(np.float64)
        counts[counts == 0] = 1.0
        class_w = torch.tensor(len(y) / (N_ACTIONS * counts), dtype=torch.float32,
                               device=DEVICE)
        log_status(status, f'collected {len(y)} samples '
                           f'action_dist={np.bincount(y, minlength=N_ACTIONS).tolist()} '
                           f'class_w={class_w.tolist()} '
                           f'in {time.monotonic() - started:.0f}s')

        # ---- train ----
        net = GitHubDinoNet(out_dim=N_ACTIONS).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        n = X.shape[0]
        idx = np.arange(n)
        for epoch in range(args.epochs):
            np.random.shuffle(idx)
            tot_loss, tot_correct, tot = 0.0, 0, 0
            for s in range(0, n, args.batch):
                mb = idx[s:s + args.batch]
                xb = torch.from_numpy(X[mb]).float().to(DEVICE) / 255.0
                yb = torch.from_numpy(y[mb]).to(DEVICE)
                logits = net(xb)
                loss = F.cross_entropy(logits, yb, weight=class_w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot_loss += loss.item() * len(mb)
                tot_correct += (logits.argmax(1) == yb).sum().item()
                tot += len(mb)
            acc = tot_correct / max(1, tot)
            log_status(status, f'epoch {epoch + 1}/{args.epochs} '
                               f'loss={tot_loss / max(1, tot):.4f} acc={acc:.4f}')
            torch.save({'model': net.state_dict(), 'algo': 'bc', 'obs': 'vision',
                        'atoms': 0, 'train_acc': acc, 'samples': int(n),
                        'score': 0}, out / 'best.pt')

        torch.save({'model': net.state_dict(), 'algo': 'bc', 'obs': 'vision',
                    'atoms': 0, 'samples': int(n)}, out / 'final.pt')
        log_status(status, 'bc training finished')
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
