#!/usr/bin/env python3
"""PPO trainer (vision obs, 8 parallel envs, synchronous rollouts).

Usage: python train_ppo.py --name ppo_vision
"""
import argparse, random, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from dino_env import (DEVICE, N_ACTIONS, TARGET_SCORE, GameEnv, PPOVisionNet,
                      batched_to_tensor, append_log, log_status, shutdown_asset_server)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='ppo_vision')
    p.add_argument('--frames', type=int, default=2_230_000)
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--hours', type=float, default=10.0)
    p.add_argument('--rollout-rounds', type=int, default=64)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatch', type=int, default=256)
    p.add_argument('--lr', type=float, default=2.5e-4)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--entropy-coef', type=float, default=0.01)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--no-adv-norm', action='store_true')
    p.add_argument('--frame-stack', type=int, default=4)
    p.add_argument('--obs-res', type=int, default=80)
    p.add_argument('--step-reward', type=float, default=0.1)
    p.add_argument('--crash-penalty', type=float, default=1.0)
    p.add_argument('--early-penalty', type=float, default=5.0)
    return p.parse_args()


def compute_gae(rewards, values, discounts, dones, last_values, lam):
    """Per-env GAE. Arrays shaped (T, N). last_values: (N,)."""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + discounts[t] * next_v * next_nonterminal - values[t]
        last_gae = delta + discounts[t] * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def main():
    args = parse_args()
    out = Path('results/exp') / args.name
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    status = out / 'status.txt'
    log_path = out / 'training_log.csv'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_status(status, f'start ppo frames_budget={args.frames} n_envs={args.n_envs} '
                       f'rollout_rounds={args.rollout_rounds} minibatch={args.minibatch} '
                       f'epochs={args.epochs} frame_stack={args.frame_stack} '
                       f'obs_res={args.obs_res} adv_norm={not args.no_adv_norm} '
                       f'step_reward={args.step_reward} crash_penalty={args.crash_penalty} '
                       f'early_penalty={args.early_penalty} device={DEVICE}')

    net = PPOVisionNet(in_channels=args.frame_stack).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    log_status(status, 'spawning environments...')
    envs = [GameEnv(i, mode='vision', seed_base=args.seed, frame_stack=args.frame_stack,
                    obs_res=args.obs_res, step_reward=args.step_reward,
                    crash_penalty=args.crash_penalty, early_penalty=args.early_penalty)
            for i in range(args.n_envs)]
    n_envs = len(envs)
    log_status(status, f'{n_envs} envs ready')

    states = {e.env_id: e.reset() for e in envs}
    log_status(status, 'all envs reset')
    pool = ThreadPoolExecutor(max_workers=n_envs)

    R = args.rollout_rounds
    started = time.monotonic()
    deadline = started + args.hours * 3600
    total_frames, updates, best_score, episode_counter = 0, 0, -1.0, 0

    def step_env(env, action):
        sid = states[env.env_id]
        nxt, reward, done, score, discount, advanced = env.step(action)
        return env, sid, action, reward, nxt, done, score, discount, advanced

    try:
        while total_frames < args.frames and time.monotonic() < deadline:
            # ---- rollout ----
            ob_s = np.zeros((R, n_envs, args.frame_stack, args.obs_res, args.obs_res),
                            dtype=np.uint8)
            acts = np.zeros((R, n_envs), dtype=np.int64)
            rews = np.zeros((R, n_envs), dtype=np.float32)
            discs = np.zeros((R, n_envs), dtype=np.float32)
            dones = np.zeros((R, n_envs), dtype=np.float32)
            logps = np.zeros((R, n_envs), dtype=np.float32)
            vals = np.zeros((R, n_envs), dtype=np.float32)

            for t in range(R):
                sid_list = [states[e.env_id] for e in envs]
                ob_s[t] = np.stack(sid_list)
                x = batched_to_tensor(sid_list, 'vision')
                with torch.inference_mode():
                    logits, v = net(x)
                    dist = Categorical(logits=logits)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                acts[t] = a.cpu().numpy()
                logps[t] = lp.cpu().numpy()
                vals[t] = v.cpu().numpy()

                actions_map = {env.env_id: int(acts[t, i]) for i, env in enumerate(envs)}
                futures = [pool.submit(step_env, env, actions_map[env.env_id]) for env in envs]
                results = [f.result() for f in futures]

                for i, (env, sid, action, reward, nxt, done, score, discount, advanced) in enumerate(results):
                    rews[t, i] = reward
                    discs[t, i] = discount
                    dones[t, i] = float(done)
                    total_frames += advanced
                    if done:
                        episode_counter += 1
                        is_best = score > best_score
                        if is_best:
                            best_score = score
                            torch.save({'model': net.state_dict(), 'score': score,
                                        'episode': episode_counter, 'algo': 'ppo',
                                        'frame_stack': args.frame_stack,
                                        'obs_res': args.obs_res},
                                       out / 'best.pt')
                        append_log(log_path, dict(
                            episode=episode_counter, seed=env.seed, score=round(score, 2),
                            frames=env.game_frames, total_frames=total_frames,
                            epsilon=0.0, mean_loss='', updates=updates, terminal='collision',
                            episode_seconds=round(time.monotonic() - started, 2),
                            elapsed_seconds=round(time.monotonic() - started, 1),
                            env=env.env_id))
                        if episode_counter % 10 == 0 or is_best or score >= TARGET_SCORE:
                            log_status(status, f'ep={episode_counter} score={score:.0f} '
                                               f'best={best_score:.0f} frames={total_frames} '
                                               f'upd={updates} sp={total_frames / max(1e-9, time.monotonic() - started):.0f}f/s')
                        if best_score >= TARGET_SCORE:
                            torch.save({'model': net.state_dict(), 'score': best_score,
                                        'updates': updates, 'algo': 'ppo',
                                        'frame_stack': args.frame_stack,
                                        'obs_res': args.obs_res}, out / 'final.pt')
                            log_status(status, f'TARGET REACHED best={best_score:.0f}')
                            return
                        states[env.env_id] = env.reset()
                    else:
                        states[env.env_id] = nxt

            # bootstrap values
            sid_list = [states[e.env_id] for e in envs]
            x = batched_to_tensor(sid_list, 'vision')
            with torch.inference_mode():
                _, v_last = net(x)
            last_values = v_last.cpu().numpy().astype(np.float32)
            last_values = last_values * (1.0 - dones[-1])  # zero if terminal at t=R-1

            adv, ret = compute_gae(rews, vals, discs, dones, last_values, args.gae_lambda)
            adv_t = torch.from_numpy(adv.reshape(-1)).to(DEVICE)
            ret_t = torch.from_numpy(ret.reshape(-1)).to(DEVICE)
            ob_t = torch.from_numpy(
                ob_s.reshape(-1, args.frame_stack, args.obs_res, args.obs_res)
            ).float().to(DEVICE) / 255.0
            act_t = torch.from_numpy(acts.reshape(-1)).to(DEVICE)
            logp_t = torch.from_numpy(logps.reshape(-1)).to(DEVICE)

            if not args.no_adv_norm:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            n_samples = ob_t.shape[0]
            idx = np.arange(n_samples)

            policy_loss_acc, value_loss_acc, entropy_acc, n_mb = 0.0, 0.0, 0.0, 0
            for _ in range(args.epochs):
                np.random.shuffle(idx)
                for s in range(0, n_samples, args.minibatch):
                    mb = idx[s:s + args.minibatch]
                    logits, v = net(ob_t[mb])
                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(act_t[mb])
                    ratio = (new_logp - logp_t[mb]).exp()
                    surr1 = ratio * adv_t[mb]
                    surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = ((v - ret_t[mb]) ** 2).mean()
                    entropy = dist.entropy().mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                    optimizer.step()
                    policy_loss_acc += policy_loss.item()
                    value_loss_acc += value_loss.item()
                    entropy_acc += entropy.item()
                    n_mb += 1

            updates += 1
            if updates % 10 == 0:
                torch.save({'model': net.state_dict(), 'updates': updates,
                            'total_frames': total_frames, 'best': best_score, 'algo': 'ppo'},
                           ckpt_dir / 'learner_latest.pt')
                log_status(status, f'update={updates} frames={total_frames} '
                                   f'best={best_score:.0f} pi={policy_loss_acc / n_mb:.4f} '
                                   f'v={value_loss_acc / n_mb:.4f} ent={entropy_acc / n_mb:.4f} '
                                   f'sp={total_frames / max(1e-9, time.monotonic() - started):.0f} f/s')

        torch.save({'model': net.state_dict(), 'score': best_score,
                    'updates': updates, 'total_frames': total_frames, 'algo': 'ppo',
                    'frame_stack': args.frame_stack, 'obs_res': args.obs_res},
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
