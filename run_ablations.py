#!/usr/bin/env python3
"""Ablation-wave orchestrator: trains up to MAX_SLOTS variants concurrently,
then greedily evaluates each on the 50 fixed eval seeds.

A variant is 'already trained' iff results/exp/<name>/final.pt exists; anything
else has its output dir wiped first so stale best.pt / appended logs can't
pollute the rerun.

Usage: python run_ablations.py            # run everything not yet done
       python run_ablations.py --dry-run  # just print the plan
"""
import argparse, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / 'results/exp'
LOG = EXP / 'ablation_orchestrator.log'
MAX_SLOTS = 2
N_ENVS = 4
EVAL_ENVS = 4
COMMON = ['--frames', '2230000', '--n-envs', str(N_ENVS), '--hours', '24']

_log_lock = threading.Lock()


def say(msg):
    line = f'[{time.strftime("%F %T")}] {msg}'
    with _log_lock:
        print(line, flush=True)
        with open(LOG, 'a') as f:
            f.write(line + '\n')


# (name, train_script or None for eval-only, extra train args, anchor)
VARIANTS = [
    # ---- anchors: greedy-eval the wave-1 winners first ----
    ('ppo_vision', None, [], 'ppo_vision'),
    ('state_dqn', None, [], 'state_dqn'),
    # ---- PPO component ablations (anchor: ppo_vision) ----
    ('abl_ppo_noent', 'train_ppo.py',
     ['--name', 'abl_ppo_noent', '--entropy-coef', '0.0'], 'ppo_vision'),
    ('abl_ppo_nogae', 'train_ppo.py',
     ['--name', 'abl_ppo_nogae', '--gae-lambda', '1.0'], 'ppo_vision'),
    ('abl_ppo_nonorm', 'train_ppo.py',
     ['--name', 'abl_ppo_nonorm', '--no-adv-norm'], 'ppo_vision'),
    ('abl_ppo_clip01', 'train_ppo.py',
     ['--name', 'abl_ppo_clip01', '--clip', '0.1'], 'ppo_vision'),
    # ---- state-feature ablations (anchor: state_dqn, 26-dim engine state) ----
    ('abl_state_notype', 'train_offpolicy.py',
     ['--name', 'abl_state_notype', '--algo', 'ddqn', '--obs', 'state',
      '--feat-mask', 'notype'], 'state_dqn'),
    ('abl_state_notrex', 'train_offpolicy.py',
     ['--name', 'abl_state_notrex', '--algo', 'ddqn', '--obs', 'state',
      '--feat-mask', 'notrex'], 'state_dqn'),
    ('abl_state_noheight', 'train_offpolicy.py',
     ['--name', 'abl_state_noheight', '--algo', 'ddqn', '--obs', 'state',
      '--feat-mask', 'noheight'], 'state_dqn'),
    ('abl_state_distonly', 'train_offpolicy.py',
     ['--name', 'abl_state_distonly', '--algo', 'ddqn', '--obs', 'state',
      '--feat-mask', 'distonly'], 'state_dqn'),
    # ---- observation / input ablations (anchor: ppo_vision) ----
    ('abl_obs_stack1', 'train_ppo.py',
     ['--name', 'abl_obs_stack1', '--frame-stack', '1'], 'ppo_vision'),
    ('abl_obs_stack2', 'train_ppo.py',
     ['--name', 'abl_obs_stack2', '--frame-stack', '2'], 'ppo_vision'),
    ('abl_obs_res64', 'train_ppo.py',
     ['--name', 'abl_obs_res64', '--obs-res', '64'], 'ppo_vision'),
    # ---- reward-shaping ablations (anchor: ppo_vision) ----
    ('abl_rew_nostep', 'train_ppo.py',
     ['--name', 'abl_rew_nostep', '--step-reward', '0'], 'ppo_vision'),
    ('abl_rew_nopenalty', 'train_ppo.py',
     ['--name', 'abl_rew_nopenalty', '--early-penalty', '0'], 'ppo_vision'),
    ('abl_rew_sparse', 'train_ppo.py',
     ['--name', 'abl_rew_sparse', '--step-reward', '0',
      '--early-penalty', '0'], 'ppo_vision'),
]


def trained(name):
    return (EXP / name / 'final.pt').exists()


def evaluated(name):
    return (EXP / name / 'eval_summary.csv').exists()


def train(name, script, extra):
    out = EXP / name
    if trained(name):
        say(f'{name}: training already complete (final.pt present)')
        return True
    if out.exists():
        say(f'{name}: wiping stale dir {out}')
        shutil.rmtree(out)
    cmd = [sys.executable, script] + extra + COMMON
    say(f'{name}: TRAIN start  ({" ".join(cmd[1:])})')
    t0 = time.monotonic()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.monotonic() - t0
    ok = trained(name)
    say(f'{name}: TRAIN {"done" if ok else "FAILED"} rc={r.returncode} '
        f'wall={dt / 60:.1f}min')
    return ok


def evaluate(name):
    if evaluated(name):
        say(f'{name}: eval already complete')
        return True
    ckpt = EXP / name / 'best.pt'
    if not ckpt.exists():
        say(f'{name}: no best.pt - cannot evaluate')
        return False
    cmd = [sys.executable, 'eval_policy.py', '--policy', 'model',
           '--name', name, '--ckpt', str(ckpt.relative_to(ROOT)),
           '--n-envs', str(EVAL_ENVS), '--episodes', '50']
    say(f'{name}: EVAL start ({" ".join(cmd[1:])})')
    t0 = time.monotonic()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.monotonic() - t0
    ok = evaluated(name)
    say(f'{name}: EVAL {"done" if ok else "FAILED"} rc={r.returncode} '
        f'wall={dt / 60:.1f}min')
    return ok


def run_one(item):
    name, script, extra, anchor = item
    if evaluated(name):
        say(f'{name}: already evaluated - skip')
        return name, True
    if script is not None:
        ok = train(name, script, extra)
        if not ok:
            say(f'{name}: retrying training once after wipe')
            ok = train(name, script, extra)
        if not ok:
            say(f'{name}: giving up on training')
            return name, False
    ok = evaluate(name)
    return name, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None,
                    help='comma-separated variant names to run (default: all)')
    ap.add_argument('--slots', type=int, default=MAX_SLOTS,
                    help='max concurrent train+eval tasks')
    args = ap.parse_args()
    EXP.mkdir(parents=True, exist_ok=True)

    variants = VARIANTS
    if args.only:
        wanted = {s.strip() for s in args.only.split(',') if s.strip()}
        unknown = wanted - {v[0] for v in VARIANTS}
        if unknown:
            raise SystemExit(f'unknown variant(s): {sorted(unknown)}')
        variants = [v for v in VARIANTS if v[0] in wanted]

    say(f'===== ablation orchestrator start pid={__import__("os").getpid()} '
        f'slots={args.slots} n_envs={N_ENVS} eval_envs={EVAL_ENVS} '
        f'variants={len(variants)}/{len(VARIANTS)} =====')
    for name, script, extra, anchor in variants:
        say(f'  plan {name:22s} anchor={anchor:11s} '
            f'{"eval-only" if script is None else " ".join([script] + extra)}')
    if args.dry_run:
        say('dry-run only - exiting')
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.slots) as pool:
        for name, ok in pool.map(run_one, variants):
            results.append((name, ok))

    say('===== ablation orchestrator finished =====')
    for name, ok in results:
        say(f'  {"OK " if ok else "FAIL"} {name}')
    subprocess.run([sys.executable, 'make_ablation_report.py'], cwd=ROOT)


if __name__ == '__main__':
    main()
