#!/usr/bin/env python3
"""Aggregate eval summaries + training curves into comparison artifacts."""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path('results/exp')

# name -> (training_log path, display label)
TRAIN_LOGS = {
    'ddqn (fast, reuse)': Path('results/training_log.csv'),
    'dqn': EXP / 'dqn_vision' / 'training_log.csv',
    'c51': EXP / 'c51_vision' / 'training_log.csv',
    'ppo': EXP / 'ppo_vision' / 'training_log.csv',
    'state_dqn': EXP / 'state_dqn' / 'training_log.csv',
}
EVAL_DIRS = ['random', 'rule', 'ddqn', 'dqn', 'c51', 'ppo', 'state_dqn']


def read_csv(p):
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def moving_avg(x, w=50):
    if len(x) < w:
        return np.asarray(x, dtype=float)
    return np.convolve(np.asarray(x, dtype=float), np.ones(w) / w, mode='valid')


def main():
    # ---- table ----
    table = []
    for name in EVAL_DIRS:
        rows = read_csv(EXP / name / 'eval_summary.csv')
        if rows:
            table.append(rows[-1])
    if table:
        keys = list(table[0].keys())
        with open(EXP / 'compare_table.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in table:
                w.writerow(r)
        print('wrote', EXP / 'compare_table.csv')

    # ---- curves ----
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                  gridspec_kw={'width_ratios': [3, 2]})
    colors = plt.cm.tab10.colors
    for ci, (label, p) in enumerate(TRAIN_LOGS.items()):
        rows = read_csv(p)
        if not rows:
            print('skip (no log):', label)
            continue
        ep = np.array([int(r['episode']) for r in rows])
        score = np.array([float(r['score']) for r in rows])
        frames = np.array([int(r['total_frames']) for r in rows]) / 1e6
        ma = moving_avg(score, 50)
        ax.plot(frames[49:], ma, lw=1.8, color=colors[ci % 10], label=f'{label} (final ma {ma[-1]:.0f})')
        best = np.maximum.accumulate(score)
        ax.plot(frames, best, lw=1, ls='--', color=colors[ci % 10], alpha=0.5)

    ax.axhline(2000, color='green', ls=':', lw=1.2, label='target 2000')
    ax.set_xlabel('training frames (M)')
    ax.set_ylabel('episode score (50-ep moving avg, dashed = best-so-far)')
    ax.set_title('Training curves (vision vs state)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # eval bars
    names, means, stds, maxs = [], [], [], []
    for name in EVAL_DIRS:
        rows = read_csv(EXP / name / 'eval_summary.csv')
        if rows:
            r = rows[-1]
            names.append(name)
            means.append(float(r['mean']))
            stds.append(float(r['std']))
            maxs.append(float(r['max']))
    if names:
        x = np.arange(len(names))
        ax2.bar(x, means, yerr=stds, capsize=4, color='steelblue', alpha=0.8,
                label='mean ± std (50 eval seeds)')
        ax2.scatter(x, maxs, color='crimson', zorder=3, label='max')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=30, ha='right')
        ax2.set_ylabel('eval score')
        ax2.set_title('Greedy eval on 50 fixed seeds')
        ax2.grid(alpha=0.3, axis='y')
        ax2.legend(fontsize=9)

    plt.tight_layout()
    out = EXP / 'compare_curves.png'
    plt.savefig(out, dpi=130)
    print('wrote', out)


if __name__ == '__main__':
    main()
