#!/usr/bin/env python3
"""Cross-seed aggregation for wave2: group eval summaries by method, bar+error plot."""
import csv, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path('results/exp')


def read_csv(p):
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def method_of(name):
    return re.sub(r'_s[0-9]+$', '', name)


def main():
    # wave1 + wave2 eval summaries
    rows = []
    for d in sorted(EXP.iterdir()):
        if not d.is_dir():
            continue
        for r in read_csv(d / 'eval_summary.csv'):
            r['dir'] = d.name
            r['method'] = method_of(d.name)
            rows.append(r)
    if not rows:
        print('no eval summaries found')
        return

    methods = sorted({r['method'] for r in rows})
    table = []
    for m in methods:
        ms = [r for r in rows if r['method'] == m]
        means = [float(r['mean']) for r in ms]
        maxs = [float(r['max']) for r in ms]
        table.append(dict(method=m, runs=len(ms),
                          mean_of_means=round(float(np.mean(means)), 2),
                          std_across_runs=round(float(np.std(means)), 2) if len(means) > 1 else 0.0,
                          best=max(maxs),
                          runs_detail='; '.join(f"{r['dir']}:mean={r['mean']},max={r['max']}"
                                                for r in ms)))
    out_csv = EXP / 'wave2_table.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for r in table:
            w.writerow(r)
    print('wrote', out_csv)
    for r in table:
        print(r)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(methods))
    m_of_m = [t['mean_of_means'] for t in table]
    s_ar = [t['std_across_runs'] for t in table]
    best = [t['best'] for t in table]
    ax.bar(x, m_of_m, yerr=s_ar, capsize=5, color='steelblue', alpha=0.85,
           label='eval mean (±std across training seeds)')
    ax.scatter(x, best, color='crimson', zorder=3, label='best episode')
    ax.axhline(2000, color='green', ls=':', lw=1.2, label='target 2000')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylabel('score (50 fixed eval seeds, greedy)')
    ax.set_title('DinoRL wave1+wave2 comparison')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_png = EXP / 'wave2_compare.png'
    plt.savefig(out_png, dpi=130)
    print('wrote', out_png)


if __name__ == '__main__':
    main()
