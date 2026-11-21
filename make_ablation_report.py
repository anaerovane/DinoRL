#!/usr/bin/env python3
"""Aggregate ablation eval summaries into one comparison table.

Reads results/exp/<variant>/eval_summary.csv plus each variant's training best
score, groups by ablation axis, and writes results/exp/ablation_report.csv
together with a markdown table on stdout.
"""
import csv
from pathlib import Path

EXP = Path(__file__).resolve().parent / 'results' / 'exp'

# variant -> (axis, anchor)
GROUPS = {
    'ppo_vision': ('anchor', '-'),
    'state_dqn': ('anchor', '-'),
    'abl_ppo_noent': ('ppo_component', 'ppo_vision'),
    'abl_ppo_nogae': ('ppo_component', 'ppo_vision'),
    'abl_ppo_nonorm': ('ppo_component', 'ppo_vision'),
    'abl_ppo_clip01': ('ppo_component', 'ppo_vision'),
    'abl_state_notype': ('state_feature', 'state_dqn'),
    'abl_state_notrex': ('state_feature', 'state_dqn'),
    'abl_state_noheight': ('state_feature', 'state_dqn'),
    'abl_state_distonly': ('state_feature', 'state_dqn'),
    'abl_obs_stack1': ('observation', 'ppo_vision'),
    'abl_obs_stack2': ('observation', 'ppo_vision'),
    'abl_obs_res64': ('observation', 'ppo_vision'),
    'abl_rew_nostep': ('reward', 'ppo_vision'),
    'abl_rew_nopenalty': ('reward', 'ppo_vision'),
    'abl_rew_sparse': ('reward', 'ppo_vision'),
}

AXIS_ORDER = ['anchor', 'ppo_component', 'state_feature', 'observation', 'reward']


def read_summary(name):
    p = EXP / name / 'eval_summary.csv'
    if not p.exists():
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def training_best(name):
    log = EXP / name / 'training_log.csv'
    if not log.exists():
        return ''
    best = None
    with open(log) as f:
        for row in csv.DictReader(f):
            try:
                s = float(row['score'])
            except (KeyError, ValueError):
                continue
            best = s if best is None or s > best else best
    return '' if best is None else round(best, 1)


def main():
    rows = []
    anchors = {}
    for name, (axis, anchor) in GROUPS.items():
        summ = read_summary(name)
        rec = dict(name=name, axis=axis, anchor=anchor,
                   train_best=training_best(name))
        if summ is None:
            rec.update(eval_mean='', median='', std='', min='', max='', n='')
            rows.append(rec)
            continue
        rec.update(eval_mean=float(summ['mean']), median=float(summ['median']),
                   std=float(summ['std']), min=float(summ['min']),
                   max=float(summ['max']), n=int(summ['n']))
        rows.append(rec)
        if axis == 'anchor':
            anchors[name] = rec

    for rec in rows:
        a = anchors.get(rec['anchor'])
        if a and isinstance(rec['eval_mean'], float) and isinstance(a['eval_mean'], float):
            rec['delta_vs_anchor'] = round(rec['eval_mean'] - a['eval_mean'], 2)
        else:
            rec['delta_vs_anchor'] = ''

    fields = ['name', 'axis', 'anchor', 'n', 'eval_mean', 'delta_vs_anchor',
              'median', 'std', 'min', 'max', 'train_best']
    out = EXP / 'ablation_report.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in rows:
            w.writerow({k: rec.get(k, '') for k in fields})
    print(f'wrote {out}')

    print(f'\n| {"variant":22s} | {"axis":14s} | {"anchor":11s} | {"n":>3s} | '
          f'{"greedy mean":>11s} | {"Δ vs anchor":>11s} | {"median":>7s} | '
          f'{"std":>6s} | {"min":>6s} | {"max":>6s} | {"train best":>10s} |')
    print('|' + '-' * 24 + '|' + '-' * 16 + '|' + '-' * 13 + '|' + '-' * 5 +
          '|' + '-' * 13 + '|' + '-' * 13 + '|' + '-' * 9 + '|' + '-' * 8 +
          '|' + '-' * 8 + '|' + '-' * 8 + '|' + '-' * 12 + '|')
    for axis in AXIS_ORDER:
        for rec in rows:
            if rec['axis'] != axis:
                continue

            def fmt(v, spec='.1f', dash='-'):
                return format(v, spec) if isinstance(v, float) else dash

            print(f"| {rec['name']:22s} | {rec['axis']:14s} | {rec['anchor']:11s} | "
                  f"{str(rec.get('n', '')):>3s} | "
                  f"{fmt(rec['eval_mean']):>11s} | "
                  f"{(format(rec['delta_vs_anchor'], '+.1f') if isinstance(rec['delta_vs_anchor'], float) else '-'):>11s} | "
                  f"{fmt(rec.get('median')):>7s} | "
                  f"{fmt(rec.get('std')):>6s} | "
                  f"{fmt(rec.get('min')):>6s} | "
                  f"{fmt(rec.get('max')):>6s} | "
                  f"{str(rec['train_best']):>10s} |")


if __name__ == '__main__':
    main()
