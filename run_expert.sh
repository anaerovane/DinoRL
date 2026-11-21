#!/bin/bash
cd /root/DinoRL
python3 -c "
import sys, os
sys.path.insert(0, '/root/DinoRL/gamescaling')
os.environ['GSD_BASE'] = '/root/DinoRL/gamescaling'
os.environ['GSD_DINORL'] = '/root/DinoRL'
os.environ['GSD_RESULTS'] = '/root/DinoRL/results'
os.chdir('/root/DinoRL')

import json
from pathlib import Path

DINORL = Path('/root/DinoRL')
nb = json.loads((DINORL / 'dinorl.ipynb').read_text())

def find(pred):
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        if pred(src):
            return src
    raise SystemExit('not found')

setup = find(lambda s: 'class LocalGameHandler' in s)
classes = find(lambda s: 'class GitHubDinoNet' in s)
env_cell = find(lambda s: 'class LocalGitHubGame' in s)
setup = setup.replace('FIREFOX_BINARY = None', 'FIREFOX_BINARY = \"/root/tools/firefox/firefox\"')

ns = {}
exec(compile(setup, 'setup', 'exec'), ns)
exec(compile(classes, 'classes', 'exec'), ns)
exec(compile(env_cell, 'env', 'exec'), ns)

from collect_expert import main
import argparse
sys.argv = ['collect_expert.py', '--episodes', '8', '--seed-offset', '40000', '--max-frames', '20000']
import collect_expert
collect_expert.load_dinorl_env = lambda: ns
main()
"
