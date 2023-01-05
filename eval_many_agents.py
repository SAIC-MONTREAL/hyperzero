"""
Script for evaluating and rolling out several RL agents.
The rollout dataset is saved for training RL approximators
using supervised learning.
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import subprocess
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', type=str, default="results")
parser.add_argument('--domain_task', type=str, default='cheetah_run')
parser.add_argument('--step_to_load', type=int, default=1000000)
parser.add_argument('--n_episodes', type=int, default=10)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--rollout_dir', type=str, default='rollout_data')
parser.add_argument('--video_dir', type=str, default='video_logs')
args = parser.parse_args()


root_dir = Path(args.rootdir)
paths = sorted(root_dir.glob(f'**/*{args.domain_task}*/**/step_*{args.step_to_load}'))

seeds_list = []

for p in paths:
    workdir = p.parents[1]
    seed = int(p.parents[2].name.split('_')[4])
    command = [
        'python',
        'eval.py',
        '--workdir',
        str(workdir),
        '--step_to_load',
        str(args.step_to_load),
        '--n_episodes',
        str(args.n_episodes),
        '--rollout_dir',
        str(args.rollout_dir),
        '--video_dir',
        str(args.video_dir),
        '--rl_regressor_workdir',
        str(None),
        '--eval_mode',
        'sl_data',
    ]
    if args.vis:
        command += ['--vis']

    print(f"Running {command}")
    process = subprocess.run(command, capture_output=True)
    print(f"Returncode of the process: {process.returncode}")
    if process.returncode == 0:
        seeds_list.append(seed)
    else:
        print(process.stderr)
