"""
run_all.py - Run all paper experiments and save outputs to output/.

Executes each experiment script in sequence, printing to stdout and
saving the output to output/<name>.txt.

Usage:
    python run_all.py                             # all experiments, all datasets
    python run_all.py coloring                    # single experiment, all datasets
    python run_all.py titanic                     # all experiments, titanic only
    python run_all.py titanic coloring            # single experiment, titanic only
    python run_all.py --skip uci_comparison       # all except uci_comparison
"""

import sys
import os
import subprocess

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, 'output')

EXPERIMENTS = [
    ('coloring',            'show_coloring.py'),
    ('pruning_phases',      'show_pruning_phases.py'),
    ('inert_analysis',      'bench_inert_analysis.py'),
    ('pruning_comparison',  'bench_pruning_comparison.py'),
    ('uci_comparison',      'bench_uci_comparison.py'),
    ('coloring_speed',      'bench_coloring_speed.py'),
    ('pruning_speed',       'bench_pruning_speed.py'),
    ('fidelity',            'bench_fidelity.py'),
    ('fidelity_uci',        'bench_fidelity_uci.py'),
]


def run_experiment(name, script, ds_args=None):
    print(f'\n{"="*60}')
    print(f'  {script}')
    print(f'{"="*60}\n')

    out_path = os.path.join(OUT_DIR, f'{name}.txt')
    cmd = [sys.executable, os.path.join(REPO, script)] + (ds_args or [])
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    output = proc.stdout
    if proc.stderr:
        output += '\n--- stderr ---\n' + proc.stderr
    print(output)
    with open(out_path, 'w') as f:
        f.write(output)
    print(f'  -> saved to output/{name}.txt')
    return proc.returncode


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    exp_names = {n for n, _ in EXPERIMENTS}
    args = sys.argv[1:]

    skip = set()
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == '--skip':
            i += 1
            while i < len(args) and not args[i].startswith('-'):
                skip.add(args[i])
                i += 1
        else:
            filtered_args.append(args[i])
            i += 1

    exp_args = [a for a in filtered_args if a in exp_names]
    ds_args = [a for a in filtered_args if a not in exp_names]

    experiments = [(n, s) for n, s in EXPERIMENTS if n in exp_args] if exp_args else EXPERIMENTS
    experiments = [(n, s) for n, s in experiments if n not in skip]

    failed = []
    for name, script in experiments:
        rc = run_experiment(name, script, ds_args)
        if rc != 0:
            failed.append(name)

    print(f'\n{"="*60}')
    if failed:
        print(f'  FAILED: {failed}')
    else:
        print(f'  All experiments completed. Results in output/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
