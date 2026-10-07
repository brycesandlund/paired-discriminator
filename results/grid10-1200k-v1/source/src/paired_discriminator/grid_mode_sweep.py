"""Run the fixed-spacing Gaussian-grid sweep using the original trainer."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import contextlib
import json
from .grid_experiment import ROOT, run, snapshot


def job(args):
    config, method, seed, path = args
    with Path(str(path) + '.log').open('w') as log, contextlib.redirect_stdout(log):
        run(config, method, seed, path)
    return str(path)


def main():
    jobs = []
    for side in [3, 5, 7]:
        config = json.loads((ROOT / f'configs/grid{side}-50k.json').read_text())
        output = ROOT / f'results/grid{side}-50k-v1'
        if output.exists():
            raise FileExistsError(f'{output}: preserve previous runs; use a new output version')
        snapshot(output, config)
        for seed in config['seeds']:
            methods = ['vanilla', 'rsgan', 'paired']
            for method in methods if seed % 2 == 0 else methods[::-1]:
                jobs.append((config, method, seed, output / f'{method}_seed{seed}'))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for path in pool.map(job, jobs):
            print('COMPLETED', path, flush=True)


if __name__ == '__main__':
    main()
