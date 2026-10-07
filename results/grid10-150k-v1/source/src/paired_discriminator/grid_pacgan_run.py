"""Launch only the fifteen agreed PacGAN2 grid trajectories."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import contextlib
import json
from .grid_pacgan import ROOT, run, snapshot


def job(args):
    config, method, seed, path = args
    with Path(str(path) + '.log').open('w') as log, contextlib.redirect_stdout(log):
        run(config, method, seed, path)
    return str(path)


def main():
    jobs = []
    outputs = [ROOT / f'results/grid{side}-pacgan2-v1' for side in (3, 5, 7)]
    for output in outputs:
        if output.exists():
            raise FileExistsError(f'{output}: preserve previous runs; use a new output version')
    for side, output in zip((3, 5, 7), outputs):
        budget = 150 if side == 7 else 50
        config = json.loads((ROOT / f'configs/grid{side}-{budget}k.json').read_text())
        snapshot(output, config)
        (output / 'workload.json').write_text(json.dumps({
            'method': 'pacgan2', 'pack_size': 2,
            'd_real_points': 256, 'd_fake_points': 256,
            'd_real_packs': 128, 'd_fake_packs': 128,
            'g_fake_points': 256, 'g_fake_packs': 128,
            'd_updates_per_step': 1, 'g_updates_per_step': 1,
            'g_real_references_used': 0,
            'note': 'Real-G and slot RNG draws retained but unused to preserve baseline streams.'
        }, indent=2) + '\n')
        for seed in config['seeds']:
            jobs.append((config, 'pacgan2', seed, output / f'pacgan2_seed{seed}'))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(job, args) for args in jobs]):
            print('COMPLETED', future.result(), flush=True)


if __name__ == '__main__':
    main()
