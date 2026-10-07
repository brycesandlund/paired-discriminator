"""Launch only the fifteen agreed two-output discriminator grid trajectories."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import contextlib
import json
from .grid_dual_slot import ROOT, run, snapshot


def job(args):
    config, method, seed, path = args
    with Path(str(path) + '.log').open('w') as log, contextlib.redirect_stdout(log):
        run(config, method, seed, path)
    return str(path)


def main():
    jobs = []
    outputs = [ROOT / f'results/grid{side}-dual-slot-v1' for side in (3, 5, 7)]
    for output in outputs:
        if output.exists():
            raise FileExistsError(f'{output}: preserve previous runs; use a new output version')
    for side, output in zip((3, 5, 7), outputs):
        budget = 150 if side == 7 else 50
        config = json.loads((ROOT / f'configs/grid{side}-{budget}k.json').read_text())
        snapshot(output, config)
        (output / 'workload.json').write_text(json.dumps({
            'method': 'dual_slot', 'outputs_per_pair': 2,
            'd_real_points': 256, 'd_fake_points': 256,
            'd_pair_counts': {'RR': 64, 'RF': 64, 'FR': 64, 'FF': 64},
            'd_bce_decisions': 512, 'd_loss_reduction': 'mean over all slots',
            'g_fake_points': 256, 'g_real_references_used': 128,
            'g_pair_counts': {'FF': 64, 'FR': 64, 'RF': 64},
            'g_bce_decisions': 256, 'g_loss_reduction': 'mean over generated slots only',
            'd_updates_per_step': 1, 'g_updates_per_step': 1,
            'note': 'Draw 256 real-G samples but use 128; retain unused slot draws to preserve baseline RNG streams.'
        }, indent=2) + '\n')
        for seed in config['seeds']:
            jobs.append((config, 'dual_slot', seed, output / f'dual_slot_seed{seed}'))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(job, args) for args in jobs]):
            print('COMPLETED', future.result(), flush=True)


if __name__ == '__main__':
    main()
