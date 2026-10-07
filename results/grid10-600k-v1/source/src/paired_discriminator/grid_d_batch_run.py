"""Run D512/G256 for three joint methods across the fixed-spacing grids."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import contextlib
import json
from .grid_d_batch import ROOT, run, snapshot


def job(args):
    config, method, seed, path = args
    with Path(str(path) + '.log').open('w') as log, contextlib.redirect_stdout(log):
        run(config, method, seed, path)
    return str(path)


def main():
    jobs = []
    outputs = [ROOT / f'results/grid{side}-d512-v1' for side in (3,5,7)]
    for output in outputs:
        if output.exists():
            raise FileExistsError(f'{output}: preserve existing runs')
    for side, output in zip((3,5,7), outputs):
        config = json.loads((ROOT / f'configs/grid{side}-{150 if side==7 else 50}k.json').read_text())
        config['discriminator_batch_size'] = 512
        snapshot(output, config)
        (output/'workload.json').write_text(json.dumps({
            'd_real_points':512, 'd_fake_points':512, 'g_fake_points':256,
            'd_updates_per_step':1, 'g_updates_per_step':1,
            'paired_d_pairs':512, 'pacgan2_d_packs':{'RR':256,'FF':256},
            'dual_slot_d_pairs':{'RR':128,'RF':128,'FR':128,'FF':128},
            'g_workloads':'Unchanged from D256 baselines for each method',
            'workers':8, 'threads_per_worker':1,
            'rng_note':'D slots use seed+80000. Original slot draws retained so G slots, real-G, noise-G and eval streams remain unchanged.',
            'extra_checkpoints':'5k/25k/75k support exact equal-D-sample comparisons to baseline 10k/50k/150k.'
        },indent=2)+'\n')
        for seed in config['seeds']:
            methods=['paired','pacgan2','dual_slot']
            for method in methods if seed%2==0 else methods[::-1]:
                jobs.append((config,method,seed,output/f'{method}_seed{seed}'))
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(job,args) for args in jobs]):
            print('COMPLETED',future.result(),flush=True)


if __name__ == '__main__':
    main()
