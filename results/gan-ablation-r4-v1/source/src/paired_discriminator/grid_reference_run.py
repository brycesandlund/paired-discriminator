"""Launch D-only nearby/underrepresented-reference grid interventions."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import contextlib
import json
from .grid_reference import ROOT, run, snapshot

METHODS=['paired_near','paired_deficit','vanilla_deficit']


def job(args):
    config,method,seed,path=args
    with Path(str(path)+'.log').open('w') as log,contextlib.redirect_stdout(log):
        run(config,method,seed,path)
    return str(path)


def main():
    audit=json.loads((ROOT/'results/grid-reference-role-audit-v1/audit.json').read_text())
    assert all(.47<r['heldout_accuracy']<.53 for r in audit['results'])
    outputs=[ROOT/f'results/grid{side}-reference-d-v1' for side in (3,5,7)]
    for output in outputs:
        if output.exists():raise FileExistsError(f'{output}: preserve existing experiments')
    jobs=[]
    for side,output in zip((3,5,7),outputs):
        config=json.loads((ROOT/f'configs/grid{side}-{150 if side==7 else 50}k.json').read_text())
        config.update(reference_phase='d_only',uniform_mix=.5,coverage_ema_decay=.99)
        snapshot(output,config)
        (output/'protocol.json').write_text(json.dumps({
            'methods':METHODS,'d_real_points':256,'d_fake_points':256,'g_fake_points':256,
            'reference_phase':'D only; G uses original random/uniform real sampling',
            'near':'Minimum-sum squared Euclidean distance one-to-one assignment of the original real and fake batches. No point is duplicated or removed.',
            'deficit':'D real mode probabilities = 0.5/K + 0.5 * max(1/K - EMA accepted generator mass, 0) / sum positive deficits; uniform fallback.',
            'ema':'Decay .99, initialized at 1/K, updated once per step using D fake batch after selecting this step’s real samples. Invalid fake mass stays in the denominator.',
            'control':'Vanilla uses the same adaptive rule driven by its own generator. Weight trajectories are not identical between methods.',
            'target':'All metrics still compare with the original uniform mixture.',
            'workers':8,'threads_per_worker':1
        },indent=2)+'\n')
        for seed in config['seeds']:
            for method in METHODS if seed%2==0 else METHODS[::-1]:
                jobs.append((config,method,seed,output/f'{method}_seed{seed}'))
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(job,args) for args in jobs]):
            print('COMPLETED',future.result(),flush=True)


if __name__=='__main__':main()
