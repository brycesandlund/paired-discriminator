"""Launch G-only deficit references; prior grid trajectories remain untouched."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import contextlib
import json
from .grid_reference_g import ROOT, run, snapshot


def job(args):
    config,seed,path=args
    with Path(str(path)+'.log').open('w') as log,contextlib.redirect_stdout(log):
        run(config,'paired_deficit',seed,path)
    return str(path)


def main():
    outputs=[ROOT/f'results/grid{side}-reference-g-v1' for side in (3,5,7)]
    for output in outputs:
        if output.exists():raise FileExistsError(output)
    jobs=[]
    for side,output in zip((3,5,7),outputs):
        config=json.loads((ROOT/f'configs/grid{side}-{150 if side==7 else 50}k.json').read_text())
        config.update(reference_phase='g_only',uniform_mix=.5,coverage_ema_decay=.99)
        snapshot(output,config)
        (output/'protocol.json').write_text(json.dumps({'method':'paired_deficit','phase':'G only; D real sampling uniform','d_real':256,'d_fake':256,'g_fake':256,'g_real_references':256,'ema':'Accepted mass of existing D fake batch; updated after both optimizer steps; decay .99; initial 1/K','weights':'0.5 uniform + 0.5 normalized positive deficits; computed from previous EMA and used for G references','evaluation_target':'original uniform Gaussian mixture','workers':8},indent=2)+'\n')
        for seed in config['seeds']:jobs.append((config,seed,output/f'paired_deficit_seed{seed}'))
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(job,args) for args in jobs]):
            print('COMPLETED',future.result(),flush=True)
    from .grid_reference_g_report import build
    build(ROOT/'results/grid-reference-g-comparison-v1')

if __name__=='__main__':main()
