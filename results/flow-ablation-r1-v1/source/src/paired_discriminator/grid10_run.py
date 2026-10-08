"""10×10 fixed-spacing grid, uniform vs D-deficit vanilla and paired."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import contextlib,json
from .grid_reference import ROOT,run,snapshot
METHODS=['vanilla','paired','vanilla_deficit','paired_deficit']

def job(args):
    config,method,seed,path=args
    with Path(str(path)+'.log').open('w') as log,contextlib.redirect_stdout(log):
        run(config,method,seed,path)
    return str(path)

def main():
    root=ROOT/'results/grid10-150k-v1'
    if root.exists():raise FileExistsError(root)
    config=json.loads((ROOT/'configs/grid10-150k.json').read_text());snapshot(root,config)
    (root/'protocol.json').write_text(json.dumps({'methods':METHODS,'seeds':config['seeds'],'steps':150000,'spacing':1.5,'sigma':.1,'center_extent':[-6.75,6.75],'deficit_phase':'D only; G references uniform','ema_decay':.99,'uniform_mix':.5,'d_real':256,'d_fake':256,'g_fake':256,'evaluation_samples':10000,'coverage_thresholds':{'legacy':.01,'half_target':.005,'relative_8pct_target':.0008},'density_edges':{'min':-7.75,'max':7.75,'width':.05},'workers':8},indent=2)+'\n')
    jobs=[]
    for seed in config['seeds']:
        for method in METHODS if seed%2==0 else METHODS[::-1]:jobs.append((config,method,seed,root/f'{method}_seed{seed}'))
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(job,args) for args in jobs]):print('COMPLETED',f.result(),flush=True)
    from .grid10_report import build
    build()
if __name__=='__main__':main()
