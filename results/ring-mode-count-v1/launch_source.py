from pathlib import Path
import json, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from paired_discriminator.experiment import run, snapshot
ROOT=Path('/Users/brycesandlund/GitHub/paired-discriminator')

def job(args):
 config,method,seed,path=args
 with open(str(path)+'.log','w') as f:
  import contextlib
  with contextlib.redirect_stdout(f):run(config,method,seed,path)
 return str(path)

if __name__=='__main__':
 base=json.loads((ROOT/'configs/ring8-50k.json').read_text())
 jobs=[]
 for modes in [16,32]:
  config={**base,'modes':modes}
  (ROOT/f'configs/ring{modes}-50k.json').write_text(json.dumps(config,indent=2)+'\n')
  path=ROOT/f'results/ring{modes}-50k-v1'
  snapshot(path,config)
  for seed in config['seeds']:
   for method in (['vanilla','rsgan','paired'] if seed%2==0 else ['paired','rsgan','vanilla']):
    jobs.append((config,method,seed,path/f'{method}_seed{seed}'))
 with ProcessPoolExecutor(max_workers=4) as pool:
  for path in pool.map(job,jobs):print('COMPLETED',path,flush=True)
