"""Fresh five-seed paired D+G deficit experiment."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib,json
from .grid10_both import ROOT,run,snapshot

def job(args):
    config,seed,path=args
    with path.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):
        run(config,'paired_deficit',seed,path)
    return str(path)

def main():
    root=ROOT/'results/grid10-both-1200k-v1'
    if root.exists():raise FileExistsError(root)
    c=json.loads((ROOT/'configs/grid10-both-1200k.json').read_text());snapshot(root,c)
    with ProcessPoolExecutor(max_workers=5) as pool:
        for future in as_completed([pool.submit(job,(c,s,root/f'paired_deficit_seed{s}')) for s in c['seeds']]):
            print('COMPLETED',future.result(),flush=True)
    from .grid10_both_report import build
    build()
if __name__=='__main__':main()
