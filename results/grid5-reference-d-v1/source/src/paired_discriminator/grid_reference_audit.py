"""Real-versus-real source-role audit for the nearby pairing policy."""
import json
from pathlib import Path
import numpy as np
import torch
from .grid_reference import ROOT, models, sample_real, near_match, discriminator_loss
from .grid_pacgan import pair_inputs


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    config=json.loads((ROOT/'configs/grid5-50k.json').read_text())
    results=[]
    for seed in range(3):
        _,d=models(config,'paired',seed)
        opt=torch.optim.Adam(d.parameters(),lr=config['learning_rate'],betas=tuple(config['betas']))
        rng=torch.Generator().manual_seed(81000+seed)
        for step in range(2000):
            a=sample_real(config,256,rng);b=sample_real(config,256,rng)
            a=near_match(a,b);slots=torch.randperm(256,generator=rng)<128
            opt.zero_grad(set_to_none=True)
            loss=discriminator_loss(d,a,b,'paired',slots);loss.backward();opt.step()
        rng=torch.Generator().manual_seed(91000+seed);accuracies=[]
        with torch.no_grad():
            for _ in range(200):
                a=sample_real(config,256,rng);b=sample_real(config,256,rng)
                a=near_match(a,b);slots=torch.randperm(256,generator=rng)<128
                predictions=d(pair_inputs(a,b,slots)).flatten()>0
                accuracies.append((predictions==slots).float().mean().item())
        results.append({'seed':seed,'heldout_accuracy':float(np.mean(accuracies)),
                        'batch_accuracy_se':float(np.std(accuracies,ddof=1)/np.sqrt(len(accuracies))),
                        'training_steps':2000,'heldout_pairs':51200})
        print(results[-1],flush=True)
    assert all(.47<r['heldout_accuracy']<.53 for r in results), 'Unexpected source-role signal: investigate before launching'
    out=ROOT/'results/grid-reference-role-audit-v1';out.mkdir(exist_ok=False)
    (out/'audit.json').write_text(json.dumps({'config':config,'results':results,
        'scope':'5x5 null diagnostic: both source batches sampled independently from the same real mixture; nearby matching and randomized slots.',
        'qualification':'Empirical chance accuracy plus source-swap-equivariance test; not a power guarantee against every possible classifier.'},indent=2)+'\n')


if __name__=='__main__':main()
