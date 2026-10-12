"""Verify/report the prespecified two-model 20-minute CIFAR comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cifar_class_representation import CLASSES

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/cifar-budget20-v1'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def report():
    protocol=read(R/'protocol.json');phase=read(R/'phase_status.json');gate=read(R/'verification.json')
    assert phase==dict(phase='complete',errors=[])
    results=read(R/'results.json');assert len(results['results'])==2 and not results['errors']
    assert len(gate['checks'])==2 and gate['paired_reset_parity']
    assert all(x['exact_cuda_resume'] and x['original_recipe_parity'] for x in gate['checks'])
    dispatch=read(R/'dispatch.json');assert [x['spec'] for x in dispatch]==protocol['specs']
    assert len({x['call_id'] for x in dispatch})==2
    for name,h in protocol['source_hashes'].items():assert sha(ROOT/f'src/paired_discriminator/{name}.py')==h,name
    for name,h in protocol['dependency_hashes'].items():assert sha(ROOT/name)==h,name
    old=read(ROOT/'results/cifar-tune-v1/confirm_paired_control_seed1/eval_020000_ema/metrics.json')
    rows=[];provenance=[]
    for spec in protocol['specs']:
        m=spec['method'];folder=R/f'{m}_seed3';r=read(folder/'completed.json');p=read(folder/'provenance.json')
        assert r['spec']==spec and p['seed']==3 and p['method']==m
        status=read(folder/'status.json')
        assert status['phase']=='training_budget_complete' and status['step']==r['step']
        assert status['training_seconds']==r['training_seconds']
        assert r==next(x for x in results['results'] if x['spec']==spec)
        from .cifar_budget import config
        assert p['config']==config(m)
        assert r['gpu']==p['gpu']=='NVIDIA A10'
        assert 1200<=r['training_seconds']<=1205 and r['step']%10==0
        assert p['budget_seconds']==1200
        for name,h in p['source_hashes'].items():assert protocol['source_hashes'][name]==h
        assert sha(folder/'training_source.py')==p['source_hashes']['cifar_budget']
        e=read(folder/'eval_ema/metrics.json');assert e==r['evaluation'] and e['first_batch_replay_exact']
        assert e['signature']['method']==m and e['signature']['seed']==3
        assert e['signature']['noise_seed']==99574 and e['signature']['samples']==10000 and e['signature']['weights']=='ema'
        assert e['signature']['source_sha256']==protocol['source_hashes']['cifar_tune']
        assert e['reference_manifest']==old['reference_manifest'] and e['classifier_weight_hashes']==old['classifier_weight_hashes']
        assert e['base_step']==0 and e['step']==r['step']
        assert sha(folder/'eval_ema/evaluation_source.py')==protocol['source_hashes']['cifar_tune']
        logs=read(folder/'training.json');assert logs[-1]['step']==r['step']
        assert all(np.isfinite(x['loss']) and (x['d_loss'] is None or np.isfinite(x['d_loss'])) for x in logs)
        assert all(b['training_seconds']>a['training_seconds'] for a,b in zip(logs,logs[1:]))
        v=dict(method=m,step=r['step'],training_minutes=r['training_seconds']/60,evaluation_minutes=e['evaluation_seconds']/60,
            worker_minutes=r['worker_seconds']/60,inference_ms_per_image=1000*r['inference']['median_seconds_per_image'],
            train_fid=e['train_metrics']['frechet_inception_distance'],fid=e['heldout_metrics']['frechet_inception_distance'],
            kid=e['heldout_metrics']['kernel_inception_distance_mean'],precision=e['heldout_metrics']['precision'],recall=e['heldout_metrics']['recall'],
            vgg_tv=e['classifiers']['cifar10_vgg16_bn']['class_tv'],resnet_tv=e['classifiers']['cifar10_resnet56']['class_tv'],
            vgg_acceptance=e['classifiers']['cifar10_vgg16_bn']['confidence']['0.9']['acceptance'],
            real_draws=r['step']*128*(2 if m=='paired' else 1),parameters_g=p['parameters_g'],parameters_d=p['parameters_d'])
        assert all(np.isfinite(x) for k,x in v.items() if k!='method')
        rows.append(v);provenance.append(p)
    assert provenance[0]['data_sha256']==provenance[1]['data_sha256']
    budget=dict(training_minutes=sum(r['training_minutes'] for r in rows),evaluation_minutes=sum(r['evaluation_minutes'] for r in rows),
        worker_minutes=sum(r['worker_minutes'] for r in rows),verification_minutes=gate['gpu_seconds']/60,
        limitations='Worker time covers data-loaded worker body, training/checkpoint/evaluation/benchmark; excludes container setup and data loading. Not exact provider billing.')
    (R/'summary.json').write_text(json.dumps(dict(records=rows,budget=budget),indent=2)+'\n')
    (R/'verification_report.json').write_text(json.dumps(dict(training_jobs=2,evaluations=2,time_budgets_verified=True,
        all_source_data_classifier_reference_checks=True,replay_checks=True,cuda_checks=True,frozen_protocol=True),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,6.5))
    for ax,r in zip(axes,rows):
        ax.imshow(plt.imread(R/f'{r["method"]}_seed3/eval_ema/samples.png'));ax.axis('off')
        ax.set_title(f"{r['method']} + EMA · {r['training_minutes']:.2f} training min\n{r['step']:,} updates · FID {r['fid']:.2f}",fontsize=12,pad=10)
    fig.suptitle('CIFAR · fresh seed 3 · fixed first 64 samples, unfiltered',y=.98,fontsize=14);fig.tight_layout(rect=[0,0,1,.9])
    fig.savefig(R/'comparison_samples.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,cl in zip(axes,['cifar10_vgg16_bn','cifar10_resnet56']):
        for i,r in enumerate(rows):
            e=read(R/f'{r["method"]}_seed3/eval_ema/metrics.json')
            ax.bar(np.arange(10)+(i-.5)*.36,np.array(e['classifiers'][cl]['class_mass'])*100,.36,label=r['method'])
        ax.axhline(10,color='black',ls=':');ax.set(xticks=range(10),xticklabels=CLASSES,ylabel='Generated class mass (%)',title=cl)
        ax.tick_params(axis='x',rotation=45)
    axes[0].legend();fig.tight_layout();fig.savefig(R/'class_mass.png',dpi=150);plt.close(fig)
    lines=['# CIFAR under a 20-minute training budget (#40)','',
        'Two previously selected recipes, fresh training seed 3, one final EMA checkpoint each. No architecture/optimizer sweep and no selection on these outputs. Best-known recipes are not proven optimal for this budget.','',
        'At the fixed 20-minute training endpoint, paired GAN has lower FID/KID and higher feature recall; flow has higher precision and better class balance under both classifiers. This reverses the longer-training FID ranking for this single seed. It does not establish an optimal recipe or a robust cross-seed ranking.', '',
        '| Recipe | Updates | Train FID ↓ | Test FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | VGG TV ↓ | ResNet TV ↓ |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} + EMA | {r['step']:,} | {r['train_fid']:.2f} | {r['fid']:.2f} | {r['kid']:.5f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['vgg_tv']:.4f} | {r['resnet_tv']:.4f} |")
    lines+=['','Single-seed outcomes: no training-seed uncertainty estimate. Both use the same evaluation sample count and frozen references/classifiers; noise 99574 is new. Test-reference metrics are descriptive given shared research history. Class TV does not measure within-class diversity.','',
        'The first 64 unfiltered samples from both methods include ambiguous or malformed objects. Visual inspection does not establish a categorical quality winner. Class-mass plots show paired overrepresenting frogs (about 19–21%) and both methods underrepresenting automobiles (about 2–4%).', '',
        '![Samples](comparison_samples.png)','','![Class proportions](class_mass.png)','','## Recipes and budget','',
        'Original paired GAN: width 64, latent 128, Adam 2e-4/betas(.5,.999), EMA .999. At 50k updates reset both Adam optimizers, sampling RNG, and EMA to raw G, reproducing the selected experiment 37 branch schedule. Flow: width 32 velocity net, Adam 2e-4/betas(.9,.999), 500-step warmup,EMA .999. Unchanged BCE/velocity-MSE, batch 128, uniform unconditional sampling; no deficit or labels.','',
        'Both stop when accumulated synchronized training time reaches 1,200 seconds (ten-update checking granularity, at most 5 seconds permitted overshoot). This time includes forward/backward/optimizers/EMA and training-loop overhead; excludes evaluation, setup and checkpoint/volume I/O. This is a training-time comparison, not equal provider billing.','',
        '| Recipe | Training min | Evaluation min | Worker-body min | Batched inference ms/image | G parameters | D parameters | Real-image draws |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {r['training_minutes']:.2f} | {r['evaluation_minutes']:.2f} | {r['worker_minutes']:.2f} | {r['inference_ms_per_image']:.4f} | {r['parameters_g']:,} | {r['parameters_d']:,} | {r['real_draws']:,} |")
    lines+=['',f"Total measured training {budget['training_minutes']:.2f} GPU-minutes, evaluation {budget['evaluation_minutes']:.2f}, small verification {budget['verification_minutes']:.2f}. Worker-body time {budget['worker_minutes']:.2f} GPU-minutes includes I/O and benchmark; do not add it to training/evaluation again. Container setup/data loading remain unmeasured.",'',
        'Inference: A10, batch 128, two warmups and median of five synchronized timed batches, reported per image. This is batched throughput, not batch-one latency. Flow uses 64 midpoint steps / 128 velocity calls versus one G call for GAN.','',
        'Why these recipes: prior paired EMA train FID 49.28 effectively tied independent scoring 49.25, with better class balance. Cross-attention cost more without a clear overall win. Flow EMA was repeatably strong; lower LR did not independently help. No newly enlarged networks or expensive searches were added. Prior studies used more compute, so this result cannot establish their equal-budget ceiling.','',
        'Cost caveat: checkpoint/volume I/O kept GPUs allocated beyond their training caps. The 58.84 worker-body GPU-minutes are substantially above the 40.00 measured training minutes; future cost caps should include this overhead. No extra training, retries or sweeps were added.', '',
        'Checks: two CUDA exact model/Adam/RNG split resumes and parity against original loops; paired reset parity; four local tests; two full first-batch generation replays; source/data/classifier/reference identity; frozen specs and timing checks.','',
        '[Frozen protocol](protocol.json) · [Metrics and budget](summary.json) · [Verification](verification_report.json)','']
    (R/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(records=rows,budget=budget),indent=2))

if __name__=='__main__':report()
