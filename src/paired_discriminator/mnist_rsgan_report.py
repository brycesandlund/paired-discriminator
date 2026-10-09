"""Ordinary MNIST coverage, balance, confidence and sample galleries."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

METHODS=('vanilla','rsgan','paired')
COLORS={'vanilla':'#4477AA','paired':'#228833','rsgan':'#CC6677'}


def report(root=Path('results/mnist-rsgan-v1')):
    old=Path('results/mnist-v1')
    def folder_for(method,seed): return (root if method=='rsgan' else old)/f'{method}_seed{seed}'
    root=Path(root);records={};metas=[];rows=[];runtimes=[]
    classifier=json.loads((old/'classifier/report.json').read_text())
    config=json.loads((old/'config.json').read_text())
    target=np.array(classifier['target_digit_mass'])
    assert hashlib.sha256((old/'classifier/classifier.pt').read_bytes()).hexdigest()==classifier['checkpoint_sha256']
    for method in METHODS:
        for seed in range(5):
            folder=folder_for(method,seed)
            meta=json.loads((folder/'run.json').read_text());metas.append(meta)
            assert meta['config']==config
            assert meta['source_sha256']==hashlib.sha256((folder/'training_source.py').read_bytes()).hexdigest()
            assert meta['dataset_sha256']==classifier['dataset']['train']['images_sha256']
            assert meta['evaluation_signature']['classifier_sha256']==classifier['checkpoint_sha256']
            status=json.loads((folder/'status.json').read_text());assert status['step']==50000
            runtimes.append({'method':method,'seed':seed,'training_minutes':status['training_seconds']/60,'gpu':meta['gpu']})
            with (folder/'training.csv').open() as f:logs=list(csv.DictReader(f))
            assert len(logs)==500 and all(np.isfinite(float(r[k])) for r in logs for k in ('d_loss','g_loss'))
            rr=[]
            for step in range(1000,50001,1000):
                r=json.loads((folder/f'metrics_{step:06d}.json').read_text())
                assert (r['method'],r['seed'],r['step'])==(method,seed,step)
                assert r['samples']==10000 and sum(r['digit_counts'])==10000
                assert r['evaluation_signature']==meta['evaluation_signature']
                assert r['evaluation_signature']['evaluation_source_sha256']==hashlib.sha256((folder/'evaluation_source.py').read_bytes()).hexdigest()
                rr.append(r)
                row={k:r[k] for k in ('method','seed','step','mode_tv','covered_digits','observed_digits','confidence_covered_digits','confidence_accepted_fraction','confidence_augmented_tv','mean_confidence')}
                for digit in range(10):
                    row[f'digit_{digit}_mass']=r['digit_mass'][digit]
                    row[f'digit_{digit}_accepted_mass']=r['accepted_digit_mass'][digit]
                assert all(np.isfinite(v) for k,v in row.items() if k!='method')
                rows.append(row)
            records[(method,seed)]=rr
    sources=[(folder_for(m,seed)/'training_source.py').read_text().replace('METHODS = ("vanilla", "paired", "rsgan")','METHODS = ("vanilla", "paired")') for m in METHODS for seed in range(5)]
    assert len(set(sources))==1
    gate=json.loads((root/'verification.json').read_text())
    assert all(gate[k] for k in ('cuda_exact_resume','optimizer_and_rng_exact','evaluation_neutral','initial_g_and_d_match_vanilla'))
    assert json.loads((root/'phase_status.json').read_text())=={'phase':'complete','errors':[]}
    assert len(rows)==750
    assert len({json.dumps(m['evaluation_signature'],sort_keys=True) for m in metas})==1
    assert len({m['parameters_g'] for m in metas})==1
    with (root/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def values(method,key,step):
        return np.array([records[(method,seed)][step//1000-1][key] for seed in range(5)])
    summary={}
    for step in (10000,50000):
        summary[str(step)]={}
        for method in METHODS:
            summary[str(step)][method]={}
            for key in ('mode_tv','covered_digits','confidence_covered_digits','confidence_accepted_fraction','confidence_augmented_tv'):
                v=values(method,key,step)
                summary[str(step)][method][key]={'mean':float(v.mean()),'std':float(v.std(ddof=1)),'by_seed':v.tolist()}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    verification={'runs':15,'evaluations':len(rows),'steps_per_run':50000,'classifier_sha256':classifier['checkpoint_sha256'],
                  'same_configuration':True,'same_reference_distribution':True,'source_hashes_valid':True,'runtimes':runtimes,
                  'parameters':{m:{k:next(x[k] for x in metas if x['method']==m) for k in ('parameters_g','parameters_d')} for m in METHODS}}
    (root/'verification_report.json').write_text(json.dumps(verification,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8));xx=np.arange(1,51)
    for ax,key,title in zip(axes.flat,('mode_tv','covered_digits','confidence_augmented_tv','confidence_accepted_fraction'),
                            ('Digit-frequency TV ↓','Digits with ≥1% generated mass ↑','Confidence-augmented TV ↓','Fraction with classifier confidence ≥0.9 ↑')):
        for method in METHODS:
            all_values=np.array([[r[key] for r in records[(method,seed)]] for seed in range(5)])
            for vv in all_values:ax.plot(xx,vv,color=COLORS[method],alpha=.22,lw=.8)
            mean=all_values.mean(0);std=all_values.std(0,ddof=1)
            ax.plot(xx,mean,color=COLORS[method],label=method,lw=2)
            ax.fill_between(xx,mean-std,mean+std,color=COLORS[method],alpha=.12)
        if key=='covered_digits':ax.set_ylim(0,10.5);ax.set_yticks(range(0,11,2))
        if key=='confidence_accepted_fraction':ax.set_ylim(0,1.02)
        if key in ('mode_tv','confidence_augmented_tv'):
            ax.axhline(classifier['test_diagnostics'][key],color='gray',ls=':',label='real test classifier reference')
        ax.set(xlabel='D and G updates (thousands)',title=title);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Ordinary MNIST · five seeds · thin lines: seeds; bands: ± sample SD')
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
    for key,filename,title in [('digit_mass','digit_mass.png','All predicted digit mass'),('accepted_digit_mass','accepted_mass.png','Mass with classifier confidence ≥0.9 (denominator: all generated images)')]:
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
        for ax,step in zip(axes,(10000,50000)):
            for mi,method in enumerate(METHODS):
                vv=values(method,key,step);x=np.arange(10)+(mi-1)*.25
                ax.bar(x,vv.mean(0),width=.23,yerr=vv.std(0,ddof=1),capsize=2,color=COLORS[method],alpha=.8,label=method)
                for si in range(5):ax.scatter(x+(si-2)*.025,vv[si],s=9,color=COLORS[method],alpha=.65,zorder=3)
            ax.scatter(np.arange(10),target,marker='_',s=250,color='black',label='training digit proportions',zorder=4)
            ax.set(xticks=np.arange(10),xlabel='Digit',title=f'{step//1000}k updates',ylabel='Fraction of all generated samples');ax.grid(axis='y',alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle(title+' · bars mean; dots seeds; whiskers SD')
        fig.tight_layout();fig.savefig(root/filename,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,sharey=True)
    for mi,method in enumerate(METHODS):
        for si,step in enumerate((10000,50000)):
            ax=axes[mi,si]
            vv=values(method,'digit_mass',step)-target
            im=ax.imshow(vv,cmap='RdBu_r',vmin=-.25,vmax=.25,aspect='auto')
            ax.set(title=f'{method} · {step//1000}k',xlabel='Predicted digit',ylabel='Seed',xticks=np.arange(10),yticks=np.arange(5))
    fig.subplots_adjust(right=.88,hspace=.45);cax=fig.add_axes([.91,.15,.02,.7]);fig.colorbar(im,cax=cax,label='Generated mass minus training mass')
    fig.suptitle('Per-seed digit imbalance · common color scale');fig.savefig(root/'digit_imbalance.png',dpi=160);plt.close(fig)
    for step in (10000,50000):
        first=Image.open(folder_for('vanilla',0)/f'samples_{step:06d}.png').convert('RGB');w,h=first.size
        canvas=Image.new('RGB',(5*(w+10),3*(h+32)),'#eeeeee');draw=ImageDraw.Draw(canvas)
        for mi,method in enumerate(METHODS):
            for seed in range(5):
                x=seed*(w+10);y=mi*(h+32)
                draw.text((x+5,y+8),f'{method} · seed {seed}',fill='black')
                canvas.paste(Image.open(folder_for(method,seed)/f'samples_{step:06d}.png').convert('RGB'),(x,y+30))
        canvas.save(root/f'samples_{step//1000}k.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,step in zip(axes,(10000,50000)):
        for method in METHODS:
            vv=values(method,'confidence_histogram',step)/10000
            ax.stairs(vv.mean(0),np.linspace(0,1,21),color=COLORS[method],label=method,lw=2)
        ax.axvline(.9,color='gray',ls=':')
        ax.set(title=f'{step//1000}k updates',xlabel='Maximum classifier probability',ylabel='Fraction per 0.05-wide bin')
        ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Uncalibrated classifier confidence · mean over five seeds')
    fig.tight_layout();fig.savefig(root/'confidence_histograms.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));conf=np.array(classifier['test_confusion_matrix']);im=ax.imshow(conf/conf.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Blues')
    ax.set(xticks=np.arange(10),yticks=np.arange(10),xlabel='Predicted digit',ylabel='True digit',title=f'Frozen classifier · official-test accuracy {classifier["test_accuracy"]:.2%}')
    fig.colorbar(im,ax=ax,label='Row fraction');fig.tight_layout();fig.savefig(root/'classifier_confusion.png',dpi=150);plt.close(fig)
    def fmt(step,method,key,percent=False):
        m=summary[str(step)][method][key];scale=100 if percent else 1
        return f"{m['mean']*scale:.3f} ± {m['std']*scale:.3f}"
    lines=['# MNIST: matched vanilla, RSGAN and paired','',
           'Five seeds per method, unconditional native 28×28 images, 50k D and G updates, evaluation every 1k. G architecture and update batch are identical across methods. Digit labels are used only for evaluation.','',
           '[Protocol](../../docs/MNIST.md) · [Raw metrics](metrics.csv) · [Machine-readable summary](summary.json)','',
           '![Learning curves](learning_curves.png)','','## Endpoint summaries','',
           'Mean ± sample standard deviation across five training seeds. Coverage requires at least 1% of all generated images per digit. Confidence-filtered metrics use a fixed 0.9 classifier threshold.','',
           '| Updates | Method | Digit coverage /10 | Mode TV ↓ | Confidence-filtered coverage /10 | Confidence acceptance % | Confidence-augmented TV ↓ |',
           '|---|---|---:|---:|---:|---:|---:|']
    for step in (10000,50000):
        for method in METHODS:
            lines.append(f"| {step//1000}k | {method} | {fmt(step,method,'covered_digits')} | {fmt(step,method,'mode_tv')} | {fmt(step,method,'confidence_covered_digits')} | {fmt(step,method,'confidence_accepted_fraction',True)} | {fmt(step,method,'confidence_augmented_tv')} |")
    lines+=['','![Digit frequencies](digit_mass.png)','','![Accepted mass](accepted_mass.png)','','![Individual seed imbalances](digit_imbalance.png)','',
            '## Individual endpoints','','| Method | Seed | Updates | Digit coverage | Mode TV | Confidence coverage | Confidence acceptance | Confidence-augmented TV |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for step in (10000,50000):
        for method in METHODS:
            for seed in range(5):
                r=records[(method,seed)][step//1000-1]
                lines.append(f"| {method} | {seed} | {step} | {r['covered_digits']} | {r['mode_tv']:.4f} | {r['confidence_covered_digits']} | {r['confidence_accepted_fraction']:.2%} | {r['confidence_augmented_tv']:.4f} |")
    lines+=['','## Random fixed-noise samples','','The same 64 fixed latent vectors are used for all methods/seeds/checkpoints. These grids are not confidence-selected.','','### 10k','','![10k samples](samples_10k.png)','','### 50k','','![50k samples](samples_50k.png)','',
            '## Samples grouped by predicted digit','','Rows are predicted digits 0–9, with the first ten matching samples in fixed random evaluation order. They are not ranked by quality or confidence. Inspect for malformed digits, classifier errors, and repeated styles.','',
            '| Method / seed | 10k | 50k |','|---|---|---|']
    for method in METHODS:
        for seed in range(5):
            stem=('../mnist-v1/' if method!='rsgan' else '')+f'{method}_seed{seed}'
            lines.append(f'| {method} / {seed} | [Gallery]({stem}/by_predicted_digit_010000.png) | [Gallery]({stem}/by_predicted_digit_050000.png) |')
    real=classifier['test_diagnostics'];ref=classifier['real_iid_10000_label_tv_reference']
    lines+=['','## Evaluator and interpretation','',
            f"The classifier was selected at epoch {classifier['selected_epoch']} using a 10k validation subset of the training split, after learning on the other 50k. Official-test accuracy is **{classifier['test_accuracy']:.2%}**. It never supplies GAN gradients. Its checkpoint is frozen and hashed.",'',
            f"On official test images, classifier-based mode TV is {real['mode_tv']:.4f}, confidence acceptance is {real['confidence_accepted_fraction']:.2%}, and confidence-augmented TV is {real['confidence_augmented_tv']:.4f}. True-label sampling from the training proportions at n=10,000 gives average TV {ref['mean']:.4f}, with central 95% interval [{ref['p025']:.4f}, {ref['p975']:.4f}] over 1,000 draws.",'',
            '![Classifier confusion matrix](classifier_confusion.png)','','![Confidence histograms](confidence_histograms.png)','',
            'MNIST is not exactly class balanced; the target is its empirical training-label distribution. Mode TV is half the sum of absolute differences between predicted generated proportions and that target. Confidence-augmented TV adds a rejected bucket with ideal target mass zero; real images can therefore have a nonzero score. Confidence is uncalibrated and can be wrong on generated images. These are digit-category proxies, not a proof of fidelity or within-digit style diversity.','',
            'Repeated checkpoints use the same evaluation noise and are correlated. Five seeds quantify training variability for this configuration, not generality over architectures or hyperparameters. No checkpoint selection or hyperparameter retuning was performed for this matched RSGAN extension. Evaluation data and classifier have shared research history; this is not untouched validation.','',
            '## Reproducibility and cost','','| Method | Seed | Recorded training minutes | GPU |','|---|---:|---:|---|']
    for r in runtimes:lines.append(f"| {r['method']} | {r['seed']} | {r['training_minutes']:.2f} | {r['gpu']} |")
    lines+=['','Training timings exclude evaluation and most checkpoint I/O. Same real/fake counts and optimizer updates do not imply equal D FLOPs: paired processes two channels jointly into the same hidden width. No wider-D or larger-D-batch arm is included here.','',
            'Seven local MNIST tests and the RSGAN CUDA gate passed. All 750 evaluation records passed configuration/reference/source checks. The sole archived trainer source difference is enabling RSGAN in METHODS. Original vanilla/paired artifacts are reused unchanged. Full checkpoints remain on Modal. Evaluation timing was not instrumented separately, so training time is not total GPU cost. Generator architecture and inference work per image are unchanged across methods.', '']
    comparison={}
    for step in (10000,50000):
        comparison[str(step)]={
            'paired_lower_tv_seeds':int((values('paired','mode_tv',step)<values('vanilla','mode_tv',step)).sum()),
            'paired_lower_confidence_augmented_tv_seeds':int((values('paired','confidence_augmented_tv',step)<values('vanilla','confidence_augmented_tv',step)).sum()),
            'full_raw_coverage_runs':{m:int((values(m,'covered_digits',step)==10).sum()) for m in METHODS},
            'full_confidence_coverage_runs':{m:int((values(m,'confidence_covered_digits',step)==10).sum()) for m in METHODS}}
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    for step in (10000,50000):
        comparison[str(step)]['paired_lower_tv_than_rsgan_seeds']=int((values('paired','mode_tv',step)<values('rsgan','mode_tv',step)).sum())
        comparison[str(step)]['paired_lower_augmented_tv_than_rsgan_seeds']=int((values('paired','confidence_augmented_tv',step)<values('rsgan','confidence_augmented_tv',step)).sum())
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    lines += ['', '## RSGAN comparison', '', 'RSGAN uses shared unary scores with pairwise logistic discriminator loss and the flipped generator target. The generator and unary discriminator initialize identically to vanilla for each seed. Primary endpoints were fixed at 10k and 50k before results; no checkpoint or seed selection. All reported aggregates average individual-seed metrics, not pooled distributions.', '']
    for step in (10000,50000):
        c=comparison[str(step)]
        lines.append(f"At {step//1000}k, paired has lower digit TV than RSGAN in {c['paired_lower_tv_than_rsgan_seeds']}/5 matched seeds and lower confidence-augmented TV in {c['paired_lower_augmented_tv_than_rsgan_seeds']}/5.")
    lines += ['', 'Visual inspection: RSGAN concentrates on 1, 7 and 9 at 50k and underrepresents 0, 2, 5 and 6. Paired has broader digit representation but still contains malformed/ambiguous samples. Neither classifier confidence nor these grids establishes complete image fidelity or within-digit diversity. RSGAN recorded 55.94 A10 training minutes total (11.19 per seed); evaluations and startup/I/O add unmeasured cost. Archived baseline timings include both A10 and A10G, so these are descriptive timings rather than a controlled speed benchmark.', '']
    lines += ['', 'Launcher packaging and verification restart/device-comparison issues were repaired before full training dispatch. No duplicate training jobs were launched. Small verification attempts and container startup/I/O are not included in recorded training time.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return summary

if __name__=='__main__':
    report()
