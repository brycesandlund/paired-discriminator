"""Overview of held-out learning curves and integrity diagnostics."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS=('vanilla','rsgan','paired')
COLORS=dict(zip(METHODS,['#4477AA','#EE6677','#228833']))


def report(root):
    root=Path(root); records=[]; metas=[]
    for method in METHODS:
        folder=root/f'{method}_seed0'
        meta=json.loads((folder/'run.json').read_text()); metas.append(meta)
        assert hashlib.sha256((folder/'training_source.py').read_bytes()).hexdigest()==meta['source_sha256']
        status=json.loads((folder/'status.json').read_text()); assert status['step']==100000
        with (folder/'training.csv').open() as f: training_rows=list(csv.DictReader(f))
        assert len(training_rows)==1000 and int(training_rows[-1]['step'])==100000
        assert all(np.isfinite(float(row[k])) for row in training_rows for k in ('d_loss','g_loss'))
        for step in range(10000,100001,10000):
            r=json.loads((folder/f'integrity_{step:06d}.json').read_text())
            assert r['step']==step and r['method']==method
            assert r['training_source_sha256']==meta['source_sha256']
            assert r['reference_manifest']['train_sha256']==meta['dataset_sha256']
            assert r['reference_manifest']['train_labels_sha256']==meta['labels_sha256']
            assert hashlib.sha256((folder/'evaluation_source.py').read_bytes()).hexdigest()==r['evaluation_source_sha256']
            records.append(r)
    assert len({r['evaluation_source_sha256'] for r in records})==1
    assert len({json.dumps(r['reference_manifest'],sort_keys=True) for r in records})==1
    for key in ('config','source_sha256','dataset_sha256','labels_sha256','torch','parameters_g'):
        assert all(m[key]==metas[0][key] for m in metas)
    rows=[]
    for r in records:
        row={'method':r['method'],'step':r['step']}
        for split in ('train','heldout'):
            row.update({split+'_'+k:v for k,v in r[split+'_metrics'].items()})
        d=r['discriminator']; n=r['nearest_neighbors']
        for k in ('train_bce','heldout_bce','train_real_acceptance','heldout_real_acceptance','train_minus_heldout_margin','train_membership_auc_from_margin'):
            row['d_'+k]=d[k]
        for space in ('inception','pixel_rms'):
            row['nn_'+space+'_train_closer_fraction']=n[space]['fraction_closer_to_train_equal_pools']
            for pool in ('train50k','train10k','heldout10k'):
                row['nn_'+space+'_'+pool+'_mean']=n[space][pool]['mean']
        row['exact_train_copies']=n['exact_generated_training_matches']
        row['unique_generated']=n['unique_generated_uint8_images']
        assert all(np.isfinite(v) for k,v in row.items() if k!='method')
        rows.append(row)
    with (root/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key,label,scale in zip(axes.flat,['frechet_inception_distance','kernel_inception_distance_mean','precision','recall'],['FID ↓','KID ×1000 ↓','Precision ↑','Recall ↑'],[1,1000,1,1]):
        for method in METHODS:
            rr=[r for r in records if r['method']==method]
            for split,style in [('heldout','-'),('train','--')]:
                ax.plot([r['step']/1000 for r in rr],[r[split+'_metrics'][key]*scale for r in rr],style,marker='o' if split=='heldout' else None,color=COLORS[method],label=method+' '+split)
        ax.set_title(label); ax.set_xlabel('Updates (thousands)'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('CBN + projection · seed 0 · solid: held-out reference; dashed: training reference')
    fig.tight_layout(); fig.savefig(root/'learning_curves.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title,scale in zip(axes,['frechet_inception_distance','kernel_inception_distance_mean','recall'],['FID: held-out − train','KID ×1000: held-out − train','Recall: held-out − train'],[1,1000,1]):
        for method in METHODS:
            rr=[r for r in records if r['method']==method]
            ax.plot([r['step']/1000 for r in rr],[(r['heldout_metrics'][key]-r['train_metrics'][key])*scale for r in rr],'o-',label=method,color=COLORS[method])
        ax.axhline(0,color='gray',ls=':'); ax.set_title(title); ax.set_xlabel('Updates (thousands)'); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); fig.savefig(root/'reference_gaps.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for method in METHODS:
        rr=[r for r in records if r['method']==method]; x=[r['step']/1000 for r in rr]
        for ax,values in zip(axes,[[r['discriminator']['train_membership_auc_from_margin'] for r in rr],
                [r['discriminator']['train_minus_heldout_margin'] for r in rr],
                [r['discriminator']['heldout_bce']-r['discriminator']['train_bce'] for r in rr]]):
            ax.plot(x,values,'o-',label=method,color=COLORS[method]); ax.set_xlabel('Updates (thousands)'); ax.grid(alpha=.2)
    for ax,title,base in zip(axes,['D train-membership AUC','D margin: train − held-out','D BCE: held-out − train'],[.5,0,0]):
        ax.set_title(title); ax.axhline(base,color='gray',ls=':')
    axes[0].legend(); fig.tight_layout(); fig.savefig(root/'discriminator_gaps.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,method in zip(axes,METHODS):
        arrays=np.load(root/f'{method}_seed0/discriminator_100000.npz')
        a,b=arrays['train_margin'],arrays['heldout_margin']
        bins=np.linspace(min(a.min(),b.min()),max(a.max(),b.max()),55)
        ax.hist(a,bins=bins,density=True,histtype='step',label='train',color='#4477AA')
        ax.hist(b,bins=bins,density=True,histtype='step',label='held-out',color='#EE6677')
        ax.set_title(method); ax.set_xlabel('Aligned real-acceptance margin')
    axes[0].set_ylabel('Density'); axes[0].legend(); fig.suptitle('Discriminator score distributions at 100k')
    fig.tight_layout(); fig.savefig(root/'discriminator_distributions_100k.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,space in zip(axes,('inception','pixel_rms')):
        for method in METHODS:
            rr=[r for r in records if r['method']==method]
            ax.plot([r['step']/1000 for r in rr],[r['nearest_neighbors'][space]['fraction_closer_to_train_equal_pools'] for r in rr],'o-',label=method,color=COLORS[method])
        ax.axhline(.5,color='gray',ls=':'); ax.set_title(space); ax.set_xlabel('Updates (thousands)'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Fraction with closer training neighbor\n(equal 10k candidate pools)'); axes[0].legend()
    fig.tight_layout(); fig.savefig(root/'nearest_preferences.png',dpi=150); plt.close(fig)
    for space in ('inception','pixel_rms'):
        fig,axes=plt.subplots(1,3,figsize=(8,12))
        for ax,method in zip(axes,METHODS):
            img=plt.imread(root/f'{method}_seed0/nn_{space}_100000.png')
            ax.imshow(img,interpolation='nearest')
            ax.set_title(method)
            ax.set_xticks([18,52,86],['G','Train','Held-out'],fontsize=8)
            ax.xaxis.tick_top(); ax.set_yticks([])
            ax.axhline(273,color='red',lw=.8)
        fig.suptitle(f'100k nearest neighbors · {space}\nAbove line: fixed queries; below: closest-to-training queries',fontsize=11)
        fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(root/f'nearest_panel_{space}_100k.png',dpi=150); plt.close(fig)
    lines=['# CIFAR-10 integrity study: 10k–100k','',
        'CBN/projection, seed 0, unchanged training settings; all three methods evaluated every 10k. Held-out references are the official CIFAR-10 test images, never used in GAN updates. Training-reference scores remain available for comparison. All metrics use 10,000 generated samples and 10,000 reference images.', '',
        '## Main findings', '',
        '- Discriminator overfitting is visible: at 100k vanilla accepts 97.5% of training real images versus 55.0% of held-out real images. RSGAN and paired also develop gaps. Train-membership AUC reaches 0.709, 0.652, and 0.655 respectively.',
        '- Generator FID/KID improve beyond 50k with diminishing average returns and late fluctuations. Vanilla has its lowest observed held-out FID at 80k (38.08); RSGAN and paired reach their lowest observed FID at 100k (38.74 and 42.53). These are descriptive minima of one fixed sweep, not selected-model claims.',
        '- Train-reference and held-out FID differ by less than 0.16 at every checkpoint, despite the growing D gap. Held-out distribution metrics alone do not expose discriminator overfitting.',
        '- No exact uint8 generated/training matches occur at any evaluated checkpoint. At 100k, equal-pool training-neighbor preference is 50.6–51.7% in Inception space and 52.8–54.1% in pixel space. These modest preferences and selected neighbor panels do not rule out approximate memorization.', '',
        '[Protocol](CIFAR10_INTEGRITY.md) · [All metrics](metrics.csv)', '',
        '## Held-out learning curves','', '![Curves](learning_curves.png)','',
        '| Updates | Method | Held-out FID ↓ | KID ×1000 ↓ | Precision ↑ | Recall ↑ |','|---:|---|---:|---:|---:|---:|']
    for step in range(10000,100001,10000):
        for method in METHODS:
            r=next(r for r in records if r['step']==step and r['method']==method); m=r['heldout_metrics']
            lines.append(f"| {step//1000}k | {method} | {m['frechet_inception_distance']:.2f} | {1000*m['kernel_inception_distance_mean']:.2f} | {m['precision']:.3f} | {m['recall']:.3f} |")
    lines+=['','## Train/held-out reference gaps','','![Reference gaps](reference_gaps.png)','','These differences use the same generated images and equal reference counts. Small differences can reflect the fixed reference samples and are not, by themselves, evidence of overfitting.','','## Discriminator generalization','','Fake images and labels are fixed across train/held-out comparisons. Paired averages both slot orientations after aligning signs. Higher margin means stronger acceptance of the real input. AUC above 0.5 indicates a tendency to score training examples higher; it is not a measured training-membership attack success rate. Absolute margins differ in scale across models.','','![D gaps](discriminator_gaps.png)','','![100k score distributions](discriminator_distributions_100k.png)','',
        '| Updates | Method | Train BCE | Held-out BCE | Train acceptance | Held-out acceptance | Membership AUC |','|---:|---|---:|---:|---:|---:|---:|']
    for step in (10000,50000,100000):
        for method in METHODS:
            d=next(r['discriminator'] for r in records if r['step']==step and r['method']==method)
            lines.append(f"| {step//1000}k | {method} | {d['train_bce']:.3f} | {d['heldout_bce']:.3f} | {d['train_real_acceptance']:.3f} | {d['heldout_real_acceptance']:.3f} | {d['train_membership_auc_from_margin']:.3f} |")
    manifest=records[0]['reference_manifest']; base=manifest['real_train10k_vs_test10k_metrics']
    lines+=['','## Nearest-neighbor checks','','![Nearest preferences](nearest_preferences.png)','',
        'Equal candidate pools contain 10k training and 10k held-out images. A separate full-50k training search is used to inspect possible copies; its larger pool is not a fair distance comparator to 10k held-out images. Distances use raw Inception features or RGB pixel RMS. Similarity alone does not prove copying; absence of exact copies does not establish novelty.', '',
        f"Exact test images also present in the training split: {manifest['exact_test_images_also_in_train']}. Real train10k versus test10k baseline: FID {base['frechet_inception_distance']:.2f}, KID ×1000 {1000*base['kernel_inception_distance_mean']:.2f}.", '',
        '| Method | Exact generated/train matches at 100k | Unique generated images / 10k | Feature-space training preference | Pixel-space training preference |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        n=next(r['nearest_neighbors'] for r in records if r['step']==100000 and r['method']==method)
        lines.append(f"| {method} | {n['exact_generated_training_matches']} | {n['unique_generated_uint8_images']} | {n['inception']['fraction_closer_to_train_equal_pools']:.3f} | {n['pixel_rms']['fraction_closer_to_train_equal_pools']:.3f} |")
    lines+=['','### 100k nearest-neighbor panels','','Columns: generated / nearest of all 50k training / nearest of 10k held-out. First eight rows are fixed queries; last eight select the closest generated-to-training distances in that space. These selected rows are not representative quality samples. Query IDs and every neighbor index/distance are retained in per-checkpoint files.']
    lines+=['','![Feature nearest neighbors](nearest_panel_inception_100k.png)','','![Pixel nearest neighbors](nearest_panel_pixel_rms_100k.png)']
    lines+=['','## 50k to 100k: additional training return','',
        '| Method | Held-out FID 50k → 100k | Held-out KID ×1000 50k → 100k | Recall 50k → 100k | Lowest observed held-out FID (descriptive) |',
        '|---|---:|---:|---:|---|']
    for method in METHODS:
        rr=[r for r in records if r['method']==method]
        a=next(r['heldout_metrics'] for r in rr if r['step']==50000)
        b=next(r['heldout_metrics'] for r in rr if r['step']==100000)
        best=min(rr,key=lambda r:r['heldout_metrics']['frechet_inception_distance'])
        lines.append(f"| {method} | {a['frechet_inception_distance']:.2f} → {b['frechet_inception_distance']:.2f} | {1000*a['kernel_inception_distance_mean']:.2f} → {1000*b['kernel_inception_distance_mean']:.2f} | {a['recall']:.3f} → {b['recall']:.3f} | {best['heldout_metrics']['frechet_inception_distance']:.2f} at {best['step']//1000}k |")
    lines+=['','All checkpoints are reported. The observed minimum is a description of this fixed sweep, not an independently tested selected model.','']
    lines+=['','## Limits','','One seed, no tuning, and correlated checkpoints. Held-out FID/KID are distributional diagnostics, not memorization detectors. Neighbor searches are restricted to the chosen feature spaces and comparisons. No independently validated class-adherence score is included. Repeated inspection of this held-out set means it is a diagnostic reference, not a pristine final benchmark for later decisions based on these results.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    repo=Path(__file__).parents[2]
    for name in ('CIFAR10_INTEGRITY.md','CIFAR10_PROJECTION.md','CIFAR10_CONDITIONAL.md','CIFAR10_CLASSMATCHED.md','CIFAR10.md'):
        shutil.copy2(repo/'docs'/name,root/name)
    shutil.copy2(__file__,root/'report_source.py')
    (root/'verification.json').write_text(json.dumps({'completed_runs':3,'evaluations':30,'finite_metrics':True,'matched_training_config_and_source':True,'matched_eval_source_and_references':True,'gpu_by_method':{m['method']:m['gpu'] for m in metas}},indent=2)+'\n')
    print('Verified 30 evaluations; wrote report and plots.')


if __name__=='__main__':
    import sys
    report(sys.argv[1])
