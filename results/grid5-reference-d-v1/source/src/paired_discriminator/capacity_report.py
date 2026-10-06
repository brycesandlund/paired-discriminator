"""Compare independent discriminator batch/width probes to frozen integrity runs."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ARMS = ('vanilla', 'rsgan', 'paired', 'd_batch256', 'd_width128')
LABELS = dict(zip(ARMS, ('Vanilla', 'RSGAN', 'Paired baseline', 'Paired D batch 256', 'Paired D width 128')))
COLORS = dict(zip(ARMS, ('#4477AA', '#EE6677', '#228833', '#AA3377', '#CC9900')))


def report(root=Path('results/cifar10-capacity-v1'), baseline=Path('results/cifar10-integrity-v1')):
    root, baseline = Path(root), Path(baseline)
    records, metadata, rows, verifications = {}, {}, [], {}
    expected_keys = ('frechet_inception_distance','kernel_inception_distance_mean','precision','recall')
    for arm in ARMS:
        folder = (baseline if arm in ARMS[:3] else root) / f'{arm}_seed0'
        meta = json.loads((folder/'run.json').read_text())
        status = json.loads((folder/'status.json').read_text())
        assert status['step'] == 100000
        assert hashlib.sha256((folder/'training_source.py').read_bytes()).hexdigest() == meta['source_sha256']
        with (folder/'training.csv').open() as f:
            logs = list(csv.DictReader(f))
        assert len(logs)==1000 and int(logs[-1]['step'])==100000
        assert all(np.isfinite(float(r[k])) for r in logs for k in ('d_loss','g_loss'))
        metadata[arm] = meta
        records[arm] = []
        for step in range(10000,100001,10000):
            r = json.loads((folder/f'integrity_{step:06d}.json').read_text())
            assert r['step']==step and r['seed']==0
            assert r['method']==(arm if arm in ARMS[:3] else 'paired')
            assert r['training_source_sha256']==meta['source_sha256']
            assert r['evaluation_source_sha256']==hashlib.sha256((folder/'evaluation_source.py').read_bytes()).hexdigest()
            assert r['reference_manifest']['train_sha256']==meta['dataset_sha256']
            assert r['reference_manifest']['train_labels_sha256']==meta['labels_sha256']
            r['training_seconds'] = float(next(log for log in logs if int(log['step'])==step)['training_seconds'])
            records[arm].append(r)
            row = {'arm':arm,'step':step,'training_seconds':r['training_seconds']}
            for split in ('train','heldout'):
                row.update({split+'_'+k:r[split+'_metrics'][k] for k in expected_keys})
            for k in ('train_bce','heldout_bce','train_real_acceptance','heldout_real_acceptance','train_membership_auc_from_margin'):
                row['d_'+k] = r['discriminator'][k]
            for space in ('inception','pixel_rms'):
                row['nn_'+space+'_train_closer_fraction'] = r['nearest_neighbors'][space]['fraction_closer_to_train_equal_pools']
            row['exact_train_copies'] = r['nearest_neighbors']['exact_generated_training_matches']
            row['unique_generated'] = r['nearest_neighbors']['unique_generated_uint8_images']
            assert all(np.isfinite(v) for k,v in row.items() if k!='arm')
            rows.append(row)
        verifications[arm] = {'step':status['step'],'gpu':meta['gpu'],'training_seconds':status['training_seconds'],'parameters_g':meta['parameters_g'],'parameters_d':meta['parameters_d']}
    assert len({json.dumps(r['reference_manifest'],sort_keys=True) for rr in records.values() for r in rr})==1
    assert len({m['dataset_sha256'] for m in metadata.values()})==1
    assert len({m['parameters_g'] for m in metadata.values()})==1
    base_config = metadata['paired']['config']
    for arm,changes in [('d_batch256',{'discriminator_batch_size':256,'discriminator_width':64}),('d_width128',{'discriminator_batch_size':128,'discriminator_width':128})]:
        assert metadata[arm]['config']==base_config|changes
    old_source = (baseline/'paired_seed0/evaluation_source.py').read_text()
    new_source = (root/'d_batch256_seed0/evaluation_source.py').read_text()
    assert old_source.replace('from paired_discriminator.cifar_integrity import','from paired_discriminator.cifar_capacity import')==new_source
    assert new_source==(root/'d_width128_seed0/evaluation_source.py').read_text()
    (root/'verification.json').write_text(json.dumps({'arms':verifications,'all_50_evaluations_valid':True,'identical_reference_manifest':True,'evaluation_change_only_model_factory_import':True,'final_evaluation_runtime':json.loads((root/'d_width128_seed0/evaluation_runtime_100000.json').read_text())},indent=2)+'\n')
    with (root/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,key,title,scale in zip(axes.flat,expected_keys,('Held-out FID ↓','Held-out KID ×1000 ↓','Held-out precision ↑','Held-out recall ↑'),(1,1000,1,1)):
        for arm in ARMS:
            ax.plot([r['step']/1000 for r in records[arm]],[r['heldout_metrics'][key]*scale for r in records[arm]],'o-',color=COLORS[arm],label=LABELS[arm],lw=1.8,ms=4)
        ax.set_title(title); ax.set_xlabel('D and G optimizer updates (thousands)'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('D batch and capacity probes · G update batch 128 in every arm · seed 0')
    fig.tight_layout(); fig.savefig(root/'learning_curves.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for arm in ARMS:
        rr=records[arm]
        axes[0].plot([r['training_seconds']/60 for r in rr],[r['heldout_metrics']['frechet_inception_distance'] for r in rr],'o-',label=LABELS[arm],color=COLORS[arm],ms=4)
        axes[1].plot([r['step']/1000 for r in rr],[r['heldout_metrics']['frechet_inception_distance']-records['paired'][i]['heldout_metrics']['frechet_inception_distance'] for i,r in enumerate(rr)],'o-',label=LABELS[arm],color=COLORS[arm],ms=4)
    axes[0].set(xlabel='Recorded training minutes (excludes most checkpoint I/O)',ylabel='Held-out FID',title='Observed runtime, not a hardware-controlled compute match')
    axes[1].set(xlabel='Updates (thousands)',ylabel='FID minus paired baseline',title='Negative = better than paired baseline')
    axes[1].axhline(0,color='gray',ls=':'); axes[0].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(root/'runtime_and_deltas.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for arm in ARMS:
        rr=records[arm]; xx=[r['step']/1000 for r in rr]
        axes[0].plot(xx,[r['discriminator']['train_membership_auc_from_margin'] for r in rr],'o-',label=LABELS[arm],color=COLORS[arm],ms=4)
        axes[1].plot(xx,[r['discriminator']['heldout_bce']-r['discriminator']['train_bce'] for r in rr],'o-',color=COLORS[arm],ms=4)
        axes[2].plot(xx,[r['nearest_neighbors']['inception']['fraction_closer_to_train_equal_pools'] for r in rr],'o-',color=COLORS[arm],ms=4)
    for ax,title in zip(axes,('D train-membership ranking AUC','D held-out BCE minus training BCE','NN closer to train: equal 10k candidate pools')):
        ax.set(xlabel='Updates (thousands)',title=title); ax.grid(alpha=.2)
    axes[0].axhline(.5,color='gray',ls=':'); axes[2].axhline(.5,color='gray',ls=':'); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root/'integrity_comparison.png',dpi=160); plt.close(fig)
    panels=[]
    for arm in ARMS:
        folder=(baseline if arm in ARMS[:3] else root)/f'{arm}_seed0'
        im=Image.open(folder/'class_grid_100000.png').convert('RGB')
        panel=Image.new('RGB',(im.width,im.height+38),'white'); panel.paste(im,(0,38))
        ImageDraw.Draw(panel).text((8,10),LABELS[arm],fill='black')
        panels.append(panel)
    canvas=Image.new('RGB',(sum(im.width for im in panels)+12*(len(panels)-1),max(im.height for im in panels)),'#eeeeee')
    x=0
    for im in panels: canvas.paste(im,(x,0)); x+=im.width+12
    canvas.save(root/'class_grids_100k.png')
    lines=['# CIFAR-10: larger D batch and wider D','',
           'Two new paired trajectories, seed 0; reuse the three integrity-study baselines. All arms run 100k D and 100k G optimizer updates. G architecture and update batch (128) are unchanged. New paired arms change only D batch or D width, with the BatchNorm side effect documented below.','',
           '[Protocol](../../docs/CIFAR10_CAPACITY.md) · [Baseline report](../cifar10-integrity-v1/REPORT.md) · [All metrics](metrics.csv)','',
           '![Learning curves](learning_curves.png)','','## Endpoints','',
           '| Arm | Steps | Held-out FID ↓ | KID ×1000 ↓ | Precision ↑ | Recall ↑ |','|---|---:|---:|---:|---:|---:|']
    for step in (10000,50000,100000):
        for arm in ARMS:
            r=next(r for r in records[arm] if r['step']==step)['heldout_metrics']
            lines.append(f"| {LABELS[arm]} | {step//1000}k | {r[expected_keys[0]]:.3f} | {r[expected_keys[1]]*1000:.3f} | {r['precision']:.4f} | {r['recall']:.4f} |")
    lines+=['','## Budget and integrity','',
            '| Arm | D batch | D parameters | G parameters | Recorded training min | Actual GPU |','|---|---:|---:|---:|---:|---|']
    for arm in ARMS:
        m=metadata[arm]; v=verifications[arm]
        db=m['config'].get('discriminator_batch_size',128)
        lines.append(f"| {LABELS[arm]} | {db} {'pairs' if arm not in ('vanilla','rsgan') else 'real + 128 fake'} | {m['parameters_d']:,} | {m['parameters_g']:,} | {v['training_seconds']/60:.2f} | {m['gpu']} |")
    lines+=['','D batch256 consumes 256 real and 256 fake examples for D, but only 128 fresh generated examples and 128 references for the G update. D width128 keeps both update batches at 128. G uses the same initial weights in both arms; batch256 also preserves initial D weights. Wider D projection dimension and score scale change along with capacity.','',
            'The batch256 arm generates D fakes in two no-gradient batches of 128, keeping G forward batch size unchanged. G BatchNorm running statistics consequently update three times per iteration instead of two; there is still only one G optimizer update. The larger D batch also advances the shared slot RNG differently. These are not matched-total-compute experiments.','',
            '![Runtime and differences](runtime_and_deltas.png)','','![Integrity comparison](integrity_comparison.png)','',
            '| Arm at 100k | D train BCE | D held-out BCE | D train acceptance | D held-out acceptance | Membership AUC | Feature NN closer to train | Pixel NN closer to train | Exact copies |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        r=records[arm][-1]; d=r['discriminator']; n=r['nearest_neighbors']
        lines.append(f"| {LABELS[arm]} | {d['train_bce']:.3f} | {d['heldout_bce']:.3f} | {d['train_real_acceptance']:.3f} | {d['heldout_real_acceptance']:.3f} | {d['train_membership_auc_from_margin']:.3f} | {n['inception']['fraction_closer_to_train_equal_pools']:.3f} | {n['pixel_rms']['fraction_closer_to_train_equal_pools']:.3f} | {n['exact_generated_training_matches']} |")
    lines+=['','D scores and acceptance tasks are not identically scaled across methods. AUC measures ranking of training versus held-out examples, not calibrated membership-attack accuracy. NN fractions use equal candidate pools of 10,000. Full-train 50k neighbors are retained for copy inspection. No exact copies does not exclude approximate memorization.','',
            '## Final class grids','','![Class grids](class_grids_100k.png)','',
            'Rows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Eight fixed noise vectors repeat across rows and arms. Requested-label rows are not an independent measurement of semantic class accuracy.','',
            '## Final nearest-neighbor panels','','Each panel shows generated queries with their nearest training and held-out images. Candidate pools in these inspection panels are train50k and held-out10k; the preference statistics above use equal 10k pools.','','- D batch256: [Inception neighbors](d_batch256_seed0/nn_inception_100000.png), [pixel neighbors](d_batch256_seed0/nn_pixel_rms_100000.png).','- D width128: [Inception neighbors](d_width128_seed0/nn_inception_100000.png), [pixel neighbors](d_width128_seed0/nn_pixel_rms_100000.png).','','## All held-out checkpoints','','| Arm | Updates | FID | KID ×1000 | Precision | Recall |','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for r in records[arm]:
            m=r['heldout_metrics']
            lines.append(f"| {LABELS[arm]} | {r['step']} | {m[expected_keys[0]]:.3f} | {m[expected_keys[1]]*1000:.3f} | {m['precision']:.4f} | {m['recall']:.4f} |")
    lines+=['','## Reproducibility and limits','','58 local tests passed, including actual G/D forward batch assertions, exact baseline replay, and resume. CUDA checks establish exact resume for both arms and exact unchanged-setting replay. All 50 evaluation records use identical references; the new evaluator differs only in its model-factory import. See `verification.json` and `cuda_resume_verification.json`.','',
            'Both training runs used A10 GPUs. The final wider-D evaluation allowed fallback GPU types after allocation stalled and was ultimately assigned NVIDIA A10 too; evaluator, weights, references and seeds were unchanged. See its evaluation_runtime_100000.json. A warm-container volume-view issue was corrected with an explicit refresh before the final evaluation.','','One seed, no learning-rate retuning, and no confidence interval over training runs. KID subset standard deviation is not a training-run uncertainty estimate. Official test images are excluded from training but repeatedly used for diagnostics. Late checkpoints are correlated; observed minimum FID is descriptive, not an independently validated selection. Historical files and baseline checkpoints are unchanged.','']
    findings = [
        '## Findings', '',
        'Doubling only D batch gives a modest final gain: FID 41.05 versus paired baseline 42.53. Its early advantage is inconsistent; 20k and 50k are worse than baseline.', '',
        'Wider D starts worse at 10k, then improves the later trajectory. At 100k it reaches FID 38.72, nearly matching RSGAN (38.74) and approaching vanilla (38.44). Its KID, precision and recall also closely match RSGAN. This supports capacity as a contributing limitation in this setup, without establishing that it explains the entire optimization gap.', '',
        'The gains cost more: recorded training time is 31.7 minutes for batch256 and 45.8 for wider D, versus 21.8 for paired baseline, 26.4 for vanilla and 27.9 for RSGAN. Wider D is about four times the original D parameters. No equal-compute advantage is established.', '',
        'D train/held-out separation grows: final membership-ranking AUC is 0.655 for paired baseline, 0.681 for batch256, and 0.716 for wider D. Better G metrics coexist with more D overfitting. Across all 20 new evaluations, zero exact generated training-image matches were found; approximate memorization is not excluded.', '',
        'These are single-seed interventions with fixed learning rates. Projection dimension and score scale change when widening, and extra no-gradient G forwards update running BatchNorm statistics in the batch256 arm. More seeds and architectural controls would be needed to establish a general mechanism.', '',
    ]
    text = '\n'.join(lines).replace('## Endpoints', '\n'.join(findings) + '\n## Endpoints')
    (root/'REPORT.md').write_text(text)
    return rows

if __name__=='__main__':
    report()
