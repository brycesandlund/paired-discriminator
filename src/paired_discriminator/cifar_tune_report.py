"""Collect lightweight tuning evidence and summarize every evaluated candidate."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def collect():
    import modal
    v=modal.Volume.from_name('paired-discriminator-cifar-tuning')
    target=ROOT/'results/cifar-tune-v1';target.mkdir(parents=True,exist_ok=True)
    for e in v.iterdir('/cifar-tune-v1',recursive=True):
        p=Path(e.path)
        if p.suffix not in ['.json','.png','.py']:continue
        rel=p.relative_to('cifar-tune-v1') if not p.is_absolute() else p.relative_to('/cifar-tune-v1')
        out=target/rel;out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes(b''.join(v.read_file(e.path)))

def build():
    root=ROOT/'results/cifar-tune-v1';rows=[]
    for p in root.rglob('metrics.json'):
        r=json.loads(p.read_text())
        if 'signature' not in r:continue
        q=r['signature'];rows.append(dict(path=str(p.relative_to(root)),method=q['method'],seed=q['seed'],weights=q['weights'],
            step=r['step'],base_step=r['base_step'],train_fid=r['train_metrics']['frechet_inception_distance'],
            fid=r['heldout_metrics']['frechet_inception_distance'],kid=r['heldout_metrics']['kernel_inception_distance_mean'],
            precision=r['heldout_metrics']['precision'],recall=r['heldout_metrics']['recall'],
            vgg_tv=r['classifiers']['cifar10_vgg16_bn']['class_tv'],resnet_tv=r['classifiers']['cifar10_resnet56']['class_tv'],
            vgg_acceptance=r['classifiers']['cifar10_vgg16_bn']['confidence']['0.9']['acceptance'],
            replay=r['first_batch_replay_exact']))
    rows.sort(key=lambda r:(r['method'],r['train_fid']))
    lines=['# CIFAR optimizer and EMA tuning (#37)','','Exploratory tuning, unconditional and uniform. '
        'Warm branches start from GAN50k or flow80k with Adam and explicit sampling streams reset identically, including controls. '
        'Raw and EMA weights evaluated with the same10k noise samples. Primary screening criterion: training-reference FID. '
        'Test-reference scores are descriptive, not an untouched final test. Classifier TV and recall are reported alongside fidelity.','',
        '## Every evaluated candidate','','| Arm / step | Weights | Train FID ↓ | Test FID ↓ | KID ↓ | Precision | Recall | VGG TV ↓ | ResNet TV ↓ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['path'].removesuffix('/metrics.json')} | {r['weights']} | {r['train_fid']:.2f} | {r['fid']:.2f} | {r['kid']:.5f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['vgg_tv']:.3f} | {r['resnet_tv']:.3f} |")
    (root/'all_metrics.json').write_text(json.dumps(rows,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Evaluations:',len(rows))
    for m in ['vanilla','paired','flow']:
        group=[r for r in rows if r['method']==m]
        if group:print(m,json.dumps(group[0]))

if __name__=='__main__':
    import sys
    if '--collect' in sys.argv:collect()
    build()
