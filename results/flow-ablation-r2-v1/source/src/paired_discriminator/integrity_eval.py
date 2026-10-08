"""Held-out metrics, discriminator generalization, and nearest-neighbor audits."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F
from paired_discriminator.cifar_integrity import configure, models, pair_inputs, atomic_json

VERSION = 'integrity-v1'
SEED = 99173


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def balanced_ids(labels, per_class=1000, seed=SEED+10):
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == y))[:per_class] for y in range(10)]
    if any(len(g) != per_class for g in groups):
        raise ValueError('Insufficient examples in class')
    return np.stack(groups, axis=1).reshape(-1)


def extract(images, model, device):
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), 128):
            x = torch.as_tensor(np.asarray(images[start:start+128]).copy(), device=device)
            result.append(model(x)[0].cpu())
    return torch.cat(result)


def extractor(device):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    return FeatureExtractorInceptionV3('inception-v3-compat', ['2048']).to(device).eval()


def metrics(fake_features, real_features, device='cuda'):
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric
    from torch_fidelity.metric_prc import prc_features_to_metric
    result = fid_statistics_to_metric(fid_features_to_statistics(fake_features), fid_features_to_statistics(real_features), False)
    result.update(kid_features_to_metric(fake_features, real_features, kid_subsets=100, kid_subset_size=1000, rng_seed=SEED, verbose=False))
    result.update(prc_features_to_metric(fake_features.to(device), real_features.to(device), prc_neighborhood=3, prc_batch_size=512, save_cpu_ram=True, verbose=False))
    return {k: float(v) for k,v in result.items()}


def nearest(query, reference, device='cuda', batch=128):
    """Exact Euclidean nearest neighbors; chunk queries, retain original ref IDs."""
    reference = torch.as_tensor(reference, dtype=torch.float32, device=device)
    rn = reference.square().sum(1)
    indices, distances = [], []
    for start in range(0,len(query),batch):
        q = torch.as_tensor(query[start:start+batch], dtype=torch.float32, device=device)
        sq = (q.square().sum(1)[:,None] + rn[None,:] - 2*q@reference.T).clamp_min_(0)
        idx = sq.argmin(1)
        # Recompute selected distances directly to avoid cancellation near duplicates.
        dist = (q-reference[idx]).square().sum(1).sqrt()
        indices.append(idx.cpu().numpy()); distances.append(dist.cpu().numpy())
    return np.concatenate(indices), np.concatenate(distances)


def summary(values):
    return {'mean':float(np.mean(values)), 'median':float(np.median(values)),
            'p01':float(np.quantile(values,.01)), 'p05':float(np.quantile(values,.05)),
            'p95':float(np.quantile(values,.95)), 'min':float(np.min(values))}


def prepare(root, device='cuda'):
    from torchvision.datasets import CIFAR10
    configure(device)
    root=Path(root); cache=root/VERSION; cache.mkdir(exist_ok=True)
    manifest=cache/'manifest.json'
    if manifest.exists():
        return json.loads(manifest.read_text())
    original=CIFAR10(str(root),train=False,download=True)
    test=original.data.transpose(0,3,1,2).copy(); test_y=np.asarray(original.targets,dtype=np.int64)
    train=np.load(root/'train_uint8.npy'); train_y=np.load(root/'train_labels.npy')
    assert len(train)==50000 and len(test)==10000
    assert np.array_equal(np.bincount(test_y),np.full(10,1000))
    np.save(cache/'test_uint8.npy',test); np.save(cache/'test_labels.npy',test_y)
    train_ids=balanced_ids(train_y); test_ids=balanced_ids(test_y)
    metric_ids=torch.randperm(len(train),generator=torch.Generator().manual_seed(SEED+2))[:10000].numpy()
    np.savez(cache/'reference_ids.npz',train_balanced=train_ids,test_balanced=test_ids,train_metrics=metric_ids)
    ext=extractor(device)
    tf=extract(train,ext,device); hf=extract(test,ext,device)
    torch.save(tf,cache/'train_features.pt'); torch.save(hf,cache/'test_features.pt')
    # Calibrate real-vs-real distribution metrics, with the same pool sizes.
    real_baseline=metrics(tf[metric_ids],hf,device)
    baseline={}
    for space,a,b in [('inception',hf,tf[train_ids]),('pixel_rms',torch.from_numpy(test).flatten(1).float()/255/(3072**.5),torch.from_numpy(train[train_ids]).flatten(1).float()/255/(3072**.5))]:
        idx,dist=nearest(a,b,device)
        baseline[space]=summary(dist)
    train_hashes={hashlib.sha256(x.tobytes()).digest() for x in train}
    overlap=sum(hashlib.sha256(x.tobytes()).digest() in train_hashes for x in test)
    result={'version':VERSION,'train_images':50000,'heldout_images':10000,'heldout_split':'official CIFAR10 test; never used in GAN updates',
            'train_sha256':digest(train),'heldout_sha256':digest(test),'train_labels_sha256':digest(train_y),'heldout_labels_sha256':digest(test_y),
            'exact_test_images_also_in_train':overlap,'train_reference_ids_sha256':digest(metric_ids),
            'balanced_train_ids_sha256':digest(train_ids),'balanced_test_ids_sha256':digest(test_ids),
            'real_train10k_vs_test10k_metrics':real_baseline,'real_test_to_train10k_nearest_baseline':baseline,
            'feature_extractor':'torch-fidelity inception-v3-compat 2048; raw uint8 preprocessing', 'torch':torch.__version__}
    atomic_json(manifest,result)
    return result


def discriminator_audit(d, fake_float, train_real, test_real, method, device):
    arrays={}
    for name, real in [('train',train_real),('heldout',test_real)]:
        scores=[]; losses=[]; correct=[]
        with torch.inference_mode():
            for start in range(0,len(real),128):
                r=torch.as_tensor(real[start:start+128].copy(),device=device).float()/127.5-1
                f=fake_float[start:start+len(r)].to(device)
                y=torch.arange(start,start+len(r),device=device)%10
                if method=='vanilla':
                    margin=d(r,y)
                    loss=F.softplus(-margin)
                    acc=(margin>0).float()
                elif method=='rsgan':
                    margin=d(r,y)-d(f,y)
                    loss=F.softplus(-margin); acc=(margin>0).float()
                else:
                    slots=torch.ones(len(r),dtype=torch.bool,device=device)
                    a=d(pair_inputs(r,f,slots),y)
                    b=-d(pair_inputs(r,f,~slots),y)
                    margin=(a+b)/2
                    loss=(F.softplus(-a)+F.softplus(-b))/2
                    acc=((a>0).float()+(b>0).float())/2
                scores.append(margin.cpu().numpy()); losses.append(loss.cpu().numpy()); correct.append(acc.cpu().numpy())
        arrays[name+'_margin']=np.concatenate(scores)
        arrays[name+'_loss']=np.concatenate(losses)
        arrays[name+'_accuracy']=np.concatenate(correct)
    n=len(train_real)
    scores=np.r_[arrays['train_margin'],arrays['heldout_margin']]
    _,inverse,counts=np.unique(scores,return_inverse=True,return_counts=True)
    ends=np.cumsum(counts)
    ranks=(ends-(counts-1)/2)[inverse]
    auc=float((ranks[:n].sum()-n*(n+1)/2)/(n*n))
    gap=arrays['train_margin']-arrays['heldout_margin']
    result={'n_per_split':n,'higher_margin_means_real_accepted':True,'paired_averages_both_orientations':method=='paired',
            'train_margin':summary(arrays['train_margin']),'heldout_margin':summary(arrays['heldout_margin']),
            'train_bce':float(arrays['train_loss'].mean()),'heldout_bce':float(arrays['heldout_loss'].mean()),
            'train_real_acceptance':float(arrays['train_accuracy'].mean()),'heldout_real_acceptance':float(arrays['heldout_accuracy'].mean()),
            'train_minus_heldout_margin':float(gap.mean()),'train_membership_auc_from_margin':auc,
            'per_class':{str(y):{'train_margin':float(arrays['train_margin'][y::10].mean()),'heldout_margin':float(arrays['heldout_margin'][y::10].mean())} for y in range(10)}}
    if method=='vanilla':
        with torch.inference_mode():
            f_logits=torch.cat([d(fake_float[i:i+128].to(device),torch.arange(i,min(i+128,n),device=device)%10).cpu() for i in range(0,n,128)])
        result['shared_fake_rejection']=float((f_logits<0).float().mean())
        result['shared_fake_bce']=float(F.softplus(f_logits).mean())
    return result,arrays


def nn_audit(fake, features, train, test, train_features, test_features, train_ids, output, step, device):
    from torchvision.utils import save_image
    arrays={}; result={}
    for space,q,tr,te in [('inception',features,train_features,test_features),
            ('pixel_rms',torch.from_numpy(fake).flatten(1).float()/255/(3072**.5),torch.from_numpy(train).flatten(1).float()/255/(3072**.5),torch.from_numpy(test).flatten(1).float()/255/(3072**.5))]:
        result[space]={}
        for name,ref in [('train50k',tr),('train10k',tr[train_ids]),('heldout10k',te)]:
            idx,dist=nearest(q,ref,device)
            if name=='train10k': idx=train_ids[idx]
            arrays[space+'_'+name+'_index']=idx; arrays[space+'_'+name+'_distance']=dist
            result[space][name]=summary(dist)
        a=arrays[space+'_train10k_distance']; b=arrays[space+'_heldout10k_distance']
        result[space]['fraction_closer_to_train_equal_pools']=float((a<b).mean())
        # 8 fixed examples and 8 closest-to-training examples; always show heldout neighbor too.
        selected=np.r_[np.arange(8),np.argsort(arrays[space+'_train50k_distance'])[:8]]
        rows=[]
        for i in selected:
            rows.extend([fake[i],train[arrays[space+'_train50k_index'][i]],test[arrays[space+'_heldout10k_index'][i]]])
        save_image(torch.from_numpy(np.stack(rows)).float()/255,output/f'nn_{space}_{step:06d}.png',nrow=3,padding=2)
        result[space]['grid_generated_indices']=selected.tolist()
    train_hashes={hashlib.sha256(x.tobytes()).digest() for x in train}
    result['exact_generated_training_matches']=sum(hashlib.sha256(x.tobytes()).digest() in train_hashes for x in fake)
    result['unique_generated_uint8_images']=len({hashlib.sha256(x.tobytes()).digest() for x in fake})
    np.savez_compressed(output/f'nearest_{step:06d}.npz',**arrays)
    return result


def evaluate(run_dir,data_root,step,device='cuda',commit=None):
    configure(device)
    import torch_fidelity
    run_dir=Path(run_dir); root=Path(data_root); cache=root/VERSION
    source_sha=digest(np.frombuffer(Path(__file__).read_bytes(),dtype=np.uint8))
    dest=run_dir/f'integrity_{step:06d}.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['evaluation_source_sha256']!=source_sha: raise ValueError('Evaluation source changed; use a new output/version')
        return old
    began=time.perf_counter()
    manifest=json.loads((cache/'manifest.json').read_text())
    ckpt=torch.load(run_dir/f'generator_{step:06d}.pt',map_location='cpu',weights_only=False)
    g,d=models(ckpt['config'],ckpt['method'],ckpt['seed'])
    g.load_state_dict(ckpt['generator']); d.load_state_dict(ckpt['discriminator'])
    g.to(device).eval(); d.to(device).eval()
    n=10000; rng=torch.Generator().manual_seed(SEED+1)
    floats=[]
    with torch.inference_mode():
        for start in range(0,n,128):
            z=torch.randn(min(128,n-start),ckpt['config']['latent_dim'],generator=rng).to(device)
            y=torch.arange(start,start+len(z),device=device)%10
            floats.append(g(z,y).cpu())
    fake_float=torch.cat(floats); fake=((fake_float+1)*127.5).round().clamp(0,255).to(torch.uint8).numpy()
    train=np.load(root/'train_uint8.npy'); test=np.load(cache/'test_uint8.npy')
    ids=np.load(cache/'reference_ids.npz'); train_ids=ids['train_balanced']; test_ids=ids['test_balanced']
    tf=torch.load(cache/'train_features.pt',weights_only=True); hf=torch.load(cache/'test_features.pt',weights_only=True)
    ff=extract(fake,extractor(device),device)
    train_metrics=metrics(ff,tf[ids['train_metrics']],device); held_metrics=metrics(ff,hf,device)
    disc,da=discriminator_audit(d,fake_float,train[train_ids],test[test_ids],ckpt['method'],device)
    np.savez_compressed(run_dir/f'discriminator_{step:06d}.npz',**da)
    neighbors=nn_audit(fake,ff,train,test,tf,hf,train_ids,run_dir,step,device)
    result={'method':ckpt['method'],'seed':ckpt['seed'],'step':step,'samples_generated':n,'reference_sizes':{'train':10000,'heldout':10000},
            'train_metrics':train_metrics,'heldout_metrics':held_metrics,'discriminator':disc,'nearest_neighbors':neighbors,
            'generated_uint8_sha256':digest(fake),'evaluation_source_sha256':source_sha,'training_source_sha256':ckpt['signature']['source_sha256'],
            'reference_manifest':manifest,'torch_fidelity':torch_fidelity.__version__,'evaluation_seconds':time.perf_counter()-began}
    (run_dir/'evaluation_source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(dest,result)
    if commit: commit()
    print(json.dumps({'method':ckpt['method'],'step':step,'train_fid':train_metrics['frechet_inception_distance'],'heldout_fid':held_metrics['frechet_inception_distance'],'d_membership_auc':disc['train_membership_auc_from_margin'],'evaluation_seconds':result['evaluation_seconds']}),flush=True)
    return result
