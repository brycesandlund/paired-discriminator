"""Frozen digit-classifier diagnostics; confidence is not a validity oracle."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from paired_discriminator.mnist import configure, atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7,128), nn.ReLU(), nn.Dropout(.25), nn.Linear(128,10))

    def forward(self, x):
        return self.net((x-.1307)/.3081)


@torch.inference_mode()
def predict(model, images, device):
    probs=[]
    for x in images.split(256):
        probs.append(model(x.to(device).float()/255).softmax(1).cpu())
    return torch.cat(probs)


def distribution_metrics(probabilities, target, threshold=.9, minimum=.01):
    probabilities=np.asarray(probabilities)
    target=np.asarray(target,dtype=np.float64)
    assert probabilities.ndim==2 and probabilities.shape[1]==10
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1,atol=1e-5)
    assert np.all(target>=0) and np.isclose(target.sum(),1)
    labels=probabilities.argmax(1); confidence=probabilities.max(1)
    counts=np.bincount(labels,minlength=10); mass=counts/len(labels)
    accepted=confidence>=threshold
    accepted_counts=np.bincount(labels[accepted],minlength=10)
    accepted_mass=accepted_counts/len(labels)
    rejected=1-accepted.mean()
    return {'samples':len(labels),'digit_counts':counts.tolist(),'digit_mass':mass.tolist(),
            'covered_digits':int((mass>=minimum).sum()),'observed_digits':int((counts>0).sum()),
            'mode_tv':float(np.abs(mass-target).sum()/2),
            'confidence_threshold':threshold,'coverage_min_mass':minimum,
            'confidence_accepted_fraction':float(accepted.mean()),
            'accepted_digit_counts':accepted_counts.tolist(),'accepted_digit_mass':accepted_mass.tolist(),
            'confidence_covered_digits':int((accepted_mass>=minimum).sum()),
            'confidence_augmented_tv':float((np.abs(accepted_mass-target).sum()+rejected)/2),
            'mean_confidence':float(confidence.mean()),
            'confidence_quantiles':np.quantile(confidence,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),
            'confidence_histogram':np.histogram(confidence,bins=np.linspace(0,1,21))[0].tolist()}


def prepare_classifier(root, device='cuda'):
    configure(device);root=Path(root);out=root/'classifier-v1';out.mkdir(exist_ok=True)
    source=sha(__file__)
    if (out/'classifier.pt').exists() and (out/'report.json').exists():
        meta=json.loads((out/'report.json').read_text())
        if meta['source_sha256']!=source:raise ValueError('Classifier/evaluation source changed')
        if meta['checkpoint_sha256']!=sha(out/'classifier.pt'):raise ValueError('Classifier checkpoint changed')
        return meta
    train=torch.from_numpy(np.load(root/'train_uint8.npy'));labels=torch.from_numpy(np.load(root/'train_labels.npy'))
    test=torch.from_numpy(np.load(root/'test_uint8.npy'));test_y=torch.from_numpy(np.load(root/'test_labels.npy'))
    ids=torch.randperm(len(train),generator=torch.Generator().manual_seed(7321))
    train_ids,val_ids=ids[:50000],ids[50000:]
    np.savez_compressed(out/'split_ids.npz',train=train_ids.numpy(),validation=val_ids.numpy())
    torch.manual_seed(7322)
    model=DigitClassifier().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    x=train[train_ids].to(device);y=labels[train_ids].to(device)
    rng=torch.Generator().manual_seed(7323);history=[];best=-1
    for epoch in range(1,11):
        model.train();order=torch.randperm(len(x),generator=rng)
        for idx in order.split(128):
            idx=idx.to(device);optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x[idx].float()/255),y[idx]);loss.backward();optimizer.step()
        model.eval();p=predict(model,train[val_ids],device)
        accuracy=(p.argmax(1)==labels[val_ids]).float().mean().item()
        row={'epoch':epoch,'validation_accuracy':accuracy,'validation_nll':F.nll_loss(p.clamp_min(1e-12).log(),labels[val_ids]).item()}
        history.append(row);print(json.dumps({'classifier':row}),flush=True)
        if accuracy>best:
            best=accuracy;best_epoch=epoch
            torch.save({'model':model.state_dict(),'epoch':epoch,'source_sha256':source},out/'classifier.pt')
    model.load_state_dict(torch.load(out/'classifier.pt',map_location=device,weights_only=False)['model']);model.eval()
    target=np.bincount(labels.numpy(),minlength=10)/len(labels)
    probs=predict(model,test,device);pred=probs.argmax(1)
    confusion=torch.bincount(test_y*10+pred,minlength=100).reshape(10,10)
    meta={'source_sha256':source,'checkpoint_sha256':sha(out/'classifier.pt'),'training_examples':50000,'validation_examples':10000,
          'selection':'highest validation accuracy; earliest epoch wins ties; official test never used for selection',
          'selected_epoch':best_epoch,'history':history,'test_accuracy':(pred==test_y).float().mean().item(),
          'test_confusion_matrix':confusion.tolist(),'target_digit_mass':target.tolist(),
          'test_diagnostics':distribution_metrics(probs.numpy(),target),
          'dataset':json.loads((root/'dataset.json').read_text()),'gpu':torch.cuda.get_device_name() if device=='cuda' else device,
          'torch':torch.__version__,'caveat':'Classifier probabilities are uncalibrated and can be confidently wrong on generated images.'}
    rng=np.random.default_rng(99177)
    real_tv=[np.abs(np.bincount(rng.choice(labels.numpy(),10000),minlength=10)/10000-target).sum()/2 for _ in range(1000)]
    meta['real_iid_10000_label_tv_reference']={'mean':float(np.mean(real_tv)),'p025':float(np.quantile(real_tv,.025)),'p975':float(np.quantile(real_tv,.975))}
    (out/'classifier_source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(out/'report.json',meta)
    return meta


class Evaluator:
    def __init__(self, root, config, method, seed, device='cuda'):
        self.root=Path(root);self.config=config;self.method=method;self.seed=seed;self.device=device
        folder=self.root/'classifier-v1'
        self.meta=json.loads((folder/'report.json').read_text())
        assert self.meta['checkpoint_sha256']==sha(folder/'classifier.pt')
        assert self.meta['source_sha256']==sha(__file__)
        # Model construction must not perturb training RNGs.
        with torch.random.fork_rng(devices=[]):self.classifier=DigitClassifier()
        self.classifier.load_state_dict(torch.load(folder/'classifier.pt',map_location='cpu',weights_only=False)['model'])
        self.classifier.to(device).eval().requires_grad_(False)
        self.target=np.asarray(self.meta['target_digit_mass'])
        self.signature={'classifier_sha256':self.meta['checkpoint_sha256'],'evaluation_source_sha256':sha(__file__),
                        'target_digit_mass':self.target.tolist()}

    @torch.inference_mode()
    def __call__(self,g,step,output):
        from torchvision.utils import save_image
        began=time.perf_counter();output=Path(output)
        path=output/f'metrics_{step:06d}.json'
        if path.exists():
            r=json.loads(path.read_text())
            if r['evaluation_signature']!=self.signature:raise ValueError('Evaluation changed')
            return r
        mode=g.training;g.eval()
        try:
            rng=torch.Generator().manual_seed(self.config['eval_seed']+1)
            images=[]
            for start in range(0,self.config['eval_samples'],128):
                z=torch.randn(min(128,self.config['eval_samples']-start),self.config['latent_dim'],generator=rng).to(self.device)
                images.append(((g(z)+1)*127.5).round().clamp(0,255).to(torch.uint8).cpu())
            images=torch.cat(images)
            probabilities=predict(self.classifier,images,self.device).numpy()
            result=distribution_metrics(probabilities,self.target,self.config['confidence_threshold'],self.config['coverage_min_mass'])
            result.update(method=self.method,seed=self.seed,step=step,evaluation_signature=self.signature,
                          generated_uint8_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                          evaluation_seconds=time.perf_counter()-began)
            np.savez_compressed(output/f'predictions_{step:06d}.npz',probabilities=probabilities)
            if step in (10000,50000):
                labels=probabilities.argmax(1)
                # First ten in fixed random evaluation order per predicted digit: no best-example sorting.
                grid=torch.zeros(100,1,28,28)
                selected=[]
                for digit in range(10):
                    ids=np.flatnonzero(labels==digit)[:10];selected.append(ids.tolist())
                    if len(ids):grid[digit*10:digit*10+len(ids)]=images[ids].float()/255
                save_image(grid,output/f'by_predicted_digit_{step:06d}.png',nrow=10,padding=2)
                result['predicted_digit_grid_indices']=selected
                np.save(output/f'generated_{step:06d}.npy',images.numpy())
            (output/'evaluation_source.py').write_bytes(Path(__file__).read_bytes())
            atomic_json(path,result)
            print(json.dumps({'evaluation':{k:result[k] for k in ('method','seed','step','mode_tv','covered_digits','confidence_covered_digits','confidence_accepted_fraction')}}),flush=True)
            return result
        finally:g.train(mode)
