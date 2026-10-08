"""Unconditional straight-line flow matching for native MNIST/CIFAR images."""

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mnist import atomic_json, configure


SHAPES = {'mnist': (1, 28, 28), 'cifar10': (3, 32, 32)}


class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.time = nn.Linear(time_dim, channels_out)
        self.norm2 = nn.GroupNorm(8, channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.skip = nn.Identity() if channels_in == channels_out else nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t))[:, :, None, None]
        return (self.skip(x) + self.conv2(F.silu(self.norm2(h)))) / math.sqrt(2)


class Velocity(nn.Module):
    """Three spatial scales, residual blocks, sinusoidal time; no class inputs."""
    def __init__(self, channels, width=32):
        super().__init__()
        w = width
        assert w >= 8 and w % 8 == 0
        self.register_buffer('time_freq', torch.exp(-math.log(10000) * torch.arange(w // 2) / (w // 2 - 1)))
        self.time_net = nn.Sequential(nn.Linear(w, 4*w), nn.SiLU(), nn.Linear(4*w, 4*w))
        self.stem = nn.Conv2d(channels, w, 3, padding=1)
        self.enc0 = Block(w, w, 4*w)
        self.down0 = nn.Conv2d(w, 2*w, 4, stride=2, padding=1)
        self.enc1 = Block(2*w, 2*w, 4*w)
        self.down1 = nn.Conv2d(2*w, 4*w, 4, stride=2, padding=1)
        self.mid0 = Block(4*w, 4*w, 4*w)
        self.mid1 = Block(4*w, 4*w, 4*w)
        self.up1 = nn.Conv2d(4*w, 2*w, 3, padding=1)
        self.dec1 = Block(4*w, 2*w, 4*w)
        self.up0 = nn.Conv2d(2*w, w, 3, padding=1)
        self.dec0 = Block(2*w, w, 4*w)
        self.out_norm = nn.GroupNorm(8, w)
        self.out = nn.Conv2d(w, channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        phase = 1000 * t[:, None] * self.time_freq[None]
        time_embedding = self.time_net(torch.cat((phase.sin(), phase.cos()), dim=1))
        a = self.enc0(self.stem(x), time_embedding)
        b = self.enc1(self.down0(a), time_embedding)
        h = self.mid1(self.mid0(self.down1(b), time_embedding), time_embedding)
        h = self.up1(F.interpolate(h, size=b.shape[-2:], mode='nearest'))
        h = self.dec1(torch.cat((h, b), dim=1), time_embedding)
        h = self.up0(F.interpolate(h, size=a.shape[-2:], mode='nearest'))
        h = self.dec0(torch.cat((h, a), dim=1), time_embedding)
        return self.out(F.silu(self.out_norm(h)))


def make_model(dataset, config, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Velocity(SHAPES[dataset][0], config['width'])


def interpolation(noise, real, t):
    t = t.reshape(-1, 1, 1, 1)
    return (1-t)*noise + t*real, real-noise


@torch.inference_mode()
def sample(model, noise, steps):
    if steps < 1:
        raise ValueError('Positive integration steps required')
    mode = model.training
    model.eval()
    try:
        x = noise.clone()
        dt = 1.0 / steps
        for i in range(steps):
            t = x.new_full((len(x),), i*dt)
            v = model(x, t)
            x = x + dt*model(x + .5*dt*v, t + .5*dt)
        return x
    finally:
        model.train(mode)


def to_uint8(images):
    return ((images + 1)*127.5).round().clamp(0, 255).to(torch.uint8)


def train(config, dataset, seed, steps, output, data, device='cuda', commit=None, on_checkpoint=None):
    configure(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assert data.dtype == torch.uint8 and tuple(data.shape[1:]) == SHAPES[dataset]
    assert steps > 0
    source = Path(__file__).read_bytes()
    signature = dict(config=config, dataset=dataset, seed=seed,
                     source_sha256=hashlib.sha256(source).hexdigest(),
                     data_sha256=hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest(),
                     torch=torch.__version__, device=device,
                     gpu=torch.cuda.get_device_name() if device == 'cuda' else None)
    model = make_model(dataset, config, seed).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=tuple(config['betas']))
    rngs = {name: torch.Generator().manual_seed(seed + offset)
            for name, offset in [('real', 20000), ('noise', 40000), ('time', 60000)]}
    start, training_seconds, rows = 0, 0., []
    latest = output / 'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('Resume source/data/config/runtime mismatch')
        model.load_state_dict(saved['model'])
        ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        for name, rng in rngs.items():
            rng.set_state(saved['rngs'][name])
        start, training_seconds, rows = saved['step'], saved['training_seconds'], saved['rows']
    if start > steps:
        raise ValueError('Requested endpoint predates saved progress')
    atomic_json(output / 'config.json', dict(config=config, dataset=dataset, seed=seed))
    atomic_json(output / 'provenance.json', signature)
    (output / 'training_source.py').write_bytes(source)
    data = data.to(device)
    batch = config['batch_size']
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()
    sync()
    began, last_step = time.perf_counter(), start
    for step in range(start + 1, steps + 1):
        model.train()
        ids = torch.randint(len(data), (batch,), generator=rngs['real']).to(device)
        real = data[ids].float()/127.5 - 1
        noise = torch.randn(batch, *SHAPES[dataset], generator=rngs['noise']).to(device)
        t = torch.rand(batch, generator=rngs['time']).to(device)
        xt, target = interpolation(noise, real, t)
        optimizer.param_groups[0]['lr'] = config['learning_rate'] * min(1., step/max(1, config['warmup_steps']))
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(xt, t), target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for a, b in zip(ema.parameters(), model.parameters()):
                a.lerp_(b, 1-config['ema_decay'])
        log = step % config['log_every'] == 0 or step == steps
        checkpoint = step % config['checkpoint_every'] == 0 or step == steps
        if log or checkpoint:
            sync()
            elapsed = time.perf_counter() - began
            training_seconds += elapsed
            value = float(loss)
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite flow loss')
            rows.append(dict(step=step, loss=value, training_seconds=training_seconds,
                             seconds_per_step=elapsed/(step-last_step)))
            print(json.dumps(dict(dataset=dataset, seed=seed, **rows[-1])), flush=True)
            if checkpoint:
                state = dict(signature=signature, step=step, model=model.state_dict(), ema=ema.state_dict(),
                             optimizer=optimizer.state_dict(), rngs={k: r.get_state() for k, r in rngs.items()},
                             training_seconds=training_seconds, rows=rows)
                torch.save(state, output / 'latest.pt.tmp')
                (output / 'latest.pt.tmp').replace(latest)
                if step % 10000 == 0 or step == steps:
                    torch.save(dict(signature=signature, step=step, model=model.state_dict(), ema=ema.state_dict()),
                               output / f'velocity_{step:06d}.pt')
                atomic_json(output / 'training.json', rows)
                atomic_json(output / 'status.json', dict(step=step, target_steps=steps, phase='training',
                            training_seconds=training_seconds, real_samples=step*batch,
                            parameters=sum(p.numel() for p in model.parameters())))
                if on_checkpoint:
                    on_checkpoint(model, step, output)
                if commit:
                    commit()
            sync()
            began, last_step = time.perf_counter(), step
    status = dict(step=steps, target_steps=steps, phase='training_complete',
                  training_seconds=training_seconds, real_samples=steps*batch,
                  parameters=sum(p.numel() for p in model.parameters()))
    atomic_json(output / 'status.json', status)
    if commit:
        commit()
    return status
