"""Paired CIFAR discriminator interaction ablations (experiment 38)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from . import cifar, cifar_tune

ARCHITECTURES = ('concat', 'concat_wide', 'shared_difference',
                 'global_context', 'self_attention', 'cross_attention')
SOURCES = ('cifar_pair_arch', 'cifar', 'cifar_tune', 'image_flow')


class Interaction(nn.Module):
    """Equal parameter count for global conditioning, self- and cross-attention."""
    def __init__(self, kind, channels=128):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(channels)
        if kind == 'global_context':
            self.own = nn.Linear(channels, channels)
            self.context = nn.Linear(channels, channels)
        else:
            self.query = nn.Linear(channels, channels // 2)
            self.key = nn.Linear(channels, channels // 2)
            self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, own, other):
        # 64 spatial tokens per input at 8x8; no cross-example normalization.
        batch, channels, height, width = own.shape
        a = self.norm(own.flatten(2).transpose(1, 2))
        b = self.norm(other.flatten(2).transpose(1, 2))
        if self.kind == 'global_context':
            mixed = torch.nn.functional.leaky_relu(self.own(a) + self.context(b.mean(1, keepdim=True)), .2)
        else:
            if self.kind == 'self_attention':
                b = a
            heads = 4
            def split(x):
                return x.reshape(batch, -1, heads, x.shape[-1] // heads).transpose(1, 2)
            q, k, v = split(self.query(a)), split(self.key(b)), split(self.value(b))
            weights = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)
            mixed = (weights @ v).transpose(1, 2).reshape(batch, height * width, channels)
        residual = self.output(mixed).transpose(1, 2).reshape_as(own)
        return own + self.scale * residual


class SharedDiscriminator(nn.Module):
    """Score(A | B) - score(B | A), so swapping slots negates the logit."""
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(.2))
        self.interaction = None if architecture == 'shared_difference' else Interaction(architecture)
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(256, 1, 4, 1, 0))

    def forward(self, inputs):
        a, b = inputs.chunk(2, dim=1)
        n = len(a)
        features = self.encoder(torch.cat((a, b)))
        if self.interaction is not None:
            other = torch.cat((features[n:], features[:n]))
            features = self.interaction(features, other)
        scores = self.head(features).flatten()
        return scores[:n] - scores[n:]


def initialize(module):
    cifar.initialize(module)
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, .02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def models(architecture, seed):
    if architecture not in ARCHITECTURES:
        raise ValueError(architecture)
    g, original_d = cifar.models({'latent_dim': 128, 'width': 64}, 'paired', seed)
    if architecture == 'concat':
        return g, original_d
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 10000)
        if architecture == 'concat_wide':
            d = cifar.Discriminator(6, 66)
        else:
            d = SharedDiscriminator(architecture)
        d.apply(initialize)
        if isinstance(d, SharedDiscriminator):
            # Keep the shared CNN's initial tensors identical across interaction arms.
            _, unary = cifar.models({'latent_dim': 128, 'width': 64}, 'vanilla', seed)
            for target, source in ((d.encoder[0], unary.net[0]), (d.encoder[2], unary.net[2]),
                                   (d.head[0], unary.net[4]), (d.head[2], unary.net[6])):
                target.load_state_dict(source.state_dict())
    return g, d


def train(architecture, seed, steps, output, data, device='cuda', batch=128, commit=None):
    if steps < 1 or batch < 2 or batch % 2:
        raise ValueError('Positive steps and an even batch of at least two are required')
    cifar.configure(device)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    sources = {name: cifar_tune.digest(Path(__file__).with_name(name + '.py')) for name in SOURCES}
    signature = dict(architecture=architecture, seed=seed, batch=batch, lr=2e-4, betas=[.5, .999],
                     ema_decay=.999, source_hashes=sources, device=device, torch=torch.__version__,
                     data_sha256=hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest())
    g, d = models(architecture, seed); g.to(device); d.to(device)
    ema = copy.deepcopy(g).requires_grad_(False)
    og = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(.5, .999))
    od = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(.5, .999))
    rng = cifar.rngs_for(seed)
    start = 0; seconds = 0.; rows = []
    if (out / 'latest.pt').exists():
        ck = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        assert ck['signature'] == signature, 'Resume signature changed'
        for model, key in ((g, 'model'), (d, 'discriminator'), (ema, 'ema')):
            model.load_state_dict(ck[key])
        og.load_state_dict(ck['optimizer_g']); od.load_state_dict(ck['optimizer_d'])
        for k, v in rng.items(): v.set_state(ck['rngs'][k])
        start, seconds, rows = ck['step'], ck['training_seconds'], ck['rows']
    if start > steps: raise ValueError('Requested endpoint precedes existing progress')
    cifar.atomic_json(out / 'provenance.json', dict(**signature,
        parameters_g=sum(p.numel() for p in g.parameters()), parameters_d=sum(p.numel() for p in d.parameters()),
        gpu=torch.cuda.get_device_name() if device == 'cuda' else 'cpu'))
    (out / 'training_source.py').write_bytes(Path(__file__).read_bytes())
    data = data.to(device)
    def draw(kind):
        indices = torch.randint(len(data), (batch,), generator=rng['real_' + kind]).to(device)
        real = data[indices].float() / 127.5 - 1
        noise = torch.randn(batch, 128, generator=rng['noise_' + kind]).to(device)
        slots = (torch.randperm(batch, generator=rng['slots']) < batch // 2).to(device)
        return real, noise, slots
    def sync():
        if device == 'cuda': torch.cuda.synchronize()
    sync(); began = time.perf_counter()
    for step in range(start + 1, steps + 1):
        g.train(); d.requires_grad_(True)
        real, noise, slots = draw('d')
        with torch.no_grad(): fake = g(noise)
        od.zero_grad(set_to_none=True)
        dl = cifar.loss_d(d, real, fake, 'paired', slots); dl.backward(); od.step()
        d.requires_grad_(False)
        real, noise, slots = draw('g')
        og.zero_grad(set_to_none=True)
        gl = cifar.loss_g(d, real, g(noise), 'paired', slots); gl.backward(); og.step()
        cifar_tune.update_ema(ema, g)
        if step % 1000 == 0 or step == steps:
            sync(); seconds += time.perf_counter() - began
            assert torch.isfinite(dl) and torch.isfinite(gl), 'Nonfinite training loss'
            rows.append(dict(step=step, d_loss=dl.item(), g_loss=gl.item(), training_seconds=seconds))
            ck = dict(signature=signature, step=step, base_step=0, model=g.state_dict(), ema=ema.state_dict(),
                      discriminator=d.state_dict(), optimizer_g=og.state_dict(), optimizer_d=od.state_dict(),
                      rngs={k: v.get_state() for k, v in rng.items()}, rows=rows, training_seconds=seconds)
            torch.save(ck, out / 'latest.pt.tmp'); (out / 'latest.pt.tmp').replace(out / 'latest.pt')
            if step % 10000 == 0 or step == steps: torch.save(ck, out / f'checkpoint_{step:06d}.pt')
            cifar.atomic_json(out / 'training.json', rows)
            cifar.atomic_json(out / 'status.json', dict(phase='training', step=step, training_seconds=seconds))
            print(architecture, seed, step, round(seconds, 2), flush=True)
            if commit: commit()
            sync(); began = time.perf_counter()
    return ck


def diagnostics(checkpoint, architecture, seed, data, device='cuda', n=128):
    """Reference effects on logits and *directions*, not the BCE scalar multiplier."""
    cifar.configure(device)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    g, d = models(architecture, seed)
    g.load_state_dict(ck['model']); d.load_state_dict(ck['discriminator'])
    g.to(device).eval(); d.to(device).eval().requires_grad_(False)
    rng = torch.Generator().manual_seed(99574)
    z = torch.randn(n, 128, generator=rng).to(device)
    ids = torch.randint(len(data), (n,), generator=rng)
    real = data[ids].to(device).float() / 127.5 - 1
    with torch.no_grad(): fake = g(z)
    del g
    def score(a, b): return d(torch.cat((a, b), dim=1))
    def gradient(ref):
        x = fake.detach().requires_grad_(True)
        s = score(x, ref)
        return s.detach(), torch.autograd.grad(s.sum(), x)[0].flatten(1)
    s, ga = gradient(real); _, gb = gradient(real.roll(1, 0))
    norm_a, norm_b = ga.norm(dim=1), gb.norm(dim=1)
    valid = (norm_a > 1e-12) & (norm_b > 1e-12)
    cosine = torch.nn.functional.cosine_similarity(ga[valid], gb[valid], dim=1)
    with torch.no_grad():
        reference_shift = score(fake, real.roll(1, 0)) - s
        interaction = s - score(fake, real.roll(1, 0)) - score(fake.roll(1, 0), real) + score(fake.roll(1, 0), real.roll(1, 0))
        swapped = score(real, fake)
        result = dict(samples=n, gradient_direction_cosine=float(cosine.mean()) if len(cosine) else None,
                      nonzero_gradient_fraction=float(valid.float().mean()),
                      reference_logit_shift_rms=float(reference_shift.square().mean().sqrt()),
                      interaction_residual_rms=float(interaction.square().mean().sqrt()),
                      logit_rms=float(s.square().mean().sqrt()),
                      swap_antisymmetry_error=float((s + swapped).abs().max()),
                      gradient_norm_mean=float(norm_a.mean()),
                      note='Raw G and D; changing a reference in an additive unary score changes BCE scaling but not this raw-logit gradient direction. Reference dependence does not prove useful coverage.')
    if isinstance(d, SharedDiscriminator) and d.interaction is not None:
        result['interaction_scale'] = float(d.interaction.scale)
    return result


def state_equal(a, b):
    if isinstance(a, dict): return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
