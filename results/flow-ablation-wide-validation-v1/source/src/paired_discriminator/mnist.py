"""Unconditional ordinary MNIST pilot. Shared loop, isolated RNG streams, exact resume."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ("vanilla", "paired")


def configure(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4 if device == "cuda" else 1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Generator(nn.Module):
    def __init__(self, latent, width):
        super().__init__()
        self.input = nn.Linear(latent, width * 2 * 7 * 7)
        self.net = nn.Sequential(
            nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(),
            nn.ConvTranspose2d(width, 1, 4, 2, 1), nn.Tanh())
        self.width = width

    def forward(self, z):
        return self.net(self.input(z).reshape(len(z), self.width * 2, 7, 7))


class Discriminator(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(width, width * 2, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1), nn.LeakyReLU(.2))
        self.output = nn.Linear(width * 4 * 3 * 3, 1)

    def forward(self, x):
        return self.output(self.features(x).flatten(1)).flatten()


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0, .02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, .02)
        nn.init.zeros_(module.bias)


def models(config, method, seed):
    if method not in METHODS:
        raise ValueError(method)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        g = Generator(config["latent_dim"], config["width"])
        g.apply(initialize)
        torch.manual_seed(seed + 10000)
        d = Discriminator(2 if method == "paired" else 1, config["width"])
        d.apply(initialize)
    return g, d


def pair_inputs(real, fake, slots):
    mask = slots[:, None, None, None]
    return torch.cat((torch.where(mask, real, fake), torch.where(mask, fake, real)), dim=1)


def loss_d(d, real, fake, method, slots):
    fake = fake.detach()
    if method == "vanilla":
        logits = d(torch.cat((real, fake)))
        target = torch.cat((torch.ones_like(slots), torch.zeros_like(slots))).float()
    elif method == "rsgan":
        logits = d(real) - d(fake)
        target = torch.ones_like(logits)
    elif method == "paired":
        logits, target = d(pair_inputs(real, fake, slots)), slots.float()
    else:
        raise ValueError(method)
    return F.binary_cross_entropy_with_logits(logits, target)


def loss_g(d, real, fake, method, slots):
    if method == "vanilla":
        logits = d(fake)
        target = torch.ones_like(logits)
    elif method == "rsgan":
        logits = d(real) - d(fake)
        target = torch.zeros_like(logits)
    elif method == "paired":
        logits, target = d(pair_inputs(real, fake, slots)), 1 - slots.float()
    else:
        raise ValueError(method)
    return F.binary_cross_entropy_with_logits(logits, target)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prepare_data(root):
    from torchvision.datasets import MNIST
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, training in (("train", True), ("test", False)):
        images_path, labels_path = root / f"{split}_uint8.npy", root / f"{split}_labels.npy"
        if not images_path.exists() or not labels_path.exists():
            dataset = MNIST(str(root), train=training, download=True)
            np.save(images_path, dataset.data[:, None].numpy())
            np.save(labels_path, dataset.targets.numpy())
        images, labels = np.load(images_path), np.load(labels_path)
        assert images.shape == (60000 if training else 10000, 1, 28, 28)
        assert images.dtype == np.uint8 and labels.shape == (len(images),)
        result[split] = {"images_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
                         "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                         "counts": np.bincount(labels, minlength=10).tolist(), "size": len(images)}
    atomic_json(root / "dataset.json", result)
    return result


def rngs_for(seed):
    return {key: torch.Generator().manual_seed(seed + offset) for key, offset in
            [("real_d", 20000), ("real_g", 30000), ("noise_d", 40000),
             ("noise_g", 50000), ("slots", 60000)]}


def train(config, method, seed, steps, output, data, device="cuda", commit=None, evaluator=None, evaluation_signature=None):
    """data is NCHW uint8; all sampling uses explicit checkpointed CPU generators."""
    configure(device)
    if steps < 1 or config["batch_size"] % 2:
        raise ValueError("Positive steps and even batch size required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "run.json"
    signature = {"config": config, "method": method, "seed": seed, "source_sha256": source_hash(),
                 "dataset_sha256": hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest(),
                 "torch": torch.__version__, "device": device, "evaluation_signature": evaluation_signature}
    if provenance_path.exists():
        stored = json.loads(provenance_path.read_text())
        if any(stored[k] != v for k, v in signature.items()):
            raise ValueError("Resume configuration, dataset, source, or runtime changed; use a new run ID")
    g, d = models(config, method, seed)
    g.to(device)
    d.to(device)
    opt_g = torch.optim.Adam(g.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]))
    opt_d = torch.optim.Adam(d.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]))
    rngs = rngs_for(seed)
    # uint8 training data fits comfortably in GPU memory and eliminates worker state.
    data = data.to(device)
    fixed_z = torch.randn(64, config["latent_dim"], generator=torch.Generator().manual_seed(config["eval_seed"])).to(device)
    start_step, training_seconds, rows = 0, 0., []
    checkpoint = output / "latest.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Checkpoint signature mismatch")
        g.load_state_dict(state["generator"])
        d.load_state_dict(state["discriminator"])
        opt_g.load_state_dict(state["optimizer_g"])
        opt_d.load_state_dict(state["optimizer_d"])
        for key, rng in rngs.items():
            rng.set_state(state["rngs"][key].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        start_step, training_seconds, rows = state["step"], state["training_seconds"], state["rows"]
    else:
        metadata = {**signature, "python": platform.python_version(), "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                    "parameters_g": sum(p.numel() for p in g.parameters()),
                    "parameters_d": sum(p.numel() for p in d.parameters())}
        atomic_json(provenance_path, metadata)
        (output / "training_source.py").write_bytes(Path(__file__).read_bytes())
        save_grid(g, fixed_z, output / "samples_000000.png")
    if evaluator is not None and start_step and start_step % config["eval_every"] == 0:
        evaluator(g, start_step, output)
    if start_step > steps:
        raise ValueError("Requested step predates saved progress")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batch = config["batch_size"]

    def draw(kind):
        indices = torch.randint(len(data), (batch,), generator=rngs[f"real_{kind}"]).to(device)
        real = data[indices].float().div(127.5).sub(1)
        noise = torch.randn(batch, config["latent_dim"], generator=rngs[f"noise_{kind}"]).to(device)
        slots = (torch.randperm(batch, generator=rngs["slots"]) < batch // 2).to(device)
        return real, noise, slots

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    segment_start = time.perf_counter()
    segment_step = start_step
    for step in range(start_step + 1, steps + 1):
        real, noise, slots = draw("d")
        d.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = g(noise)
        dl = loss_d(d, real, fake, method, slots)
        dl.backward()
        opt_d.step()
        real, noise, slots = draw("g")
        d.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        gl = loss_g(d, real, g(noise), method, slots)
        gl.backward()
        opt_g.step()
        if step % config["log_every"] == 0 or step == steps:
            sync()
            elapsed = time.perf_counter() - segment_start
            training_seconds += elapsed
            row = {"step": step, "d_loss": dl.item(), "g_loss": gl.item(),
                   "training_seconds": training_seconds, "seconds_per_step": elapsed / (step - segment_step)}
            if not np.isfinite([row["d_loss"], row["g_loss"]]).all():
                raise RuntimeError("Nonfinite training loss")
            rows.append(row)
            print(json.dumps({"method": method, "seed": seed, **row}), flush=True)
            segment_start, segment_step = time.perf_counter(), step
        if step % config["checkpoint_every"] == 0 or step == steps:
            sync()
            # Flush partial timing segments if checkpoint/log periods differ.
            if segment_step != step:
                training_seconds += time.perf_counter() - segment_start
            state = {"signature": signature, "step": step, "generator": g.state_dict(),
                     "discriminator": d.state_dict(), "optimizer_g": opt_g.state_dict(),
                     "optimizer_d": opt_d.state_dict(), "rngs": {k: v.get_state() for k, v in rngs.items()},
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                     "training_seconds": training_seconds, "rows": rows}
            torch.save(state, output / "latest.pt.tmp")
            (output / "latest.pt.tmp").replace(checkpoint)
            if step % 10000 == 0:
                torch.save({"generator": g.state_dict(), "discriminator": d.state_dict(), "signature": signature, "config": config, "step": step,
                            "method": method, "seed": seed}, output / f"generator_{step:06d}.pt")
            save_grid(g, fixed_z, output / f"samples_{step:06d}.png")
            if evaluator is not None and step % config["eval_every"] == 0:
                evaluator(g, step, output)
            with (output / "training.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            status = {"step": step, "target_steps": steps, "training_seconds": training_seconds,
                      "last_log": rows[-1], "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}
            atomic_json(output / "status.json", status)
            if commit:
                commit()
            segment_start, segment_step = time.perf_counter(), step
    return json.loads((output / "status.json").read_text())


def save_grid(g, z, path):
    from torchvision.utils import save_image
    was_training = g.training
    g.eval()
    with torch.no_grad():
        images = (g(z).cpu() + 1) / 2
    save_image(images, path, nrow=8, padding=2)
    g.train(was_training)
