"""Integrity study: unchanged CBN/projection training with G+D snapshots every 10k. Shared loop, isolated RNG streams, exact resume."""
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

METHODS = ("vanilla", "rsgan", "paired")


def configure(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4 if device == "cuda" else 1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class ConditionalBatchNorm(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=False)
        self.affine = nn.Embedding(classes, 2 * channels)
        self.channels = channels

    def forward(self, x, labels):
        gamma, beta = self.affine(labels).chunk(2, dim=1)
        return self.bn(x) * gamma[:, :, None, None] + beta[:, :, None, None]


class Generator(nn.Module):
    def __init__(self, latent, width, classes=10):
        super().__init__()
        self.classes = classes
        self.convs = nn.ModuleList([
            nn.ConvTranspose2d(latent, width * 4, 4, 1, 0, bias=False),
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1, bias=False)])
        self.norms = nn.ModuleList([ConditionalBatchNorm(c, classes) for c in (width * 4, width * 2, width)])
        self.output = nn.ConvTranspose2d(width, 3, 4, 2, 1)

    def forward(self, z, labels):
        x = z[:, :, None, None]
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x), labels))
        return self.output(x).tanh()


class Discriminator(nn.Module):
    def __init__(self, channels, width, classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(width, width * 2, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1), nn.LeakyReLU(.2))
        self.unconditional = nn.Linear(width * 4, 1)
        self.embedding = nn.Embedding(classes, width * 4)

    def representation(self, x):
        # Global sum pooling: one 256-dimensional vector at width 64.
        return self.features(x).sum(dim=(2, 3))

    def forward(self, x, labels):
        h = self.representation(x)
        return self.unconditional(h).flatten() + (self.embedding(labels) * h).sum(dim=1)


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0, .02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, 0, .02)
    elif isinstance(module, ConditionalBatchNorm):
        # apply() visits children first, so override the generic embedding init.
        nn.init.normal_(module.affine.weight[:, :module.channels], 1, .02)
        nn.init.zeros_(module.affine.weight[:, module.channels:])


def models(config, method, seed):
    if method not in METHODS:
        raise ValueError(method)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        g = Generator(config["latent_dim"], config["width"], config["classes"])
        g.apply(initialize)
        torch.manual_seed(seed + 10000)
        d = Discriminator(6 if method == "paired" else 3, config.get("discriminator_width", config["width"]), config["classes"])
        d.apply(initialize)
    return g, d


def pair_inputs(real, fake, slots):
    mask = slots[:, None, None, None]
    return torch.cat((torch.where(mask, real, fake), torch.where(mask, fake, real)), dim=1)


def loss_d(d, real, fake, method, slots, labels):
    fake = fake.detach()
    if method == "vanilla":
        logits = d(torch.cat((real, fake)), torch.cat((labels, labels)))
        target = torch.cat((torch.ones_like(slots), torch.zeros_like(slots))).float()
    elif method == "rsgan":
        logits = d(real, labels) - d(fake, labels)
        target = torch.ones_like(logits)
    elif method == "paired":
        logits, target = d(pair_inputs(real, fake, slots), labels), slots.float()
    else:
        raise ValueError(method)
    return F.binary_cross_entropy_with_logits(logits, target)


def loss_g(d, real, fake, method, slots, labels):
    if method == "vanilla":
        logits = d(fake, labels)
        target = torch.ones_like(logits)
    elif method == "rsgan":
        logits = d(real, labels) - d(fake, labels)
        target = torch.zeros_like(logits)
    elif method == "paired":
        logits, target = d(pair_inputs(real, fake, slots), labels), 1 - slots.float()
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
    from torchvision.datasets import CIFAR10
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    original = CIFAR10(str(root), train=True, download=True)
    array = original.data.transpose(0, 3, 1, 2)
    labels = np.asarray(original.targets, dtype=np.int64)
    array_path = root / "train_uint8.npy"
    if array_path.exists():
        assert np.array_equal(np.load(array_path), array), "Cached image order changed"
    else:
        np.save(array_path, array)
    assert np.array_equal(np.bincount(labels), np.full(10, 5000))
    np.save(root / "train_labels.npy", labels)
    result = {"dataset_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
              "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest()}
    atomic_json(root / "classmatched_dataset.json", result)
    return result


def class_indices(labels, classes):
    groups = [(labels == i).nonzero().flatten() for i in range(classes)]
    if not groups or min(map(len, groups)) == 0 or len({len(x) for x in groups}) != 1:
        raise ValueError("Expected equal nonempty class counts")
    return torch.stack(groups)


def draw_matched_indices(groups, count, label_rng, image_rng):
    # Each class appears floor(B/C) or ceil(B/C) times; the extra classes rotate.
    offset = torch.randint(len(groups), (1,), generator=label_rng)
    labels = (torch.arange(count) + offset) % len(groups)
    labels = labels[torch.randperm(count, generator=label_rng)]
    within_class = torch.randint(groups.shape[1], (count,), generator=image_rng)
    return groups[labels, within_class], labels


def rngs_for(seed):
    return {key: torch.Generator().manual_seed(seed + offset) for key, offset in
            [("real_d", 20000), ("real_g", 30000), ("noise_d", 40000),
             ("noise_g", 50000), ("slots", 60000), ("labels_d", 80000), ("labels_g", 90000)]}


def train(config, method, seed, steps, output, data, labels, device="cuda", commit=None):
    """NCHW uint8 data plus aligned labels; all sampling streams are checkpointed."""
    configure(device)
    labels = labels.cpu().long()
    assert len(labels) == len(data)
    groups = class_indices(labels, config["classes"])
    if steps < 1 or any(b < 2 or b % 2 for b in (config["batch_size"], config.get("discriminator_batch_size", config["batch_size"]))):
        raise ValueError("Positive steps and even batch size required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "run.json"
    signature = {"config": config, "method": method, "seed": seed, "source_sha256": source_hash(),
                 "dataset_sha256": hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest(),
                 "labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
                 "torch": torch.__version__, "device": device}
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
    fixed_y = (torch.arange(64) % config["classes"]).to(device)
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
        save_grid(g, fixed_z, fixed_y, output / "samples_000000.png")
    if start_step > steps:
        raise ValueError("Requested step predates saved progress")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    def draw(kind):
        batch = config.get("discriminator_batch_size", config["batch_size"]) if kind == "d" else config["batch_size"]
        indices, y = draw_matched_indices(groups, batch, rngs[f"labels_{kind}"], rngs[f"real_{kind}"])
        indices, y = indices.to(device), y.to(device)
        real = data[indices].float().div(127.5).sub(1)
        noise = torch.randn(batch, config["latent_dim"], generator=rngs[f"noise_{kind}"]).to(device)
        slots = (torch.randperm(batch, generator=rngs["slots"]) < batch // 2).to(device)
        return real, noise, slots, y

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    segment_start = time.perf_counter()
    segment_step = start_step
    for step in range(start_step + 1, steps + 1):
        real, noise, slots, y = draw("d")
        d.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            # Preserve G forward batch size, including conditional BatchNorm.
            # Extra D examples require extra no-grad G forwards (and BN updates).
            fake = torch.cat([g(z, labels_chunk) for z, labels_chunk in
                              zip(noise.split(config["batch_size"]), y.split(config["batch_size"]))])
        dl = loss_d(d, real, fake, method, slots, y)
        dl.backward()
        opt_d.step()
        real, noise, slots, y = draw("g")
        d.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        gl = loss_g(d, real, g(noise, y), method, slots, y)
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
            save_grid(g, fixed_z, fixed_y, output / f"samples_{step:06d}.png")
            if step % 10000 == 0:
                save_class_grid(g, fixed_z[:8], output / f"class_grid_{step:06d}.png")
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


def save_grid(g, z, labels, path):
    from torchvision.utils import save_image
    was_training = g.training
    g.eval()
    with torch.no_grad():
        images = (g(z, labels).cpu() + 1) / 2
    save_image(images, path, nrow=8, padding=2)
    g.train(was_training)


def save_class_grid(g, z, path):
    # Ten rows in CIFAR label order, same eight latent vectors in every row.
    labels = torch.arange(g.classes, device=z.device).repeat_interleave(len(z))
    save_grid(g, z.repeat(g.classes, 1), labels, path)


class Uint8Dataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]

