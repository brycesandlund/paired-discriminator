"""Class-matched CIFAR-10 pilot. Shared loop, isolated RNG streams, exact resume."""
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


class Generator(nn.Module):
    def __init__(self, latent, width, classes=10):
        super().__init__()
        self.classes = classes
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent + classes, width * 4, 4, 1, 0, bias=False), nn.BatchNorm2d(width * 4), nn.ReLU(),
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
            nn.ConvTranspose2d(width, 3, 4, 2, 1), nn.Tanh())

    def forward(self, z, labels):
        condition = F.one_hot(labels, self.classes).to(z.dtype)
        inputs = torch.cat((z, condition), dim=1)
        return self.net(inputs[:, :, None, None])


class Discriminator(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        # No batch normalization: unary critics must not compare via batch statistics.
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(width, width * 2, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(width * 4, 1, 4, 1, 0))

    def forward(self, x):
        return self.net(x).flatten()


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
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
        g = Generator(config["latent_dim"], config["width"], config["classes"])
        g.apply(initialize)
        torch.manual_seed(seed + 10000)
        d = Discriminator(6 if method == "paired" else 3, config["width"])
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
    if steps < 1 or config["batch_size"] % 2:
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
    batch = config["batch_size"]

    def draw(kind):
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
            fake = g(noise, y)
        dl = loss_d(d, real, fake, method, slots)
        dl.backward()
        opt_d.step()
        real, noise, slots, y = draw("g")
        d.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        gl = loss_g(d, real, g(noise, y), method, slots)
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
            if step in (10000, 50000):
                torch.save({"generator": g.state_dict(), "config": config, "step": step,
                            "method": method, "seed": seed}, output / f"generator_{step:06d}.pt")
            save_grid(g, fixed_z, fixed_y, output / f"samples_{step:06d}.png")
            if step in (10000, 50000):
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


def evaluate(run_dir, data_root, step, device="cuda", commit=None):
    import torch_fidelity
    configure(device)
    run_dir, data_root = Path(run_dir), Path(data_root)
    dest = run_dir / f"metrics_{step:06d}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    ckpt = torch.load(run_dir / f"generator_{step:06d}.pt", map_location="cpu", weights_only=False)
    config = ckpt["config"]
    g, _ = models(config, ckpt["method"], ckpt["seed"])
    g.load_state_dict(ckpt["generator"])
    g.to(device).eval()
    n = config["eval_samples"]
    rng = torch.Generator().manual_seed(config["eval_seed"] + 1)
    generated = []
    with torch.no_grad():
        for start in range(0, n, 128):
            z = torch.randn(min(128, n - start), config["latent_dim"], generator=rng).to(device)
            y = (torch.arange(start, start + len(z)) % config["classes"]).to(device)
            generated.append(((g(z, y).cpu() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8))
    fake = torch.cat(generated)
    real = torch.from_numpy(np.load(data_root / "train_uint8.npy"))
    ids = torch.randperm(len(real), generator=torch.Generator().manual_seed(config["eval_seed"] + 2))[:n]
    real = real[ids]
    began = time.perf_counter()
    metrics = torch_fidelity.calculate_metrics(
        input1=Uint8Dataset(fake), input2=Uint8Dataset(real), cuda=device == "cuda",
        fid=True, kid=True, prc=True, isc=False, feature_extractor="inception-v3-compat",
        feature_layer_prc="2048", batch_size=128, kid_subsets=100, kid_subset_size=1000,
        prc_neighborhood=3, prc_batch_size=512, save_cpu_ram=True,
        rng_seed=config["eval_seed"], cache_root=str(data_root / "fidelity-cache"),
        input2_cache_name=f"cifar10-train-fixed-{n}-seed{config['eval_seed']+2}", verbose=True)
    result = {"method": ckpt["method"], "seed": ckpt["seed"], "step": step,
              "samples_generated": n, "samples_real": n,
              "conditioning": "one-hot G only; exactly balanced requested classes at evaluation", "reference": "fixed subset of CIFAR10 training split",
              "features": "torch-fidelity inception-v3-compat 2048; also used for precision/recall",
              "torch_fidelity": torch_fidelity.__version__, "evaluation_seconds": time.perf_counter() - began,
              **{k: float(v) for k, v in metrics.items()}}
    atomic_json(dest, result)
    if commit:
        commit()
    return result
