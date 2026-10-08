"""PacGAN2 on fixed-spacing grids; frozen baseline loop with packed BCE losses."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]


def centers(config):
    side = config["grid_side"]
    if side < 1 or side * side != config["modes"] or config["spacing"] <= 0:
        raise ValueError("A square grid with positive spacing is required")
    axis = (torch.arange(side) - (side - 1) / 2) * config["spacing"]
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    return torch.stack((x.flatten(), y.flatten()), dim=1)


def sample_real(config, count, rng):
    indices = torch.randint(config["modes"], (count,), generator=rng)
    noise = torch.randn(count, 2, generator=rng) * config["sigma"]
    return centers(config)[indices] + noise


def mlp(input_dim, width, output_dim):
    return nn.Sequential(nn.Linear(input_dim, width), nn.LeakyReLU(0.2),
                         nn.Linear(width, width), nn.LeakyReLU(0.2),
                         nn.Linear(width, output_dim))


def models(config, method, seed):
    # Initialization is isolated from every sampling stream. Both methods receive
    # exactly the same G parameters for each seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = mlp(config["latent_dim"], config["generator_width"], 2)
        torch.manual_seed(seed + 10000)
        discriminator = mlp(4 if method in ("paired", "pacgan2") else 2,
                            config["discriminator_width"], 1)
    return generator, discriminator


def pair_inputs(real, fake, real_in_a):
    """Inputs A,B and ground-truth probability target 'A is real'."""
    mask = real_in_a[:, None].bool()
    return torch.cat((torch.where(mask, real, fake),
                      torch.where(mask, fake, real)), dim=1)


def pack(points):
    """Pair consecutive independent draws; every point appears exactly once."""
    if points.ndim != 2 or points.shape[1] != 2 or len(points) % 2:
        raise ValueError("PacGAN2 requires an even number of 2D points")
    return points.reshape(-1, 4)


def discriminator_loss(discriminator, real, fake, method, real_in_a):
    fake = fake.detach()
    if method == "pacgan2":
        logits = discriminator(torch.cat((pack(real), pack(fake)))).flatten()
        targets = torch.cat((torch.ones(len(real) // 2, device=real.device),
                             torch.zeros(len(fake) // 2, device=fake.device)))
    elif method == "vanilla":
        logits = discriminator(torch.cat((real, fake))).flatten()
        targets = torch.cat((torch.ones(len(real), device=real.device),
                             torch.zeros(len(fake), device=fake.device)))
    elif method == "rsgan":
        logits = (discriminator(real) - discriminator(fake)).flatten()
        targets = torch.ones_like(logits)
    else:
        logits = discriminator(pair_inputs(real, fake, real_in_a)).flatten()
        targets = real_in_a.float()
    # Mean BCE over individual classification decisions in both methods.
    return F.binary_cross_entropy_with_logits(logits, targets)


def generator_loss(discriminator, real, fake, method, real_in_a):
    if method == "pacgan2":
        logits = discriminator(pack(fake)).flatten()
        targets = torch.ones_like(logits)
    elif method == "vanilla":
        logits = discriminator(fake).flatten()
        targets = torch.ones_like(logits)
    elif method == "rsgan":
        logits = (discriminator(real) - discriminator(fake)).flatten()
        targets = torch.zeros_like(logits)
    else:
        logits = discriminator(pair_inputs(real, fake, real_in_a)).flatten()
        targets = 1 - real_in_a.float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def coverage_metrics(points, config):
    distance = torch.cdist(points.cpu(), centers(config))
    nearest_distance, nearest = distance.min(dim=1)
    valid = nearest_distance <= config["valid_radius_sigma"] * config["sigma"]
    counts = torch.bincount(nearest[valid], minlength=config["modes"])
    mass = counts.double() / len(points)
    invalid = 1 - mass.sum().item()
    # Treat invalid samples as an extra category whose target mass is zero.
    tv = 0.5 * ((mass - 1 / config["modes"]).abs().sum().item() + invalid)
    result = {
        "coverage": int((mass >= config["coverage_min_mass"]).sum()),
        "valid_fraction": 1 - invalid,
        "mode_tv": tv,
    }
    result.update({f"mode_{i}_mass": value for i, value in enumerate(mass.tolist())})
    return result


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run(config, method, seed, directory):
    directory.mkdir(parents=True, exist_ok=False)
    device = torch.device(config["device"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    generator, discriminator = models(config, method, seed)
    generator.to(device)
    discriminator.to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=config["learning_rate"],
                             betas=tuple(config["betas"]))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate"],
                             betas=tuple(config["betas"]))
    rngs = {key: torch.Generator().manual_seed(seed + offset) for key, offset in
            [("real_d", 20000), ("real_g", 30000), ("noise_d", 40000),
             ("noise_g", 50000), ("slots", 60000), ("eval", 70000)]}
    eval_z = torch.randn(config["eval_samples"], config["latent_dim"],
                         generator=rngs["eval"]).to(device)
    batch = config["batch_size"]
    rows = []
    start = time.perf_counter()
    training_seconds = 0.0
    d_value = g_value = float("nan")

    def evaluate(step):
        generator.eval()
        with torch.no_grad():
            points = generator(eval_z).cpu()
        row = {"method": method, "seed": seed, "step": step,
               "training_seconds": training_seconds,
               "wall_seconds": time.perf_counter() - start,
               "d_loss": d_value, "g_loss": g_value,
               **coverage_metrics(points, config)}
        rows.append(row)
        with (directory / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{method:7s} seed={seed} step={step:5d} "
              f"coverage={row['coverage']}/{config['modes']} "
              f"valid={row['valid_fraction']:.3f} TV={row['mode_tv']:.3f} "
              f"train_s={training_seconds:.1f}", flush=True)
        generator.train()
        return points

    def save_checkpoint(path, step):
        torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                    "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict(),
                    "rng_states": {key: rng.get_state() for key, rng in rngs.items()},
                    "step": step, "config": config, "seed": seed, "method": method}, path)

    evaluate(0)
    for step in range(1, config["steps"] + 1):
        synchronize(device)
        step_start = time.perf_counter()
        real = sample_real(config, batch, rngs["real_d"]).to(device)
        noise = torch.randn(batch, config["latent_dim"], generator=rngs["noise_d"]).to(device)
        # Balanced, randomly assigned slots. Draw for both methods to preserve streams.
        slots = (torch.randperm(batch, generator=rngs["slots"]) < batch // 2).to(device)
        discriminator.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = generator(noise)
        loss_d = discriminator_loss(discriminator, real, fake, method, slots)
        loss_d.backward()
        opt_d.step()

        real = sample_real(config, batch, rngs["real_g"]).to(device)
        noise = torch.randn(batch, config["latent_dim"], generator=rngs["noise_g"]).to(device)
        slots = (torch.randperm(batch, generator=rngs["slots"]) < batch // 2).to(device)
        discriminator.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        loss_g = generator_loss(discriminator, real, generator(noise), method, slots)
        loss_g.backward()
        opt_g.step()
        synchronize(device)
        training_seconds += time.perf_counter() - step_start
        if step % config["eval_every"] == 0 or step == config["steps"]:
            d_value, g_value = loss_d.item(), loss_g.item()
            if not math.isfinite(d_value + g_value):
                raise RuntimeError(f"Nonfinite loss: {method}, seed {seed}, step {step}")
            points = evaluate(step)
            if step in (10000, 50000, 100000, 150000):
                np.save(directory / f"samples_step{step}.npy", points.numpy())
                save_checkpoint(directory / f"checkpoint_{step}.pt", step)

    np.save(directory / "final_samples.npy", points.numpy())
    torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict(),
                "rng_states": {key: rng.get_state() for key, rng in rngs.items()},
                "step": config["steps"], "config": config, "seed": seed, "method": method},
               directory / "checkpoint.pt")
    metadata = {"config": config, "method": method, "seed": seed,
                "parameters_g": sum(p.numel() for p in generator.parameters()),
                "parameters_d": sum(p.numel() for p in discriminator.parameters()),
                "torch": torch.__version__, "python": platform.python_version(),
                "platform": platform.platform(), "training_seconds": training_seconds,
                "wall_seconds": time.perf_counter() - start}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def snapshot(directory, config):
    source = directory / "source"
    source.mkdir(parents=True, exist_ok=False)
    files = [ROOT / name for name in ["pyproject.toml", "uv.lock", ".python-version"]]
    files += list((ROOT / "src").rglob("*.py"))
    files += list((ROOT / "tests").rglob("*.py"))
    hashes = {}
    for path in files:
        if path.exists():
            relative = path.relative_to(ROOT)
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    (directory / "provenance.json").write_text(json.dumps({
        "git_commit": head.stdout.strip() if head.returncode == 0 else None,
        "source_sha256": hashes,
    }, indent=2) + "\n")

