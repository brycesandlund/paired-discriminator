"""Grid reference interventions, preserving original batches and GAN losses."""
import csv
import json
import math
import platform
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from . import grid_pacgan as baseline
from .grid_experiment import ROOT, centers, sample_real, coverage_metrics, synchronize, snapshot


def base_method(method):
    return method.split("_", 1)[0]


def models(config, method, seed):
    return baseline.models(config, base_method(method), seed)


def discriminator_loss(discriminator, real, fake, method, slots):
    return baseline.discriminator_loss(discriminator, real, fake, base_method(method), slots)


def generator_loss(discriminator, real, fake, method, slots):
    return baseline.generator_loss(discriminator, real, fake, base_method(method), slots)


def near_match(real, fake):
    """Minimum total squared-distance bijection; leave fake order untouched.

    Symmetric cost and one-to-one assignment preserve both sample marginals.
    Assignment is a discrete, stop-gradient operation if used during G training.
    """
    if real.shape != fake.shape or real.ndim != 2 or real.shape[1] != 2:
        raise ValueError("Equal-sized 2D sample batches required")
    cost = cdist(real.detach().cpu().numpy(), fake.detach().cpu().numpy(), 'sqeuclidean')
    row, col = linear_sum_assignment(cost)
    permutation = np.empty(len(real), dtype=np.int64)
    permutation[col] = row
    return real[torch.from_numpy(permutation).to(real.device)]


def accepted_mass(fake, config):
    distances = torch.cdist(fake.detach().cpu(), centers(config))
    nearest_distance, nearest = distances.min(1)
    valid = nearest_distance <= config['valid_radius_sigma'] * config['sigma']
    return torch.bincount(nearest[valid], minlength=config['modes']).double() / len(fake)


def deficit_weights(ema_mass, uniform_mix=.5):
    """50% uniform + 50% normalized positive deficit from ideal accepted mass."""
    if not 0 <= uniform_mix <= 1:
        raise ValueError("uniform_mix must be in [0,1]")
    uniform = torch.full_like(ema_mass, 1 / len(ema_mass))
    deficit = (uniform - ema_mass).clamp_min(0)
    adaptive = deficit / deficit.sum() if deficit.sum() > 1e-15 else uniform
    return uniform_mix * uniform + (1 - uniform_mix) * adaptive


def sample_weighted(config, count, rng, weights):
    indices = torch.multinomial(weights, count, replacement=True, generator=rng)
    noise = torch.randn(count, 2, generator=rng) * config['sigma']
    return centers(config)[indices] + noise


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
    policy = method.split("_", 1)[1] if "_" in method else "random"
    both_phases = config.get("reference_phase", "d_only") == "both"
    ema_mass = torch.full((config["modes"],), 1 / config["modes"], dtype=torch.float64)
    weights = torch.full_like(ema_mass, 1 / config["modes"])
    distance_before = distance_after = 0.0
    distance_steps = 0
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
               **coverage_metrics(points, config),
               "d_pair_rms_before_mean": distance_before / max(1, distance_steps),
               "d_pair_rms_after_mean": distance_after / max(1, distance_steps),
               **{f"reference_weight_{i}": w for i,w in enumerate(weights.tolist())},
               **{f"ema_mass_{i}": w for i,w in enumerate(ema_mass.tolist())}}
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
                    "step": step, "config": config, "seed": seed, "method": method, "ema_mass": ema_mass}, path)

    evaluate(0)
    for step in range(1, config["steps"] + 1):
        synchronize(device)
        step_start = time.perf_counter()
        weights = deficit_weights(ema_mass, config.get("uniform_mix", .5)) if policy == "deficit" else torch.full_like(ema_mass, 1 / config["modes"])
        real = (sample_weighted(config, batch, rngs["real_d"], weights) if policy == "deficit"
                else sample_real(config, batch, rngs["real_d"])).to(device)
        noise = torch.randn(batch, config["latent_dim"], generator=rngs["noise_d"]).to(device)
        # Balanced, randomly assigned slots. Draw for both methods to preserve streams.
        slots = (torch.randperm(batch, generator=rngs["slots"]) < batch // 2).to(device)
        discriminator.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = generator(noise)
        if policy == "near":
            distance_before += (real - fake).square().sum(1).mean().sqrt().item()
            real = near_match(real, fake)
            distance_after += (real - fake).square().sum(1).mean().sqrt().item()
            distance_steps += 1
        observed_mass = accepted_mass(fake, config) if policy == "deficit" else None
        loss_d = discriminator_loss(discriminator, real, fake, method, slots)
        loss_d.backward()
        opt_d.step()

        real = (sample_weighted(config, batch, rngs["real_g"], weights) if policy == "deficit" and both_phases
                else sample_real(config, batch, rngs["real_g"])).to(device)
        noise = torch.randn(batch, config["latent_dim"], generator=rngs["noise_g"]).to(device)
        slots = (torch.randperm(batch, generator=rngs["slots"]) < batch // 2).to(device)
        discriminator.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        fake_g = generator(noise)
        if policy == "near" and both_phases:
            real = near_match(real, fake_g)
        loss_g = generator_loss(discriminator, real, fake_g, method, slots)
        loss_g.backward()
        opt_g.step()
        if observed_mass is not None:
            decay = config.get("coverage_ema_decay", .99)
            ema_mass = decay * ema_mass + (1 - decay) * observed_mass
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
                "step": config["steps"], "config": config, "seed": seed, "method": method, "ema_mass": ema_mass},
               directory / "checkpoint.pt")
    metadata = {"config": config, "method": method, "seed": seed,
                "parameters_g": sum(p.numel() for p in generator.parameters()),
                "parameters_d": sum(p.numel() for p in discriminator.parameters()),
                "torch": torch.__version__, "python": platform.python_version(),
                "platform": platform.platform(), "training_seconds": training_seconds,
                "wall_seconds": time.perf_counter() - start}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


