"""D-only batch intervention; original losses and generator workload unchanged."""
import csv
import json
import math
import platform
import time
import numpy as np
import torch
from .grid_dual_slot import ROOT, models, sample_real, coverage_metrics, synchronize, snapshot
from . import grid_pacgan, grid_dual_slot


def discriminator_loss(discriminator, real, fake, method, slots):
    trainer = grid_dual_slot if method == "dual_slot" else grid_pacgan
    return trainer.discriminator_loss(discriminator, real, fake, method, slots)


def generator_loss(discriminator, real, fake, method, slots):
    trainer = grid_dual_slot if method == "dual_slot" else grid_pacgan
    return trainer.generator_loss(discriminator, real, fake, method, slots)


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
    d_batch = config.get("discriminator_batch_size", batch)
    if d_batch != batch:
        # Keep the original shared slot stream's consumption unchanged for G.
        rngs["slots_d_large"] = torch.Generator().manual_seed(seed + 80000)
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
        real = sample_real(config, d_batch, rngs["real_d"]).to(device)
        noise = torch.randn(d_batch, config["latent_dim"], generator=rngs["noise_d"]).to(device)
        # Balanced, randomly assigned slots. Draw for both methods to preserve streams.
        slots = (torch.randperm(batch, generator=rngs["slots"]) < batch // 2).to(device)
        if d_batch != batch:
            slots = (torch.randperm(d_batch, generator=rngs["slots_d_large"]) < d_batch // 2).to(device)
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
            if step in (5000, 10000, 25000, 50000, 75000, 100000, 150000):
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


