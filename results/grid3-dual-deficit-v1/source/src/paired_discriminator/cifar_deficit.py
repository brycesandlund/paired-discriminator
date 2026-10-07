"""Unconditional CIFAR D-only class deficit sampling; frozen original GAN math."""
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from . import cifar
from .cifar import configure, models, rngs_for, loss_d, loss_g, atomic_json, save_grid
from .cifar_class_representation import REPO, digest

METHODS = ("vanilla", "paired", "vanilla_deficit", "paired_deficit")
WEIGHTS = {"cifar10_resnet56": "187c023aee0c9cf3093a682d9447a538cbaf5489f7ae78a7df4edf2246ce380b",
           "cifar10_vgg16_bn": "6ee7ea24b52cfbbe9751608a81d5a7b2f5bac4e8f7d19e420030072ca257ed97"}


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes() + Path(cifar.__file__).read_bytes()).hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def load_classifier(name, weight_root, device="cuda"):
    path = Path(weight_root) / f"{name}.pt"
    if digest(path) != WEIGHTS[name]:
        raise ValueError("Classifier weights changed")
    model = torch.hub.load(REPO, name, pretrained=False, trust_repo=True)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval().requires_grad_(False)


def class_groups(labels):
    groups = [(labels == k).nonzero().flatten() for k in range(10)]
    if any(len(g) == 0 for g in groups):
        raise ValueError("All ten real classes required")
    return groups


def deficit_weights(ema):
    deficit = (.1 - ema).clamp_min(0)
    if float(deficit.sum()) == 0:
        return torch.full_like(ema, .1)
    return .05 + .5 * deficit / deficit.sum()


def sample_indices(groups, weights, count, rng):
    classes = torch.multinomial(weights, count, replacement=True, generator=rng)
    indices = torch.empty(count, dtype=torch.long)
    for k, group in enumerate(groups):
        mask = classes == k
        indices[mask] = group[torch.randint(len(group), (int(mask.sum()),), generator=rng)]
    return indices


@torch.inference_mode()
def classifier_probabilities(model, fake):
    # Quantize identically to retrospective evaluation, then normalize for the frozen classifier.
    image = ((fake.detach() + 1) * 127.5).round().clamp(0, 255) / 255
    mean = fake.new_tensor([.4914, .4822, .4465])[None, :, None, None]
    std = fake.new_tensor([.2023, .1994, .2010])[None, :, None, None]
    return model((image - mean) / std).softmax(1)


def accepted_mass(probabilities, threshold=.9):
    confidence, labels = probabilities.detach().max(1)
    # Rejected images remain in the denominator, as in the grid experiment.
    return torch.bincount(labels[confidence >= threshold], minlength=10).cpu().double() / len(labels)


def train(config, method, seed, steps, output, data, labels, classifier, device="cuda", commit=None):
    """data is NCHW uint8; all sampling uses explicit checkpointed CPU generators."""
    configure(device)
    base_method = method.removesuffix("_deficit")
    if method not in METHODS:
        raise ValueError(method)
    adaptive = method.endswith("_deficit")
    if config["uniform_floor"] != .5 or not 0 <= config["ema_decay"] < 1:
        raise ValueError("This protocol requires a 50% uniform floor and valid EMA decay")
    labels = labels.cpu().long()
    if len(labels) != len(data):
        raise ValueError("Image/label length mismatch")
    groups = class_groups(labels)
    classifier = classifier.to(device).eval().requires_grad_(False)
    ema = torch.full((10,), .1, dtype=torch.float64)
    if steps < 1 or config["batch_size"] % 2:
        raise ValueError("Positive steps and even batch size required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "run.json"
    signature = {"config": config, "method": method, "seed": seed, "source_sha256": source_hash(), "classifier_sha256": model_digest(classifier),
                 "labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
                 "dataset_sha256": hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest(),
                 "torch": torch.__version__, "device": device}
    if provenance_path.exists():
        stored = json.loads(provenance_path.read_text())
        if any(stored[k] != v for k, v in signature.items()):
            raise ValueError("Resume configuration, dataset, source, or runtime changed; use a new run ID")
    g, d = models(config, base_method, seed)
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
        ema = state["ema"].cpu()
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
    if start_step > steps:
        raise ValueError("Requested step predates saved progress")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batch = config["batch_size"]

    def draw(kind):
        if kind == "d" and adaptive:
            indices = sample_indices(groups, deficit_weights(ema), batch, rngs["real_d"]).to(device)
        else:
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
        # Reuse existing fake-D images; this classifier never participates in autograd.
        observed = accepted_mass(classifier_probabilities(classifier, fake), config["confidence_threshold"])
        dl = loss_d(d, real, fake, base_method, slots)
        dl.backward()
        opt_d.step()
        real, noise, slots = draw("g")
        d.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        gl = loss_g(d, real, g(noise), base_method, slots)
        gl.backward()
        opt_g.step()
        ema = config["ema_decay"] * ema + (1 - config["ema_decay"]) * observed
        if step % config["log_every"] == 0 or step == steps:
            sync()
            elapsed = time.perf_counter() - segment_start
            training_seconds += elapsed
            row = {"step": step, "d_loss": dl.item(), "g_loss": gl.item(),
                   "training_seconds": training_seconds, "seconds_per_step": elapsed / (step - segment_step)}
            if not np.isfinite([row["d_loss"], row["g_loss"]]).all():
                raise RuntimeError("Nonfinite training loss")
            row.update({f"accepted_ema_{k}": float(ema[k]) for k in range(10)})
            weights = deficit_weights(ema) if adaptive else torch.full((10,), .1)
            row.update({f"next_real_weight_{k}": float(weights[k]) for k in range(10)})
            rows.append(row)
            print(json.dumps({"method": method, "seed": seed, **row}), flush=True)
            segment_start, segment_step = time.perf_counter(), step
        if step % config["checkpoint_every"] == 0 or step == steps:
            sync()
            # Flush partial timing segments if checkpoint/log periods differ.
            if segment_step != step:
                training_seconds += time.perf_counter() - segment_start
            state = {"ema": ema, "signature": signature, "step": step, "generator": g.state_dict(),
                     "discriminator": d.state_dict(), "optimizer_g": opt_g.state_dict(),
                     "optimizer_d": opt_d.state_dict(), "rngs": {k: v.get_state() for k, v in rngs.items()},
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                     "training_seconds": training_seconds, "rows": rows}
            torch.save(state, output / "latest.pt.tmp")
            (output / "latest.pt.tmp").replace(checkpoint)
            if step % config["eval_every"] == 0 or step == steps:
                torch.save({"generator": g.state_dict(), "config": config, "step": step,
                            "method": method, "seed": seed, "signature": signature}, output / f"generator_{step:06d}.pt")
            save_grid(g, fixed_z, output / f"samples_{step:06d}.png")
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

