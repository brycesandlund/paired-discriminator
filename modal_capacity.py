"""Modal entrypoints for the CIFAR pilot; no training code depends on Modal."""
from pathlib import Path
import json
import re

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("paired-discriminator-cifar10-capacity")
data_volume = modal.Volume.from_name("paired-discriminator-cifar10-data", create_if_missing=True)
run_volume = modal.Volume.from_name("paired-discriminator-cifar10-runs", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(uv_project_dir=str(ROOT), extras=["cifar"])
         .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TORCH_HOME": "/data/torch",
               "PYTHONPATH": "/root"})
         .add_local_dir(ROOT / "src" / "paired_discriminator", "/root/paired_discriminator")
         .add_local_dir(ROOT / "configs", "/project/configs")
         .add_local_file(ROOT / "pyproject.toml", "/project/pyproject.toml")
         .add_local_file(ROOT / "uv.lock", "/project/uv.lock"))


def validate_run_id(run_id):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("Use a short alphanumeric run ID, with hyphens/underscores")


@app.function(image=image, volumes={"/data": data_volume}, timeout=1800, cpu=2, memory=4096)
def prepare():
    from paired_discriminator.cifar_capacity import prepare_data
    digest = prepare_data("/data/cifar10")
    data_volume.commit()
    return digest


@app.function(image=image, gpu="A10", cpu=4, memory=16384,
              volumes={"/data": data_volume, "/runs": run_volume},
              timeout=7200, retries=1, max_containers=3, scaledown_window=2)
def train_job(run_id: str, arm: str, seed: int, steps: int):
    method = "paired"
    import shutil
    import numpy as np
    import torch
    from paired_discriminator.cifar_capacity import train
    validate_run_id(run_id)
    config = json.loads(Path(f"/project/configs/cifar10-{arm}.json").read_text())
    output = Path("/runs") / run_id / f"{arm}_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    # Each independent job owns its directory; no shared summary writes.
    for name in ["pyproject.toml", "uv.lock"]:
        dst = output / name
        if dst.exists() and dst.read_bytes() != (Path("/project") / name).read_bytes():
            raise ValueError(f"Resume dependency manifest changed: {name}")
        shutil.copy2(Path("/project") / name, dst)
    data = torch.from_numpy(np.load("/data/cifar10/train_uint8.npy"))
    labels = torch.from_numpy(np.load("/data/cifar10/train_labels.npy"))
    return train(config, method, seed, steps, output, data, labels, commit=run_volume.commit)


@app.function(image=image, gpu="A10", cpu=4, memory=16384,
              volumes={"/data": data_volume, "/runs": run_volume},
              timeout=3600, retries=1, max_containers=3, scaledown_window=2)
def evaluate_job(run_id: str, arm: str, seed: int, step: int):
    from paired_discriminator.capacity_eval import evaluate
    # A warm evaluator may retain a volume view from an earlier checkpoint.
    run_volume.reload()
    validate_run_id(run_id)
    result = evaluate(Path("/runs") / run_id / f"{arm}_seed{seed}", "/data/cifar10", step,
                      commit=run_volume.commit)
    data_volume.commit()
    return result



ARMS = ("d_batch256", "d_width128")

@app.function(image=image, gpu="A10", cpu=4, memory=16384,
              volumes={"/data": data_volume, "/runs": run_volume}, timeout=900, scaledown_window=2)
def verify():
    import numpy as np
    import torch
    from paired_discriminator.cifar_capacity import train, models, atomic_json
    from paired_discriminator.cifar_integrity import train as previous_train, models as previous_models
    base = json.loads(Path("/project/configs/cifar10-projection.json").read_text())
    all_data = torch.from_numpy(np.load("/data/cifar10/train_uint8.npy"))
    all_labels = torch.from_numpy(np.load("/data/cifar10/train_labels.npy"))
    ids = torch.cat([(all_labels == y).nonzero().flatten()[:32] for y in range(10)])
    data, labels = all_data[ids], all_labels[ids]
    root = Path("/runs/cifar10-capacity-cuda-resume-check-v1")
    results = {}
    for arm in ARMS + ("baseline",):
        config = base.copy() if arm == "baseline" else json.loads(Path(f"/project/configs/cifar10-{arm}.json").read_text())
        g, d = models(config, "paired", 0)
        oldg, oldd = previous_models(base, "paired", 0)
        for key, value in g.state_dict().items():
            torch.testing.assert_close(value, oldg.state_dict()[key], rtol=0, atol=0)
        if arm != "d_width128":
            for key, value in d.state_dict().items():
                torch.testing.assert_close(value, oldd.state_dict()[key], rtol=0, atol=0)
        config.update(log_every=10, checkpoint_every=20)
        train(config, "paired", 0, 40, root / arm / "full", data, labels)
        train(config, "paired", 0, 20, root / arm / "split", data, labels)
        train(config, "paired", 0, 40, root / arm / "split", data, labels)
        a = torch.load(root / arm / "full/latest.pt", weights_only=False)
        b = torch.load(root / arm / "split/latest.pt", weights_only=False)
        for key in ("generator", "discriminator", "rngs"):
            for name in a[key]:
                torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
        assert [(r["d_loss"], r["g_loss"]) for r in a["rows"]] == [(r["d_loss"], r["g_loss"]) for r in b["rows"]]
        if arm == "baseline":
            previous_train(config, "paired", 0, 40, root / arm / "previous", data, labels)
            old = torch.load(root / arm / "previous/latest.pt", weights_only=False)
            for key in ("generator", "discriminator", "rngs"):
                for name in a[key]:
                    torch.testing.assert_close(a[key][name], old[key][name], rtol=0, atol=0)
        results[arm] = {"exact_resume": True, "parameters_g": sum(p.numel() for p in g.parameters()), "parameters_d": sum(p.numel() for p in d.parameters())}
    result = {"arms": results, "baseline_exact_replay": True, "generator_initialization_unchanged": True, "steps": 40}
    atomic_json(root / "verification.json", result)
    run_volume.commit()
    return result

@app.local_entrypoint()
def main(run_id: str = "cifar10-capacity-v1", steps: int = 100000):
    validate_run_id(run_id)
    print("verification", json.dumps(verify.remote()), flush=True)
    calls = [(arm, train_job.spawn(run_id, arm, 0, steps)) for arm in ARMS]
    for arm, call in calls:
        print(arm, json.dumps(call.get()), flush=True)

@app.local_entrypoint()
def score_range(run_id: str = "cifar10-capacity-v1", start: int = 10000, end: int = 100000):
    import time
    validate_run_id(run_id)
    for step in range(start, end + 1, 10000):
        deadline = time.monotonic() + 10800
        while True:
            ready = []
            for arm in ARMS:
                try:
                    entries = run_volume.listdir(f"/{run_id}/{arm}_seed0")
                    ready.append(any(e.path.endswith(f"generator_{step:06d}.pt") for e in entries))
                except Exception as exc:
                    if "not found" not in str(exc).lower() and "no such" not in str(exc).lower(): raise
                    ready.append(False)
            if all(ready): break
            if time.monotonic() > deadline: raise TimeoutError(f"Checkpoint {step} missing")
            time.sleep(20)
        calls = [(arm, evaluate_job.spawn(run_id, arm, 0, step)) for arm in ARMS]
        for arm, call in calls:
            result = call.get()
            print(arm, step, json.dumps(result["heldout_metrics"]), flush=True)


@app.function(image=image, gpu=["A10", "L4", "A100", "H100", "T4"], cpu=4, memory=16384,
              volumes={"/data": data_volume, "/runs": run_volume},
              timeout=3600, retries=1, scaledown_window=2)
def evaluate_final_fallback(run_id: str = "cifar10-capacity-v1"):
    """Evaluation-only GPU fallback when A10 allocation is unavailable."""
    import torch
    from paired_discriminator.capacity_eval import evaluate
    from paired_discriminator.cifar_capacity import atomic_json
    validate_run_id(run_id)
    run_volume.reload()
    folder = Path('/runs') / run_id / 'd_width128_seed0'
    result = evaluate(folder, '/data/cifar10', 100000, commit=run_volume.commit)
    atomic_json(folder / 'evaluation_runtime_100000.json', {
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'reason': 'A10 evaluation allocation stalled; training stayed on A10',
        'training_modified': False})
    run_volume.commit()
    return result['heldout_metrics']
