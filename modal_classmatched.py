"""Modal entrypoints for the CIFAR pilot; no training code depends on Modal."""
from pathlib import Path
import json
import re

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("paired-discriminator-cifar10-classmatched")
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
    from paired_discriminator.cifar_classmatched import prepare_data
    digest = prepare_data("/data/cifar10")
    data_volume.commit()
    return digest


@app.function(image=image, gpu="A10", cpu=4, memory=16384,
              volumes={"/data": data_volume, "/runs": run_volume},
              timeout=7200, retries=1, max_containers=3, scaledown_window=2)
def train_job(run_id: str, method: str, seed: int, steps: int):
    import shutil
    import numpy as np
    import torch
    from paired_discriminator.cifar_classmatched import train
    validate_run_id(run_id)
    config = json.loads(Path("/project/configs/cifar10-classmatched.json").read_text())
    output = Path("/runs") / run_id / f"{method}_seed{seed}"
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
              timeout=3600, retries=1, max_containers=1, scaledown_window=2)
def evaluate_job(run_id: str, method: str, seed: int, step: int):
    from paired_discriminator.cifar_classmatched import evaluate
    validate_run_id(run_id)
    result = evaluate(Path("/runs") / run_id / f"{method}_seed{seed}", "/data/cifar10", step,
                      commit=run_volume.commit)
    data_volume.commit()
    return result


@app.local_entrypoint()
def main(run_id: str = "cifar10-classmatched-smoke-v1", steps: int = 300, seed: int = 0, evaluate: bool = False):
    validate_run_id(run_id)
    print(prepare.remote())
    calls = [(method, train_job.spawn(run_id, method, seed, steps)) for method in ("vanilla", "rsgan", "paired")]
    for method, call in calls:
        print(method, json.dumps(call.get()))
    if evaluate:
        for step in (s for s in (10000, 50000) if s <= steps):
            for method in ("vanilla", "rsgan", "paired"):
                print(method, json.dumps(evaluate_job.remote(run_id, method, seed, step)))


@app.function(image=image, gpu="A10", cpu=4, memory=8192,
              volumes={"/data": data_volume, "/runs": run_volume}, timeout=900, scaledown_window=2)
def verify_resume():
    import numpy as np
    import torch
    from paired_discriminator.cifar_classmatched import train, atomic_json
    config = json.loads(Path("/project/configs/cifar10-classmatched.json").read_text())
    config.update(log_every=10, checkpoint_every=20)
    all_data = torch.from_numpy(np.load("/data/cifar10/train_uint8.npy"))
    all_labels = torch.from_numpy(np.load("/data/cifar10/train_labels.npy"))
    indices = torch.cat([(all_labels == y).nonzero().flatten()[:32] for y in range(10)])
    data, labels = all_data[indices], all_labels[indices]
    root = Path("/runs/cifar10-classmatched-cuda-resume-check-v1")
    for method in ("vanilla", "rsgan", "paired"):
        train(config, method, 0, 40, root / method / "full", data, labels)
        train(config, method, 0, 20, root / method / "split", data, labels)
        train(config, method, 0, 40, root / method / "split", data, labels)
        a = torch.load(root / method / "full/latest.pt", map_location="cpu", weights_only=False)
        b = torch.load(root / method / "split/latest.pt", map_location="cpu", weights_only=False)
        for key in ("generator", "discriminator", "rngs"):
            for name in a[key]:
                torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
        for x, y in zip(a["rows"], b["rows"]):
            assert x["d_loss"] == y["d_loss"] and x["g_loss"] == y["g_loss"]
    result = {"cuda_exact_resume": True, "methods": ["vanilla", "rsgan", "paired"], "steps": 40, "split_at": 20}
    atomic_json(root / "verification.json", result)
    run_volume.commit()
    return result


@app.local_entrypoint()
def score(run_id: str, step: int = 10000, seed: int = 0):
    """Score retained checkpoints while longer training continues."""
    validate_run_id(run_id)
    for method in ("vanilla", "rsgan", "paired"):
        print(method, json.dumps(evaluate_job.remote(run_id, method, seed, step)))
