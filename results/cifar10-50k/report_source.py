"""Summarize downloaded CIFAR pilot artifacts without rerunning training."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paired_discriminator.cifar import METHODS

COLORS = {"vanilla": "#4175b5", "paired": "#de8541", "rsgan": "#39966c"}


def report(root):
    root = Path(root)
    records = []
    runs = []
    for directory in sorted(root.glob("*_seed*")):
        meta = json.loads((directory / "run.json").read_text())
        assert hashlib.sha256((directory / "training_source.py").read_bytes()).hexdigest() == meta["source_sha256"]
        status = json.loads((directory / "status.json").read_text())
        with (directory / "training.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert int(rows[-1]["step"]) == status["step"]
        assert all(np.isfinite(float(r[k])) for r in rows for k in ("d_loss", "g_loss"))
        runs.append((directory, meta, status, rows))
        for path in sorted(directory.glob("metrics_*.json")):
            record = json.loads(path.read_text())
            assert record["method"] == meta["method"] and record["seed"] == meta["seed"]
            assert all(np.isfinite(record[k]) for k in ("frechet_inception_distance", "kernel_inception_distance_mean", "precision", "recall"))
            records.append(record)
    assert runs and records, "No completed evaluation artifacts"
    assert len({r[1]["dataset_sha256"] for r in runs}) == 1
    assert len({json.dumps(r[1]["config"], sort_keys=True) for r in runs}) == 1
    with (root / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, method in zip(axes, METHODS):
        for _, meta, _, rows in runs:
            if meta["method"] != method:
                continue
            x = [int(r["step"]) for r in rows]
            ax.plot(x, [float(r["d_loss"]) for r in rows], label="D")
            ax.plot(x, [float(r["g_loss"]) for r in rows], label="G")
        ax.set(title=method, xlabel="Generator updates", ylabel="BCE loss")
        ax.legend(frameon=False)
    fig.suptitle("Training diagnostics · loss magnitudes are not comparable across objectives")
    fig.tight_layout()
    fig.savefig(root / "training_losses.png", dpi=150)
    plt.close(fig)
    for step in sorted({r["step"] for r in records}):
        selected = [(p, m) for p, m, _, _ in runs if (p / f"samples_{step:06d}.png").exists()]
        fig, axes = plt.subplots(1, len(selected), figsize=(4.5 * len(selected), 4.8), squeeze=False)
        for ax, (p, meta) in zip(axes[0], sorted(selected, key=lambda r: (METHODS.index(r[1]["method"]), r[1]["seed"]))):
            ax.imshow(plt.imread(p / f"samples_{step:06d}.png"), interpolation="nearest")
            ax.set_title(f"{meta['method']} · seed {meta['seed']}")
            ax.axis("off")
        fig.suptitle(f"CIFAR-10 · {step:,} updates · identical 64 fixed latent draws")
        fig.tight_layout()
        fig.savefig(root / f"samples_{step:06d}.png", dpi=150)
        plt.close(fig)
    lines = ["# CIFAR-10: Modal pilot", "", "One matched training seed per method; fixed settings, no method-specific tuning. All runs used NVIDIA A10, float32, TF32 disabled. The objective here is to validate the image experiment and obtain an initial comparison, not to establish superiority.", "",
        "## Evaluation", "", "10,000 generated images versus the same fixed 10,000-image training subset. FID/KID and precision/recall use torch-fidelity's Inception-v3-compatible 2048-D features. This is FID-10k, not a standard FID-50k result. Higher precision/recall is better; lower FID/KID is better. KID below is multiplied by 1,000.", "",
        "| Method | Seed | Updates | FID ↓ | KID ×1000 ↓ | Precision ↑ | Recall ↑ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(records, key=lambda r: (r["step"], METHODS.index(r["method"]), r["seed"])):
        lines.append(f"| {r['method']} | {r['seed']} | {r['step']:,} | {r['frechet_inception_distance']:.2f} | {1000*r['kernel_inception_distance_mean']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} |")
    lines += ["", "Precision and recall are feature-space diagnostics, not literal class coverage. No seed uncertainty can be estimated from this pilot. The evaluation uses a training-set reference; it does not independently test memorization.", "", "## Runtime and architecture", "", "| Method | D parameters | G parameters | Training seconds | Evaluation seconds |", "|---|---:|---:|---:|---:|"]
    for _, meta, status, _ in sorted(runs, key=lambda r: METHODS.index(r[1]["method"])):
        seconds = sum(r["evaluation_seconds"] for r in records if (r["method"], r["seed"]) == (meta["method"], meta["seed"]))
        lines.append(f"| {meta['method']} | {meta['parameters_d']:,} | {meta['parameters_g']:,} | {status['training_seconds']:.1f} | {seconds:.1f} |")
    lines += ["", "Training time excludes checkpoint I/O, sample grids, setup, and evaluation. It is not a Modal billing report. All methods see the same number of real/fake images per D update; computation differs. Full protocol: [CIFAR10.md](CIFAR10.md).", "", "## Samples", ""]
    for step in sorted({r["step"] for r in records}):
        lines += [f"![Fixed samples at {step}](samples_{step:06d}.png)", ""]
    lines += ["## Training diagnostics", "", "![Training loss](training_losses.png)", "", "Loss magnitudes have different meanings across objectives and are not a ranking metric.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    shutil.copy2(Path(__file__).parents[2] / "docs/CIFAR10.md", root / "CIFAR10.md")
    shutil.copy2(__file__, root / "report_source.py")
    (root / "verification.json").write_text(json.dumps({"run_count": len(runs), "evaluation_count": len(records),
        "source_hashes_verified": True, "matched_config_and_dataset": True, "finite_metrics_and_losses": True}, indent=2) + "\n")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    import sys
    report(sys.argv[1])
