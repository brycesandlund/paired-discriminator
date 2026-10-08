"""Verify and report a CIFAR checkpoint extension against preserved pilot artifacts."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paired_discriminator.cifar import METHODS
from paired_discriminator.cifar_report import COLORS, report


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def compare(baseline, extension):
    baseline, extension = Path(baseline), Path(extension)
    matched_rows = 0
    records = []
    for method in METHODS:
        name = f"{method}_seed0"
        old, new = baseline / name, extension / name
        assert json.loads((old / "run.json").read_text()) == json.loads((new / "run.json").read_text())
        for filename in ("training_source.py", "pyproject.toml", "uv.lock", "metrics_010000.json",
                         "samples_010000.png", "generator_010000.pt"):
            assert hashlib.sha256((old / filename).read_bytes()).digest() == hashlib.sha256((new / filename).read_bytes()).digest(), filename
        before, after = read_csv(old / "training.csv"), read_csv(new / "training.csv")
        assert int(before[-1]["step"]) == 10000
        assert int(after[-1]["step"]) == 50000
        assert after[:len(before)] == before, "Original training history changed"
        matched_rows += len(before)
        status = json.loads((new / "status.json").read_text())
        assert status["step"] == 50000 and status["target_steps"] == 50000
        a = json.loads((new / "metrics_010000.json").read_text())
        b = json.loads((new / "metrics_050000.json").read_text())
        for field in ("samples_generated", "samples_real", "reference", "features", "torch_fidelity", "seed", "method"):
            assert a[field] == b[field], field
        assert a["step"] == 10000 and b["step"] == 50000
        records.append((method, a, b))
    report(extension)
    verification = {"matched_original_training_rows": matched_rows,
        "source_config_runtime_and_dependency_manifests_unchanged": True,
        "original_10k_generator_grids_and_metrics_unchanged": True,
        "evaluation_protocol_unchanged": True, "all_methods_reached_50000": True,
        "methods": list(METHODS), "seed": 0}
    (extension / "extension_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, field, title in zip(axes, ("frechet_inception_distance", "precision", "recall"),
                                ("FID ↓", "Precision ↑", "Recall ↑")):
        for method, a, b in records:
            ax.plot([10000, 50000], [a[field], b[field]], "o-", color=COLORS[method], label=method)
        ax.set(title=title, xlabel="Generator updates", xticks=[10000, 50000], ylim=(0, None))
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False)
    fig.suptitle("CIFAR-10 · seed 0 · two evaluated checkpoints, 10k images per evaluation")
    fig.tight_layout()
    fig.savefig(extension / "extension_metrics.png", dpi=150)
    plt.close(fig)
    lines = ["# CIFAR-10 extension: 10k → 50k updates", "",
        "Same three seed-0 trajectories, resumed from their 10k model/optimizer/RNG checkpoints. No training setting, source, or evaluation protocol changed. Original 10k artifacts are preserved. This is still one seed, not evidence of general superiority.", "",
        "Each metric uses 10,000 generated images and the same fixed 10,000-real-image training subset. Thus the 50k-update endpoint is still FID-10k, not FID-50k. Precision and recall are Inception feature-space metrics, not class coverage.", "",
        "| Method | FID 10k → 50k ↓ | KID ×1000 10k → 50k ↓ | Precision 10k → 50k ↑ | Recall 10k → 50k ↑ |",
        "|---|---:|---:|---:|---:|"]
    for method, a, b in records:
        lines.append(f"| {method} | {a['frechet_inception_distance']:.2f} → {b['frechet_inception_distance']:.2f} | "
                     f"{1000*a['kernel_inception_distance_mean']:.2f} → {1000*b['kernel_inception_distance_mean']:.2f} | "
                     f"{a['precision']:.3f} → {b['precision']:.3f} | {a['recall']:.3f} → {b['recall']:.3f} |")
    lines += ["", "![Endpoint metrics](extension_metrics.png)", "",
        "Lines connect only the two evaluated checkpoints; intermediate metric values were not measured.", "",
        "## Fixed samples at 50k", "", "![50k samples](samples_050000.png)", "",
        "## Fixed samples at 10k", "", "![10k samples](samples_010000.png)", "",
        "## Verification", "", f"All {matched_rows} original training-log rows match exactly, including timing. The original generator checkpoint, sample grid, evaluation JSON, source, dependency manifests, and run metadata are unchanged for every method. The earlier CPU and A10 tests separately established exact interrupted-versus-uninterrupted resume for all three methods.", "",
        "[Full metrics and diagnostics](REPORT.md) · [Protocol](CIFAR10.md) · [Verification](extension_verification.json)", ""]
    (extension / "EXTENSION.md").write_text("\n".join(lines))
    shutil.copy2(__file__, extension / "comparison_source.py")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    import sys
    compare(sys.argv[1], sys.argv[2])
