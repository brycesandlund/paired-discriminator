"""Assemble prespecified 10k/50k comparisons, reusing immutable baseline runs."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from paired_discriminator.experiment import ROOT, centers
from paired_discriminator.report import make_report


def build(output):
    sources = {"vanilla": ROOT / "results/ring8-50k", "paired": ROOT / "results/ring8-50k",
               "rsgan": ROOT / "results/ring8-rsgan-50k"}
    config = json.loads((sources["rsgan"] / "config.json").read_text())
    for source in set(sources.values()):
        assert json.loads((source / "config.json").read_text()) == config
        provenance = json.loads((source / "provenance.json").read_text())
        for name, digest in provenance["source_sha256"].items():
            assert hashlib.sha256((source / "source" / name).read_bytes()).hexdigest() == digest
    output.mkdir(parents=True, exist_ok=False)
    records = []
    checked = 0
    for step in (10000, 50000):
        arm = output / f"{step // 1000}k"
        arm.mkdir()
        (arm / "config.json").write_text(json.dumps({**config, "steps": step}, indent=2) + "\n")
        for method, source in sources.items():
            for seed in config["seeds"]:
                name = f"{method}_seed{seed}"
                src = source / name
                with (src / "metrics.csv").open() as f:
                    rows = [r for r in csv.DictReader(f) if int(r["step"]) <= step]
                assert len(rows) == step // config["eval_every"] + 1
                assert int(rows[-1]["step"]) == step
                if step == 10000 and method != "rsgan":
                    sample_path = ROOT / "results/ring8-v1" / name / "final_samples.npy"
                else:
                    sample_path = src / ("samples_step10000.npy" if step == 10000 else "final_samples.npy")
                points = np.load(sample_path)
                assert points.shape == (config["eval_samples"], 2) and np.isfinite(points).all()
                distance = np.linalg.norm(points[:, None, :] - centers(config).numpy()[None, :, :], axis=2)
                nearest = distance.argmin(axis=1)
                valid = distance.min(axis=1) <= config["sigma"] * config["valid_radius_sigma"]
                mass = np.bincount(nearest[valid], minlength=config["modes"]) / len(points)
                last = rows[-1]
                np.testing.assert_allclose(mass, [float(last[f"mode_{i}_mass"]) for i in range(config["modes"])], rtol=0, atol=1e-12)
                assert (mass >= config["coverage_min_mass"]).sum() == int(last["coverage"])
                np.testing.assert_allclose(valid.mean(), float(last["valid_fraction"]), rtol=0, atol=1e-12)
                tv = .5 * (np.abs(mass - 1 / config["modes"]).sum() + 1 - mass.sum())
                np.testing.assert_allclose(tv, float(last["mode_tv"]), rtol=0, atol=1e-12)
                checked += 1
                dst = arm / name
                dst.mkdir()
                with (dst / "metrics.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(last))
                    writer.writeheader()
                    writer.writerows(rows)
                shutil.copy2(sample_path, dst / "final_samples.npy")
                metadata = json.loads((src / "metadata.json").read_text())
                metadata.update(config={**config, "steps": step}, training_seconds=float(last["training_seconds"]),
                                wall_seconds=float(last["wall_seconds"]), source_run=str(src.relative_to(ROOT)),
                                source_samples=str(sample_path.relative_to(ROOT)), derived_view=True)
                (dst / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                records.append(last)
        make_report(arm)
    with (output / "endpoints.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Relativistic GAN comparison", "",
        "RSGAN uses a shared unary critic C and logit C(real) − C(fake). D minimizes BCE with target 1; G minimizes BCE with target 0. This matches the [author's RSGAN implementation](https://github.com/AlexiaJM/RelativisticGAN#to-add-relativism-to-your-own-gans-in-pytorch-you-can-use-pieces-of-code-from-below). This is not the batch-average RaSGAN variant.", "",
        "Same five seeds, initial generator weights, data/noise streams, optimizer, batch size, and evaluation draws. RSGAN and vanilla also have identical initial critic weights and 17,025 critic parameters; paired has 17,281. No method-specific tuning. Each D update sees 256 real and 256 fake samples. RSGAN makes 256 relative decisions, paired 256 joint decisions, and vanilla 512 unary decisions. Equal update budgets are not equal FLOPs.", "",
        "RSGAN is trained once to 50k, with the prespecified 10k evaluation retained. These are two training budgets on the same trajectories, not independent arms. Existing vanilla/paired results are reused unchanged. No best-checkpoint selection.", "",
        "Mean ± sample SD across seeds:", "",
        "| Updates | Method | Modes covered | Valid samples | Mode TV ↓ |", "|---:|---|---:|---:|---:|"]
    for step in (10000, 50000):
        for method in sources:
            group = [r for r in records if int(r["step"]) == step and r["method"] == method]
            c, v, t = [np.array([float(r[k]) for r in group]) for k in ("coverage", "valid_fraction", "mode_tv")]
            lines.append(f"| {step:,} | {method} | {c.mean():.2f} ± {c.std(ddof=1):.2f} | {v.mean():.1%} ± {v.std(ddof=1):.1%} | {t.mean():.3f} ± {t.std(ddof=1):.3f} |")
    lines += ["", "## Individual seeds", "", "| Updates | Method | Seed | Coverage | Valid samples | Mode TV |", "|---:|---|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['step']} | {r['method']} | {r['seed']} | {r['coverage']} | {float(r['valid_fraction']):.1%} | {float(r['mode_tv']):.3f} |")
    lines += ["", "## Late window: 40k–50k", "", "Average within each seed over recorded evaluations, then across seeds. These correlated evaluations are not extra replicates.", "", "| Method | Coverage | Valid samples | Mode TV |", "|---|---:|---:|---:|"]
    for method, source in sources.items():
        means = []
        for seed in config["seeds"]:
            with (source / f"{method}_seed{seed}" / "metrics.csv").open() as f:
                late = [r for r in csv.DictReader(f) if int(r["step"]) >= 40000]
            means.append([np.mean([float(r[k]) for r in late]) for k in ("coverage", "valid_fraction", "mode_tv")])
        c, v, t = np.mean(means, axis=0)
        lines.append(f"| {method} | {c:.2f} | {v:.1%} | {t:.3f} |")
    lines += ["", "## Figures and data", "", "[10k report](10k/REPORT.md) · [50k report](50k/REPORT.md) · [All seed endpoints](endpoints.csv)", "", "![Training curves](50k/training_curves.png)", "", "![50k samples](50k/final_samples.png)", "", "This remains one untuned synthetic task. Metrics assess mode coverage, accepted mass balance, and proximity to centers, not full within-mode distribution quality. Results do not establish general superiority or the mechanism responsible.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    (output / "verification.json").write_text(json.dumps({"sample_sets_independently_verified": checked,
        "source_hashes_verified": True, "configs_match": True,
        "source_suites": {k: str(v.relative_to(ROOT)) for k, v in sources.items()}}, indent=2) + "\n")
    shutil.copy2(__file__, output / "comparison_source.py")


if __name__ == "__main__":
    import sys
    build(Path(sys.argv[1]))
