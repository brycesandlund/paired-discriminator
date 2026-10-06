"""Compare a deterministic extension against the original experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def compare(baseline: Path, extension: Path):
    old_config = json.loads((baseline / "config.json").read_text())
    config = json.loads((extension / "config.json").read_text())
    assert config["steps"] > old_config["steps"]
    assert {k: v for k, v in config.items() if k != "steps"} == {
        k: v for k, v in old_config.items() if k != "steps"}
    training_source = "src/paired_discriminator/experiment.py"
    assert (baseline / "source" / training_source).read_bytes() == (
        extension / "source" / training_source).read_bytes(), "Training code changed"
    provenance = json.loads((extension / "provenance.json").read_text())
    for name, digest in provenance["source_sha256"].items():
        assert hashlib.sha256((extension / "source" / name).read_bytes()).hexdigest() == digest

    metrics = ["coverage", "valid_fraction", "mode_tv"]
    records = []
    comparisons = 0
    window_start = max(old_config["steps"], config["steps"] - 10000)
    for method in ["vanilla", "paired"]:
        for seed in config["seeds"]:
            name = f"{method}_seed{seed}"
            old = read_rows(baseline / name / "metrics.csv")
            new = read_rows(extension / name / "metrics.csv")
            by_step = {int(row["step"]): row for row in new}
            assert len(new) == config["steps"] // config["eval_every"] + 1
            assert int(new[-1]["step"]) == config["steps"]
            for row in old:
                rerun = by_step[int(row["step"])]
                for key in row:
                    if key not in {"training_seconds", "wall_seconds"}:
                        assert row[key] == rerun[key], (name, row["step"], key)
                comparisons += 1
            # Independently re-evaluate saved samples using NumPy distances.
            points = np.load(extension / name / "final_samples.npy")
            from paired_discriminator.experiment import centers
            target = centers(config).numpy()
            distances = np.linalg.norm(points[:, None, :] - target[None, :, :], axis=2)
            ids = distances.argmin(axis=1)
            valid = distances.min(axis=1) <= config["valid_radius_sigma"] * config["sigma"]
            counts = np.bincount(ids[valid], minlength=config["modes"])
            mass = counts / len(points)
            final = new[-1]
            np.testing.assert_allclose(mass, [float(final[f"mode_{i}_mass"]) for i in range(config["modes"])], atol=1e-12)
            assert int((mass >= config["coverage_min_mass"]).sum()) == int(final["coverage"])
            np.testing.assert_allclose(valid.mean(), float(final["valid_fraction"]), atol=1e-12)
            tv = .5 * (np.abs(mass - 1 / config["modes"]).sum() + 1 - mass.sum())
            np.testing.assert_allclose(tv, float(final["mode_tv"]), atol=1e-12)
            assert np.isfinite(points).all()
            late = [row for row in new if int(row["step"]) >= window_start]
            record = {"method": method, "seed": seed}
            for metric in metrics:
                record[f"original_{metric}"] = float(old[-1][metric])
                record[f"extended_{metric}"] = float(final[metric])
                record[f"late_mean_{metric}"] = float(np.mean([float(row[metric]) for row in late]))
            record["late_min_coverage"] = min(int(row["coverage"]) for row in late)
            record["late_full_coverage_fraction"] = float(np.mean([
                int(row["coverage"]) == config["modes"] for row in late]))
            records.append(record)

    with (extension / "extension_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    verification = {"matched_prefix_evaluations": comparisons,
                    "config_difference": {"steps": [old_config["steps"], config["steps"]]},
                    "training_source_identical": True, "source_hashes_verified": True,
                    "final_metrics_independently_recomputed": True,
                    "late_window": [window_start, config["steps"]]}
    (extension / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    lines = ["# Extension: 10,000 → 50,000 updates", "",
             "Same five seeds, models, optimizer settings, random streams, and evaluation samples. "
             "The deterministic training run was replayed from initialization and extended; "
             f"all {comparisons} original evaluation rows match exactly (excluding timing).", "",
             "## Endpoint comparison", "",
             "Mean ± sample SD across five seeds. Lower mode TV is better; it penalizes both "
             "incorrect mode masses and invalid samples.", "",
             "| Method | Updates | Modes covered | Valid samples | Mode TV |",
             "|---|---:|---:|---:|---:|"]
    for method in ["vanilla", "paired"]:
        group = [r for r in records if r["method"] == method]
        for prefix, step in [("original", old_config["steps"]), ("extended", config["steps"])]:
            values = [np.array([r[f"{prefix}_{metric}"] for r in group]) for metric in metrics]
            c, v, t = values
            lines.append(f"| {method} | {step:,} | {c.mean():.2f} ± {c.std(ddof=1):.2f} | "
                         f"{v.mean():.1%} ± {v.std(ddof=1):.1%} | {t.mean():.3f} ± {t.std(ddof=1):.3f} |")
    lines += ["", "## Seed-level endpoints", "",
              "| Method | Seed | Modes at 10k → 50k | Valid at 10k → 50k | TV at 10k → 50k |",
              "|---|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['method']} | {r['seed']} | {r['original_coverage']:g} → {r['extended_coverage']:g} | "
                     f"{r['original_valid_fraction']:.1%} → {r['extended_valid_fraction']:.1%} | "
                     f"{r['original_mode_tv']:.3f} → {r['extended_mode_tv']:.3f} |")
    lines += ["", f"## Late training: {window_start:,}–{config['steps']:,} updates", "",
              "These are averages over the recorded late-window evaluations within each seed, "
              "then averaged across seeds. Checkpoints are correlated, not additional independent replicates. "
              "Full-coverage checks refer only to recorded evaluations, not every training step.", "",
              "| Method | Mean coverage | Mean validity | Mean TV | Full-coverage checks |",
              "|---|---:|---:|---:|---:|"]
    for method in ["vanilla", "paired"]:
        group = [r for r in records if r["method"] == method]
        means = [np.mean([r[f"late_mean_{metric}"] for r in group]) for metric in metrics]
        full = np.mean([r["late_full_coverage_fraction"] for r in group])
        lines.append(f"| {method} | {means[0]:.2f} | {means[1]:.1%} | {means[2]:.3f} | {full:.1%} |")
    lines += ["", "## Plots", "", "![Training curves](training_curves.png)", "",
              "![Final generated samples](final_samples.png)", "", "![Accepted mode mass](mode_mass.png)", "",
              "## Limits", "", "This extends one untuned setup. It does not prove convergence, "
              "general superiority, or that the proposed reference-dependent gradient mechanism explains "
              "any difference. Both methods see 256 real and 256 generated samples in each discriminator "
              "update; unary makes two decisions per pair, while paired makes one joint decision. "
              "Full metric definitions and parameter counts are in REPORT.md.", ""]
    (extension / "EXTENSION.md").write_text("\n".join(lines))
    shutil.copyfile(__file__, extension / "comparison_source.py")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("extension", type=Path)
    args = parser.parse_args()
    compare(args.baseline, args.extension)
