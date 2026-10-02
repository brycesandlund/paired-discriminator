"""Build plots and a compact report from saved experiment results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"vanilla": "#4175b5", "paired": "#de8541"}


def make_report(root: Path):
    config = json.loads((root / "config.json").read_text())
    runs = []
    for path in sorted(root.glob("*_seed*/metrics.csv")):
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        meta = json.loads((path.parent / "metadata.json").read_text())
        runs.append({"path": path.parent, "method": rows[0]["method"],
                     "seed": int(rows[0]["seed"]), "rows": rows, "meta": meta})
    methods = [m for m in ["vanilla", "paired"] if any(r["method"] == m for r in runs)]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fields = [("coverage", "Modes covered ↑", (0, config["modes"] + .3)),
              ("valid_fraction", "Valid samples ↑", (0, 1.02)),
              ("mode_tv", "Mode imbalance (TV) ↓", (0, 1.02))]
    summary = {}
    for method in methods:
        group = [r for r in runs if r["method"] == method]
        summary[method] = {}
        for ax, (field, title, ylim) in zip(axes, fields):
            arrays = np.array([[float(row[field]) for row in run["rows"]] for run in group])
            steps = [int(row["step"]) for row in group[0]["rows"]]
            for array in arrays:
                ax.plot(steps, array, color=COLORS[method], alpha=.2, linewidth=.8)
            mean = arrays.mean(axis=0)
            sd = arrays.std(axis=0, ddof=1) if len(group) > 1 else np.zeros_like(mean)
            ax.plot(steps, mean, color=COLORS[method], label=method, linewidth=2)
            ax.fill_between(steps, mean - sd, mean + sd, color=COLORS[method], alpha=.12)
            ax.set(title=title, xlabel="Generator updates", ylim=ylim)
            ax.grid(alpha=.15)
            summary[method][field] = {"mean": float(mean[-1]), "sd": float(sd[-1]),
                                      "values": arrays[:, -1].tolist()}
        times = [r["meta"]["training_seconds"] for r in group]
        summary[method]["training_seconds"] = {"mean": float(np.mean(times)), "values": times}
        summary[method]["parameters_d"] = group[0]["meta"]["parameters_d"]
    axes[0].legend(frameon=False)
    fig.suptitle(f"{config['modes']}-mode ring · standard training · mean ± seed SD", fontsize=14)
    fig.tight_layout()
    fig.savefig(root / "training_curves.png")
    plt.close(fig)

    seeds = sorted({r["seed"] for r in runs})
    fig, axes = plt.subplots(len(methods), len(seeds), figsize=(3 * len(seeds), 3 * len(methods)),
                             squeeze=False, sharex=True, sharey=True)
    angles = np.arange(config["modes"]) * 2 * np.pi / config["modes"]
    target = config["radius"] * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    limit = config["radius"] + 1
    for run in runs:
        ax = axes[methods.index(run["method"]), seeds.index(run["seed"])]
        samples = np.load(run["path"] / "final_samples.npy")
        ax.scatter(samples[:3000, 0], samples[:3000, 1], s=2, alpha=.25,
                   color=COLORS[run["method"]], rasterized=True)
        for center in target:
            ax.add_patch(plt.Circle(center, config["valid_radius_sigma"] * config["sigma"],
                                    fill=False, color="#333333", linewidth=.8))
        last = run["rows"][-1]
        ax.set(title=f"{run['method']} · seed {run['seed']}\n"
                      f"{last['coverage']}/{config['modes']} modes · {float(last['valid_fraction']):.1%} valid",
               xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    fig.suptitle("Final generated samples · circles mark the fixed acceptance regions", fontsize=14)
    fig.tight_layout()
    fig.savefig(root / "final_samples.png")
    plt.close(fig)

    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 3 * len(methods)), squeeze=False)
    for index, method in enumerate(methods):
        ax = axes[index, 0]
        group = [r for r in runs if r["method"] == method]
        values = np.array([[float(r["rows"][-1][f"mode_{i}_mass"]) for i in range(config["modes"])]
                           for r in group])
        ax.bar(np.arange(config["modes"]), values.mean(axis=0), color=COLORS[method], alpha=.7)
        for row in values:
            ax.scatter(np.arange(config["modes"]), row, color="#333333", s=12, alpha=.6)
        ax.axhline(1 / config["modes"], color="#333333", linestyle="--", label="Target mass")
        ax.set(title=method, xlabel="Mode", ylabel="Fraction of all samples", xticks=np.arange(config["modes"]))
        ax.legend(frameon=False)
    fig.suptitle("Accepted mass per mode · dots are individual seeds", fontsize=14)
    fig.tight_layout()
    fig.savefig(root / "mode_mass.png")
    plt.close(fig)

    with (root / "final_metrics.csv").open("w", newline="") as handle:
        final = [r["rows"][-1] for r in runs]
        writer = csv.DictWriter(handle, fieldnames=list(final[0]))
        writer.writeheader()
        writer.writerows(final)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Vanilla versus paired discriminator: eight-mode ring", "",
             f"{len(seeds)} paired seeds; {config['steps']:,} generator updates per run; "
             f"batch {config['batch_size']}; CPU threads {config['threads']}; device {config['device']}.", "",
             "Results are at the prespecified final step, not each run's best checkpoint. "
             "Intervals below are sample standard deviations across seeds, not confidence intervals.", "",
             "| Method | Modes covered | Valid samples | Mode TV ↓ | Mean training time | D parameters |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, stats in summary.items():
        c, v, t = [stats[k] for k in ["coverage", "valid_fraction", "mode_tv"]]
        lines.append(f"| {method} | {c['mean']:.2f} ± {c['sd']:.2f} | "
                     f"{v['mean']:.1%} ± {v['sd']:.1%} | {t['mean']:.3f} ± {t['sd']:.3f} | "
                     f"{stats['training_seconds']['mean']:.1f} s | {stats['parameters_d']:,} |")
    lines += ["", "## Definitions", "",
              f"Data: {config['modes']} equally weighted isotropic Gaussians, ring radius {config['radius']}, "
              f"per-axis standard deviation {config['sigma']}. Evaluation uses {config['eval_samples']:,} "
              "fixed latent draws, separate from training randomness.", "",
              f"A sample is valid within {config['valid_radius_sigma']}σ of its nearest center. "
              f"Coverage requires at least {config['coverage_min_mass']:.0%} of all evaluation samples per mode. "
              "Per-mode mass includes only valid samples. TV compares these masses plus an invalid category "
              "to uniform target weights and zero invalid mass. Even true 2-D Gaussians have about 1.1% "
              "of their mass outside a 3σ radius, so their finite-sample validity is not 100%.", "",
              "## Controls and limits", "",
              "Both methods share the generator architecture, initial generator weights, independent matched "
              "real/noise streams, optimizer settings, batch size, 1:1 update ratio, and seed list. "
              "The paired discriminator has two extra input features and slightly more parameters. "
              "Both use mean BCE for discriminator decisions and a non-saturating generator loss. "
              "Vanilla makes two unary decisions per real/fake pair; paired makes one randomized-slot decision. "
              "Equal update counts do not imply equal FLOPs. Timing excludes evaluations and checkpoint I/O.", "",
              "This is one untuned configuration on an easy synthetic distribution; it is not evidence "
              "of general superiority or of a particular gradient mechanism. Seed-level comparisons matter.", "",
              "## Plots", "", "![Training curves](training_curves.png)", "",
              "![Final samples](final_samples.png)", "", "![Mode masses](mode_mass.png)", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    make_report(Path(sys.argv[1]))
