from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from model import ModelConfig, run_simulation


NETWORK_ORDER = ["random", "small_world", "scale_free"]
INTERVENTION_ORDER = ["none", "random", "centrality"]


def plot_main_results(csv_path: str, output_dir: str) -> None:
    df = pd.read_csv(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    required = {
        "network_type",
        "intervention",
        "final_reach",
        "peak_believer_fraction",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    summary = (
        df.groupby(["network_type", "intervention"])["final_reach"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    mean_pivot = summary.pivot(
        index="network_type", columns="intervention", values="mean"
    ).reindex(index=NETWORK_ORDER, columns=INTERVENTION_ORDER)
    std_pivot = summary.pivot(
        index="network_type", columns="intervention", values="std"
    ).reindex(index=NETWORK_ORDER, columns=INTERVENTION_ORDER)

    ax = mean_pivot.plot(
        kind="bar",
        yerr=std_pivot,
        capsize=3,
        figsize=(9, 5.5),
    )
    ax.set_ylabel("Mean final reach")
    ax.set_xlabel("Network structure")
    ax.set_ylim(0, 1.05)
    ax.set_title("Misinformation reach by network and intervention")
    ax.legend(title="Intervention")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "final_reach_bar.png", dpi=200)
    plt.close()

    peak_summary = (
        df.groupby(["network_type", "intervention"])["peak_believer_fraction"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    peak_mean = peak_summary.pivot(
        index="network_type", columns="intervention", values="mean"
    ).reindex(index=NETWORK_ORDER, columns=INTERVENTION_ORDER)
    peak_std = peak_summary.pivot(
        index="network_type", columns="intervention", values="std"
    ).reindex(index=NETWORK_ORDER, columns=INTERVENTION_ORDER)

    ax = peak_mean.plot(
        kind="bar",
        yerr=peak_std,
        capsize=3,
        figsize=(9, 5.5),
    )
    ax.set_ylabel("Mean peak believer fraction")
    ax.set_xlabel("Network structure")
    ax.set_ylim(0, 1.05)
    ax.set_title("Peak misinformation prevalence")
    ax.legend(title="Intervention")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "peak_believers_bar.png", dpi=200)
    plt.close()

    summary.to_csv(out / "main_summary.csv", index=False)
    peak_summary.to_csv(out / "peak_summary.csv", index=False)


def plot_representative_time_series(output_dir: str, n: int = 300, seed: int = 42) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    network_type = "scale_free"
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for intervention in INTERVENTION_ORDER:
        cfg = ModelConfig(
            n=n,
            mean_degree=8,
            network_type=network_type,
            intervention=intervention,
            fact_checker_fraction=0.05 if intervention != "none" else 0.0,
            misinformation_prob=0.15,
            correction_prob=0.35,
            resistant_fraction=0.05,
            initial_believers=5,
            max_steps=150,
            seed=seed,
        )
        history, _, _ = run_simulation(cfg)
        hist = pd.DataFrame(history)
        ax.plot(hist["step"], hist["believer"] / cfg.n, label=intervention)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Believer fraction")
    ax.set_ylim(bottom=0)
    ax.set_title("Representative misinformation dynamics (scale-free network)")
    ax.legend(title="Intervention")
    plt.tight_layout()
    plt.savefig(out / "representative_time_series.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create project result figures")
    parser.add_argument(
        "--input",
        default="../results/main_experiment.csv",
        help="Experiment CSV created by experiments.py",
    )
    parser.add_argument(
        "--output-dir",
        default="../results/figures",
        help="Directory in which figures and summaries are written",
    )
    parser.add_argument(
        "--time-series",
        action="store_true",
        help="Also create one representative time-series figure",
    )
    parser.add_argument("--time-series-n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plot_main_results(args.input, args.output_dir)
    if args.time_series:
        plot_representative_time_series(
            args.output_dir, n=args.time_series_n, seed=args.seed
        )

    print(f"Figures saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
