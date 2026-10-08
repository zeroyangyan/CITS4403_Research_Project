from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NETWORKS = ["random", "small_world", "scale_free"]
STRATEGIES = ["none", "random", "centrality"]
LEVELS = ["homogeneous", "low", "high"]

METRICS = {
    "final_reach": "Final reach",
    "peak_believer_fraction": "Peak believer fraction",
    "time_to_peak": "Time to peak",
    "ended_with_active_believers":
        "Proportion ending with active believers",
}


def wilson_interval(successes, count):
    z = 1.96
    p = successes / count
    denominator = 1 + z * z / count

    centre = (
        p + z * z / (2 * count)
    ) / denominator

    half_width = (
        z * np.sqrt(
            p * (1 - p) / count
            + z * z / (4 * count * count)
        ) / denominator
    )

    return centre - half_width, centre + half_width


def plot_main_results(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    required = {
        "network_type",
        "heterogeneity",
        "intervention",
        *METRICS.keys(),
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    active = (
        df["ended_with_active_believers"]
        .astype(str)
        .str.lower()
    )

    if not active.isin(
        ["true", "false", "1", "0"]
    ).all():
        raise ValueError("Invalid active-believer values")

    df["ended_with_active_believers"] = (
        active.isin(["true", "1"]).astype(int)
    )

    group_columns = [
        "network_type",
        "heterogeneity",
        "intervention",
    ]

    rows = []

    for keys, group in df.groupby(group_columns):
        for metric in METRICS:
            values = group[metric]
            count = len(values)

            row = dict(zip(group_columns, keys))
            row.update({
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(),
                "count": count,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
            })

            if metric == "ended_with_active_believers":
                low, high = wilson_interval(
                    values.sum(), count
                )
                row["ci95_low"] = low
                row["ci95_high"] = high

            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(
        output / "summary.csv", index=False
    )

    present_levels = [
        level for level in LEVELS
        if level in set(df["heterogeneity"])
    ]

    for metric, label in METRICS.items():
        fig, axes = plt.subplots(
            1, 3,
            figsize=(15, 5),
            sharey=True,
        )

        for ax, network in zip(axes, NETWORKS):
            x = np.arange(len(present_levels))
            width = 0.24

            for index, strategy in enumerate(STRATEGIES):
                subset = summary[
                    (summary["network_type"] == network)
                    & (summary["intervention"] == strategy)
                    & (summary["metric"] == metric)
                ].set_index(
                    "heterogeneity"
                ).reindex(present_levels)

                means = subset["mean"].to_numpy()

                if metric == "ended_with_active_believers":
                    errors = np.vstack([
                        means - subset["ci95_low"].to_numpy(),
                        subset["ci95_high"].to_numpy() - means,
                    ])
                else:
                    errors = (
                        subset["std"]
                        .fillna(0)
                        .to_numpy()
                    )

                ax.bar(
                    x + (index - 1) * width,
                    means,
                    width,
                    yerr=errors,
                    capsize=3,
                    label=strategy,
                )

            ax.set_title(network)
            ax.set_xticks(x)
            ax.set_xticklabels(present_levels)
            ax.set_xlabel("Heterogeneity")
            ax.set_ylim(bottom=0)

            if metric == "ended_with_active_believers":
                ax.set_ylim(0, 1.05)

        axes[0].set_ylabel(label)
        axes[-1].legend(title="Intervention")

        error_label = (
            "95% Wilson confidence intervals"
            if metric == "ended_with_active_believers"
            else "mean ± standard deviation"
        )

        fig.suptitle(f"{label}: {error_label}")
        fig.tight_layout()
        fig.savefig(
            output / f"{metric}.png",
            dpi=200,
        )
        plt.close(fig)


def main() -> None:
    """Read command-line arguments and generate figures from the results CSV."""
    parser = argparse.ArgumentParser(description="Create project result figures")
    parser.add_argument(
        "--input",
        default="../results/combined_experiment.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="../results/combined_figures",
    )

    args = parser.parse_args()
    plot_main_results(args.input, args.output_dir)
    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()