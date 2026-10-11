from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


NETWORKS = ["random", "small_world", "scale_free"]
LEVELS = ["homogeneous", "low", "high"]
STRATEGIES = ["none", "random", "centrality"]

COLORS = {
    "none": "#94a3b8",
    "random": "#f59e0b",
    "centrality": "#2563eb",
}

KEYS = [
    "experiment",
    "scenario",
    "network_type",
    "heterogeneity",
    "n",
    "mean_degree",
    "misinformation_prob",
    "correction_prob",
    "fact_checker_fraction",
    "intervention",
]

METRICS = [
    "final_reach",
    "peak_believer_fraction",
    "time_to_peak",
    "ended_with_active_believers",
]


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n

    centre = (p + z * z / (2 * n)) / denominator

    half = (
        z * np.sqrt(
            p * (1 - p) / n
            + z * z / (4 * n * n)
        ) / denominator
    )

    return (
        0.0 if k == 0 else max(0.0, centre - half),
        1.0 if k == n else min(1.0, centre + half),
    )


def interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)

    mean = values.mean()
    std = values.std(ddof=1)
    half = t.ppf(0.975, n - 1) * std / np.sqrt(n)

    return mean, std, mean - half, mean + half


def make_summary(df):
    rows = []

    for values, group in df.groupby(KEYS, dropna=False):
        for metric in METRICS:
            mean, std, low, high = interval(group[metric])

            if metric == "ended_with_active_believers":
                low, high = wilson(
                    group[metric].sum(),
                    len(group),
                )

            row = dict(zip(KEYS, values))
            row.update(
                metric=metric,
                mean=mean,
                std=std,
                count=len(group),
                ci95_low=low,
                ci95_high=high,
                interval_method=(
                    "Wilson"
                    if metric == "ended_with_active_believers"
                    else "Student t mean"
                ),
            )
            rows.append(row)

    return pd.DataFrame(rows)


def make_contrasts(df):
    keys = [key for key in KEYS if key != "intervention"]

    selected = df[
        df.intervention.isin(["random", "centrality"])
    ]

    rows = []

    for values, group in selected.groupby(keys, dropna=False):
        for metric in METRICS:
            pivot = group.pivot(
                index="repeat",
                columns="intervention",
                values=metric,
            ).dropna()

            if not {"random", "centrality"}.issubset(pivot.columns):
                continue

            # 同一网络重复编号下，计算度数策略减随机策略。
            differences = pivot.centrality - pivot.random
            mean, std, low, high = interval(differences)

            if high < 0:
                interpretation = "centrality_lower"
            elif low > 0:
                interpretation = "centrality_higher"
            else:
                interpretation = "uncertain"

            row = dict(zip(keys, values))
            row.update(
                metric=metric,
                count=len(differences),
                mean_difference=mean,
                std_difference=std,
                ci95_low=low,
                ci95_high=high,
                contrast="centrality minus random",
                interval_method="Student t on network-block differences",
                interpretation=interpretation,
            )
            rows.append(row)

    return pd.DataFrame(rows)


def main_figures(summary, out):
    main = summary[summary.experiment == "main"]

    labels = {
        "final_reach": "Final reach",
        "peak_believer_fraction": "Peak believer fraction",
        "time_to_peak": "Time to peak",
        "ended_with_active_believers": (
            "Active believers at time limit (run proportion)"
        ),
    }

    for metric, label in labels.items():
        fig, axes = plt.subplots(
            1, 3,
            figsize=(14, 4.6),
            sharey=True,
        )

        for ax, network in zip(axes, NETWORKS):
            x = np.arange(len(LEVELS))
            width = 0.24

            for i, strategy in enumerate(STRATEGIES):
                sub = main[
                    (main.network_type == network)
                    & (main.intervention == strategy)
                    & (main.metric == metric)
                ].set_index("heterogeneity").reindex(LEVELS)

                means = sub["mean"].to_numpy()

                errors = np.vstack([
                    means - sub.ci95_low.to_numpy(),
                    sub.ci95_high.to_numpy() - means,
                ])

                ax.bar(
                    x + (i - 1) * width,
                    means,
                    width,
                    yerr=np.maximum(errors, 0),
                    capsize=3,
                    label=strategy,
                    color=COLORS[strategy],
                )

            ax.set_xticks(x, LEVELS)
            ax.set_title(network.replace("_", " "))
            ax.set_xlabel("Heterogeneity")
            ax.set_ylim(bottom=0)

            if metric == "ended_with_active_believers":
                ax.set_ylim(0, 1.05)

        axes[0].set_ylabel(label)
        axes[-1].legend(title="Strategy")

        fig.suptitle(
            f"{label}: mean / proportion with 95% confidence intervals"
        )
        fig.tight_layout()
        fig.savefig(out / f"main_{metric}.png", dpi=180)
        plt.close(fig)


def contrast_figure(contrasts, out):
    data = contrasts[
        (contrasts.experiment == "main")
        & (contrasts.metric == "final_reach")
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, network in enumerate(NETWORKS):
        sub = data[
            data.network_type == network
        ].set_index("heterogeneity").reindex(LEVELS)

        means = sub.mean_difference.to_numpy()

        errors = np.vstack([
            means - sub.ci95_low.to_numpy(),
            sub.ci95_high.to_numpy() - means,
        ])

        ax.errorbar(
            np.arange(len(LEVELS)) + (i - 1) * 0.06,
            means,
            yerr=np.maximum(errors, 0),
            marker="o",
            capsize=4,
            label=network,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(LEVELS)), LEVELS)
    ax.set_ylabel("Final reach: degree-based minus random")
    ax.set_xlabel("Heterogeneity")
    ax.set_title("Shared-network strategy contrasts (95% intervals)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "strategy_difference.png", dpi=180)
    plt.close(fig)


def extension_figures(df, out):
    for experiment in ["budget", "sensitivity", "structure"]:
        ext = df[
            (df.experiment == experiment)
            | (df.experiment == "main")
        ].copy()

        if not (ext.experiment == experiment).any():
            continue

        ext = ext[
            ext.intervention.isin(["random", "centrality"])
        ]

        if experiment == "budget":
            # 5%预算使用主实验结果。
            ext.loc[
                ext.experiment == "main",
                "scenario",
            ] = "budget_0.05"

            order = [
                "budget_0.01",
                "budget_0.025",
                "budget_0.05",
                "budget_0.075",
                "budget_0.1",
            ]
            ticklabels = ["1%", "2.5%", "5%", "7.5%", "10%"]

        elif experiment == "sensitivity":
            order = [
                "adoption_0.1",
                "baseline",
                "adoption_0.2",
                "correction_0.2",
                "correction_0.5",
            ]
            ticklabels = [
                "p=.10",
                "baseline",
                "p=.20",
                "q=.20",
                "q=.50",
            ]

        else:
            baseline_n = int(
                ext.loc[ext.experiment == "main", "n"].iloc[0]
            )

            scales = ext.loc[
                ext.experiment == "structure",
                ["scenario", "n"],
            ].drop_duplicates()

            scale_order = {
                row.scenario: int(row.n)
                for row in scales.itertuples()
                if row.scenario.startswith("n_")
            }

            scale_order["baseline"] = baseline_n

            order = sorted(
                scale_order,
                key=scale_order.get,
            ) + ["degree_4", "degree_12"]

            ticklabels = [
                f"N={scale_order[key]}"
                for key in order[:-2]
            ] + ["degree=4", "degree=12"]

        fig, axes = plt.subplots(
            3, 3,
            figsize=(15, 11),
            sharey=True,
        )

        for row, network in enumerate(NETWORKS):
            for column, level in enumerate(LEVELS):
                ax = axes[row, column]

                for strategy in ["random", "centrality"]:
                    sub = ext[
                        (ext.network_type == network)
                        & (ext.heterogeneity == level)
                        & (ext.intervention == strategy)
                    ]

                    means = []
                    lows = []
                    highs = []

                    for scenario in order:
                        mean, std, low, high = interval(
                            sub[
                                sub.scenario == scenario
                            ].final_reach
                        )
                        means.append(mean)
                        lows.append(low)
                        highs.append(high)

                    means = np.asarray(means)

                    errors = np.vstack([
                        means - np.asarray(lows),
                        np.asarray(highs) - means,
                    ])

                    ax.errorbar(
                        range(len(order)),
                        means,
                        yerr=np.maximum(errors, 0),
                        marker="o",
                        capsize=3,
                        label=strategy,
                        color=COLORS[strategy],
                    )

                ax.set_xticks(
                    range(len(order)),
                    ticklabels,
                    rotation=20,
                )
                ax.set_title(f"{network} / {level}")
                ax.set_ylim(0, 1.05)

                if column == 0:
                    ax.set_ylabel("Mean final reach")

        axes[0, 2].legend()
        fig.suptitle(
            f"{experiment}: final reach with 95% mean intervals"
        )
        fig.tight_layout()
        fig.savefig(
            out / f"{experiment}_final_reach.png",
            dpi=160,
        )
        plt.close(fig)


def generate_plots(input_file, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    active = (
        df.ended_with_active_believers
        .astype(str)
        .str.lower()
    )

    if not active.isin(["true", "false", "1", "0"]).all():
        raise ValueError(
            "Invalid ended_with_active_believers values"
        )

    df["ended_with_active_believers"] = (
        active.isin(["true", "1"]).astype(int)
    )

    summary = make_summary(df)
    contrasts = make_contrasts(df)

    summary.to_csv(out / "summary.csv", index=False)
    contrasts.to_csv(
        out / "strategy_contrasts.csv",
        index=False,
    )

    main_figures(summary, out)
    contrast_figure(contrasts, out)
    extension_figures(df, out)

    print(
        f"Figures and summaries saved to: {out.resolve()}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    generate_plots(
        args.input,
        args.output_dir,
    )


if __name__ == "__main__":
    main()