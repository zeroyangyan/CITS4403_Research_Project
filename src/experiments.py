from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from model import ModelConfig, run_simulation


NETWORKS = ["random", "small_world", "scale_free"]
STRATEGIES = ["none", "random", "centrality"]
LEVELS = ["homogeneous", "low", "high"]


def check_output(out_csv):
    path = Path(out_csv)
    if (
        path.exists()
        or path.with_suffix(".metadata.json").exists()
    ):
        raise FileExistsError(
            "Output exists. Choose a new filename."
        )


def save_results(out_csv, rows):
    df = pd.DataFrame(rows)

    # 自动检查同一重复实验是否共享网络。
    graph_counts = df.groupby(
        ["network_type", "repeat"]
    )["graph_sha256"].nunique()

    if not graph_counts.eq(1).all():
        raise AssertionError(
            "Compared conditions do not share the same graph"
        )

    # 同一异质性条件中的策略必须共享用户属性。
    attribute_counts = df.groupby(
        ["network_type", "repeat", "heterogeneity"]
    )["attribute_sha256"].nunique()

    if not attribute_counts.eq(1).all():
        raise AssertionError(
            "Compared strategies do not share agent attributes"
        )

    # 有干预的两种策略必须使用相同实际预算。
    interventions = df[df["intervention"] != "none"]
    budget_counts = interventions.groupby(
        [
            "network_type",
            "repeat",
            "heterogeneity",
            "fact_checker_fraction",
        ]
    )["num_fact_checkers"].nunique()

    if not budget_counts.eq(1).all():
        raise AssertionError(
            "Fact-checker counts differ between strategies"
        )

    output = Path(out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_dir = Path(__file__).resolve().parent

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=source_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip())

    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None

    metadata = {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "source_commit": commit,
        "git_dirty": dirty,
        "source_sha256": {
            path.name: hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(source_dir.glob("*.py"))
        },
        "initialisation_protocol":
            "independent_eligible_sampling_v1",
        "runs": len(df),
        "heterogeneity": {
            "distribution": "paired_symmetric_uniform",
            "strengths": {
                "homogeneous": 0.0,
                "low": 0.25,
                "high": 0.90,
            },
            "note": (
                "Attributes are sampled once and remain fixed. "
                "Whole-population means match configured means. "
                "These distributions are modelling assumptions."
            ),
        },
        "comparison_note": (
            "Strategies share networks and latent attributes. "
            "Initial states are independently sampled from "
            "eligible nodes. Runs across strategies are grouped "
            "by network replicate, not independent graph samples."
        ),
    }

    df.to_csv(output, index=False, mode="x")

    with output.with_suffix(".metadata.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    return df


def run_factorial_experiment(
    out_csv,
    n=500,
    repeats=30,
    mean_degree=8,
    max_steps=150,
    heterogeneity_levels=None,
):
    if repeats < 1:
        raise ValueError("repeats must be positive")

    levels = (
        LEVELS if heterogeneity_levels is None
        else list(heterogeneity_levels)
    )

    if not levels or not set(levels).issubset(LEVELS):
        raise ValueError("Invalid heterogeneity levels")

    check_output(out_csv)
    rows = []
    total = len(NETWORKS) * len(levels) * len(STRATEGIES) * repeats

    for network_type in NETWORKS:
        for rep in range(repeats):
            for level in levels:
                for intervention in STRATEGIES:
                    cfg = ModelConfig(
                        n=n,
                        mean_degree=mean_degree,
                        network_type=network_type,
                        intervention=intervention,
                        heterogeneity=level,
                        fact_checker_fraction=(
                            0.0 if intervention == "none"
                            else 0.05
                        ),
                        max_steps=max_steps,
                        # 不随策略或异质性水平改变。
                        seed=1000 + rep,
                    )

                    _, metrics, _ = run_simulation(cfg)
                    metrics["repeat"] = rep
                    rows.append(metrics)

            print(
                f"Completed {len(rows)}/{total}",
                flush=True,
            )

    return save_results(out_csv, rows)


def run_budget_sweep(
    out_csv,
    n=500,
    repeats=30,
    network_type="scale_free",
    heterogeneity="homogeneous",
):
    if repeats < 1:
        raise ValueError("repeats must be positive")

    check_output(out_csv)
    rows = []

    for rep in range(repeats):
        for fraction in [0.01, 0.025, 0.05, 0.075, 0.10]:
            for strategy in ["random", "centrality"]:
                cfg = ModelConfig(
                    n=n,
                    network_type=network_type,
                    intervention=strategy,
                    heterogeneity=heterogeneity,
                    fact_checker_fraction=fraction,
                    seed=3000 + rep,
                )

                _, metrics, _ = run_simulation(cfg)
                metrics["repeat"] = rep
                rows.append(metrics)

    return save_results(out_csv, rows)


def run_budget_sweep(
    out_csv: str,
    n: int = 500,
    repeats: int = 30,
    network_type: str = "scale_free",
    model_mode: str = "homogeneous",
    susceptibility_concentration: float = 20.0,
    correction_concentration: float = 20.0,
) -> pd.DataFrame:
    """Compare random and degree-targeted fact-checking across budgets."""
    fractions = [0.01, 0.025, 0.05, 0.075, 0.10]
    strategies = ["random", "centrality"]

    rows = []
    for fraction, strategy, rep in itertools.product(
        fractions,
        strategies,
        range(repeats),
    ):
        cfg = ModelConfig(
            n=n,
            mean_degree=8,
            network_type=network_type,
            intervention=strategy,
            fact_checker_fraction=fraction,
            misinformation_prob=0.15,
            correction_prob=0.35,
            resistant_fraction=0.05,
            initial_believers=5,
            max_steps=150,
            model_mode=model_mode,
            susceptibility_concentration=susceptibility_concentration,
            correction_concentration=correction_concentration,
            seed=3000 + rep,
        )
        _, metrics, _ = run_simulation(cfg)
        metrics["repeat"] = rep
        rows.append(metrics)

    df = pd.DataFrame(rows)
    output = Path(out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def run_heterogeneity_smoke(
    out_csv: str,
    n: int = 100,
    repeats: int = 2,
    susceptibility_concentration: float = 20.0,
    correction_concentration: float = 20.0,
) -> pd.DataFrame:
    """Fast check covering homogeneous and explicit Beta-heterogeneous modes."""
    rows = []
    modes = ["homogeneous", "heterogeneous"]
    strategies = ["random", "centrality"]

    for mode, strategy, rep in itertools.product(
        modes,
        strategies,
        range(repeats),
    ):
        cfg = ModelConfig(
            n=n,
            mean_degree=8,
            network_type="scale_free",
            intervention=strategy,
            fact_checker_fraction=0.05,
            misinformation_prob=0.15,
            correction_prob=0.35,
            resistant_fraction=0.05,
            initial_believers=5,
            max_steps=60,
            model_mode=mode,
            susceptibility_concentration=susceptibility_concentration,
            correction_concentration=correction_concentration,
            seed=9000 + rep,
        )
        _, metrics, _ = run_simulation(cfg)
        metrics["repeat"] = rep
        rows.append(metrics)

    df = pd.DataFrame(rows)
    output = Path(out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["main", "budget"],
        default="main",
    )
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=150)

    parser.add_argument(
        "--heterogeneity",
        choices=["all", *LEVELS],
        default="all",
    )

    parser.add_argument(
        "--output",
        default="../results/combined_experiment.csv",
    )

    args = parser.parse_args()

    if args.experiment == "main":
        levels = (
            LEVELS if args.heterogeneity == "all"
            else [args.heterogeneity]
        )

        df = run_factorial_experiment(
            args.output,
            n=args.n,
            repeats=args.repeats,
            max_steps=args.max_steps,
            heterogeneity_levels=levels,
        )

    else:
        if args.heterogeneity == "all":
            parser.error(
                "For budget experiments, choose one "
                "--heterogeneity level."
            )

        df = run_budget_sweep(
            args.output,
            n=args.n,
            repeats=args.repeats,
            heterogeneity=args.heterogeneity,
        )

    print(
        df.groupby([
            "network_type",
            "heterogeneity",
            "intervention",
        ])["final_reach"].agg(
            ["mean", "std", "count"]
        ).round(4)
    )


if __name__ == "__main__":
    main()