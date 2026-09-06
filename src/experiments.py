from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from model import ModelConfig, run_simulation


def run_factorial_experiment(
    out_csv: str,
    n: int = 500,
    repeats: int = 30,
    mean_degree: int = 8,
    max_steps: int = 150,
) -> pd.DataFrame:
    """
    Main experiment:
      3 network structures x 3 interventions x repeated random seeds.
    Keeps model parameters fixed so the primary effect comes from
    network structure and fact-checking strategy.
    """
    networks = ["random", "small_world", "scale_free"]
    interventions = ["none", "random", "centrality"]

    rows = []
    for network_type, intervention, rep in itertools.product(
        networks, interventions, range(repeats)
    ):
        cfg = ModelConfig(
            n=n,
            mean_degree=mean_degree,
            network_type=network_type,
            intervention=intervention,
            fact_checker_fraction=0.05 if intervention != "none" else 0.0,
            misinformation_prob=0.15,
            correction_prob=0.35,
            resistant_fraction=0.05,
            initial_believers=5,
            max_steps=max_steps,
            seed=1000 + rep,
        )
        _, metrics, _ = run_simulation(cfg)
        metrics["repeat"] = rep
        rows.append(metrics)

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def run_budget_sweep(
    out_csv: str,
    n: int = 500,
    repeats: int = 30,
    network_type: str = "scale_free",
) -> pd.DataFrame:
    """
    Independent-investigation extension:
    ask how much fact-checking budget is needed before targeted intervention
    substantially outperforms random placement.
    """
    fractions = [0.01, 0.025, 0.05, 0.075, 0.10]
    strategies = ["random", "centrality"]

    rows = []
    for fraction, strategy, rep in itertools.product(
        fractions, strategies, range(repeats)
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
            seed=3000 + rep,
        )
        _, metrics, _ = run_simulation(cfg)
        metrics["repeat"] = rep
        rows.append(metrics)

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["main", "budget"], default="main")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="../results/main_experiment.csv")
    args = parser.parse_args()

    if args.experiment == "main":
        df = run_factorial_experiment(args.output, n=args.n, repeats=args.repeats)
    else:
        df = run_budget_sweep(args.output, n=args.n, repeats=args.repeats)

    print(df.groupby(["network_type", "intervention"])["final_reach"]
          .agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()
