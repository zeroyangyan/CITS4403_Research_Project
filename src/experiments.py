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
    model_mode: str = "homogeneous",
    susceptibility_concentration: float = 20.0,
    correction_concentration: float = 20.0,
) -> pd.DataFrame:
    """Run the 3-network x 3-intervention factorial experiment."""
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
            model_mode=model_mode,
            susceptibility_concentration=susceptibility_concentration,
            correction_concentration=correction_concentration,
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
    model_mode: str = "homogeneous",
    susceptibility_concentration: float = 20.0,
    correction_concentration: float = 20.0,
) -> pd.DataFrame:
    """Compare random and degree-targeted fact-checking across intervention budgets."""
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
            model_mode=model_mode,
            susceptibility_concentration=susceptibility_concentration,
            correction_concentration=correction_concentration,
            seed=3000 + rep,
        )
        _, metrics, _ = run_simulation(cfg)
        metrics["repeat"] = rep
        rows.append(metrics)

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def run_heterogeneity_smoke(
    out_csv: str,
    n: int = 100,
    repeats: int = 2,
    susceptibility_concentration: float = 20.0,
    correction_concentration: float = 20.0,
) -> pd.DataFrame:
    """Small fast check covering both homogeneous and heterogeneous modes."""
    rows = []
    modes = ["homogeneous", "heterogeneous"]
    strategies = ["random", "centrality"]

    for mode, strategy, rep in itertools.product(modes, strategies, range(repeats)):
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
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["main", "budget", "smoke"], default="main")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="../results/main_experiment.csv")
    parser.add_argument(
        "--mode",
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
        help="Behavioural probability mode for main/budget experiments.",
    )
    parser.add_argument("--susceptibility-concentration", type=float, default=20.0)
    parser.add_argument("--correction-concentration", type=float, default=20.0)
    args = parser.parse_args()

    if args.experiment == "main":
        df = run_factorial_experiment(
            args.output,
            n=args.n,
            repeats=args.repeats,
            model_mode=args.mode,
            susceptibility_concentration=args.susceptibility_concentration,
            correction_concentration=args.correction_concentration,
        )
    elif args.experiment == "budget":
        df = run_budget_sweep(
            args.output,
            n=args.n,
            repeats=args.repeats,
            model_mode=args.mode,
            susceptibility_concentration=args.susceptibility_concentration,
            correction_concentration=args.correction_concentration,
        )
    else:
        df = run_heterogeneity_smoke(
            args.output,
            n=args.n,
            repeats=args.repeats,
            susceptibility_concentration=args.susceptibility_concentration,
            correction_concentration=args.correction_concentration,
        )

    print(
        df.groupby(["model_mode", "network_type", "intervention"])["final_reach"]
        .agg(["mean", "std", "count"])
    )


if __name__ == "__main__":
    main()
