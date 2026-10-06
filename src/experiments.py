from __future__ import annotations

import argparse
import hashlib
import itertools
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
INTERVENTIONS = ["none", "random", "centrality"]
LEVELS = ["homogeneous", "low", "high"]


def environment_record():
    source_dir = Path(__file__).resolve().parent

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=source_dir,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        git_dirty = None

    hashes = {}
    for name in (
        "model.py",
        "experiments.py",
        "plots.py",
        "checkpoint2_demo.py",
    ):
        path = source_dir / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "networkx": nx.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_commit": commit,
        "git_dirty": git_dirty,
        "source_sha256": hashes,
        "comparison_design": "independent runs; no paired inference",
        "heterogeneity_strengths": {
            "homogeneous": 0.0,
            "low": 0.25,
            "high": 0.90,
        },
        "explicit_beta_mode": {
            "distribution": "beta",
            "default_susceptibility_concentration": 20.0,
            "default_correction_concentration": 20.0,
        },
    }


def run_factorial_experiment(
    out_csv: str,
    n: int = 500,
    repeats: int = 30,
    mean_degree: int = 8,
    max_steps: int = 150,
) -> pd.DataFrame:
    """Run the latest-main 3-network x 3-level x 3-intervention experiment."""
    if repeats < 1:
        raise ValueError("repeats must be positive")

    output = Path(out_csv)
    metadata_path = output.with_suffix(".metadata.json")

    if output.exists() or metadata_path.exists():
        raise FileExistsError(
            "Output already exists. Choose a new output filename."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(NETWORKS) * len(INTERVENTIONS) * len(LEVELS) * repeats

    combinations = itertools.product(
        NETWORKS,
        LEVELS,
        INTERVENTIONS,
        range(repeats),
    )

    for run_id, (network, level, strategy, rep) in enumerate(combinations):
        cfg = ModelConfig(
            n=n,
            mean_degree=mean_degree,
            network_type=network,
            intervention=strategy,
            heterogeneity=level,
            fact_checker_fraction=(
                0.0 if strategy == "none" else 0.05
            ),
            max_steps=max_steps,
            seed=1000 + run_id,
        )

        _, metrics, _ = run_simulation(cfg)
        metrics["repeat"] = rep
        metrics["run_id"] = run_id
        rows.append(metrics)

        if (run_id + 1) % 30 == 0 or run_id + 1 == total:
            print(f"Completed {run_id + 1}/{total}", flush=True)

    df = pd.DataFrame(rows)

    # mode="x" prevents accidental overwrite of an existing experiment.
    df.to_csv(output, index=False, mode="x")

    metadata = environment_record()
    metadata["experiment"] = {
        "n": n,
        "repeats": repeats,
        "mean_degree": mean_degree,
        "max_steps": max_steps,
        "networks": NETWORKS,
        "interventions": INTERVENTIONS,
        "heterogeneity_levels": LEVELS,
        "total_runs": total,
        "seed_start": 1000,
        "seed_end": 1000 + total - 1,
    }

    with metadata_path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

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
        choices=["main", "budget", "smoke"],
        default="main",
    )
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument(
        "--output",
        default="results/heterogeneity_experiment.csv",
    )
    parser.add_argument(
        "--mode",
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
        help="Explicit probability mode for budget/smoke experiments.",
    )
    parser.add_argument(
        "--susceptibility-concentration",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--correction-concentration",
        type=float,
        default=20.0,
    )
    args = parser.parse_args()

    if args.experiment == "main":
        df = run_factorial_experiment(
            args.output,
            n=args.n,
            repeats=args.repeats,
            max_steps=args.max_steps,
        )
        grouped = df.groupby(
            ["network_type", "heterogeneity", "intervention"]
        )["final_reach"].agg(["mean", "std", "count"]).round(4)

    elif args.experiment == "budget":
        df = run_budget_sweep(
            args.output,
            n=args.n,
            repeats=args.repeats,
            model_mode=args.mode,
            susceptibility_concentration=args.susceptibility_concentration,
            correction_concentration=args.correction_concentration,
        )
        grouped = df.groupby(
            ["model_mode", "network_type", "intervention"]
        )["final_reach"].agg(["mean", "std", "count"]).round(4)

    else:
        df = run_heterogeneity_smoke(
            args.output,
            n=args.n,
            repeats=args.repeats,
            susceptibility_concentration=args.susceptibility_concentration,
            correction_concentration=args.correction_concentration,
        )
        grouped = df.groupby(
            ["model_mode", "network_type", "intervention"]
        )["final_reach"].agg(["mean", "std", "count"]).round(4)

    print(grouped)


if __name__ == "__main__":
    main()
