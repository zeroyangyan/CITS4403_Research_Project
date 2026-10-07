from __future__ import annotations

import argparse
import itertools
import json
import hashlib
import platform
import sys
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from model import ModelConfig, run_simulation


def check_output(out_csv: str) -> None:
    path = Path(out_csv)
    metadata_path = path.with_suffix(".metadata.json")

    if path.exists() or metadata_path.exists():
        raise FileExistsError(
            "Choose a new output filename; "
            "existing results are protected."
        )


def save_metadata(
    out_csv: str,
    df: pd.DataFrame,
) -> None:
    source = Path(__file__).resolve().parent

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=source,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )

    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None

    source_hashes = {
        path.name: hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(source.glob("*.py"))
    }

    data = {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "pandas": pd.__version__,
        },
        "source_commit": commit,
        "git_dirty": dirty,
        "source_sha256": source_hashes,
        "initialisation_protocol":
            "independent_eligible_sampling_v1",
        "runs": len(df),
        "note": (
            "Strategies share the graph within each network "
            "replicate. Initial resistant agents and believers "
            "are independently sampled from eligible nodes. "
            "Initial states are not exactly matched."
        ),
    }

    metadata_path = Path(out_csv).with_suffix(
        ".metadata.json"
    )

    with metadata_path.open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=2)


def run_factorial_experiment(
    out_csv: str,
    n: int = 500,
    repeats: int = 30,
    mean_degree: int = 8,
    max_steps: int = 150,
) -> pd.DataFrame:
    """
    3种网络 × 3种策略 × repeats次重复。

    同一网络类型和重复编号下，三种策略使用相同网络。
    初始状态独立抽样，不声称完全匹配。
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")

    check_output(out_csv)

    networks = [
        "random",
        "small_world",
        "scale_free",
    ]
    interventions = [
        "none",
        "random",
        "centrality",
    ]

    rows = []

    for network_type, rep, intervention in itertools.product(
        networks,
        range(repeats),
        interventions,
    ):
        cfg = ModelConfig(
            n=n,
            mean_degree=mean_degree,
            network_type=network_type,
            intervention=intervention,
            fact_checker_fraction=(
                0.0 if intervention == "none" else 0.05
            ),
            misinformation_prob=0.15,
            correction_prob=0.35,
            resistant_fraction=0.05,
            initial_believers=5,
            max_steps=max_steps,
            # 同一重复编号共享基础种子。
            # model.py 会按用途和策略派生独立随机流。
            seed=1000 + rep,
        )

        _, metrics, _ = run_simulation(cfg)
        metrics["repeat"] = rep
        rows.append(metrics)

    df = pd.DataFrame(rows)

    Path(out_csv).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_csv,
        index=False,
        mode="x",
    )

    save_metadata(out_csv, df)
    return df


def run_budget_sweep(
    out_csv: str,
    n: int = 500,
    repeats: int = 30,
    network_type: str = "scale_free",
) -> pd.DataFrame:
    """比较不同核查者预算下的随机与度数策略。"""
    if repeats < 1:
        raise ValueError("repeats must be positive")

    check_output(out_csv)

    fractions = [
        0.01,
        0.025,
        0.05,
        0.075,
        0.10,
    ]
    strategies = [
        "random",
        "centrality",
    ]

    rows = []

    for fraction, rep, strategy in itertools.product(
        fractions,
        range(repeats),
        strategies,
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

    Path(out_csv).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_csv,
        index=False,
        mode="x",
    )

    save_metadata(out_csv, df)
    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["main", "budget"],
        default="main",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--output",
        default="../results/issue2_main_experiment.csv",
    )

    args = parser.parse_args()

    if args.experiment == "main":
        df = run_factorial_experiment(
            args.output,
            n=args.n,
            repeats=args.repeats,
        )
    else:
        df = run_budget_sweep(
            args.output,
            n=args.n,
            repeats=args.repeats,
        )

    print(
        df.groupby(
            ["network_type", "intervention"]
        )["final_reach"].agg(
            ["mean", "std", "count"]
        )
    )


if __name__ == "__main__":
    main()