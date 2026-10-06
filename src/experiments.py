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

        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=source_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = None
        git_dirty = None

    hashes = {}
    for name in (
        "model.py", "experiments.py",
        "plots.py", "checkpoint2_demo.py"
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
    }


def run_factorial_experiment(
    out_csv: str,
    n: int = 500,
    repeats: int = 30,
    mean_degree: int = 8,
    max_steps: int = 150,
) -> pd.DataFrame:
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
        NETWORKS, LEVELS, INTERVENTIONS, range(repeats)
    )

    for run_id, (network, level, strategy, rep) in enumerate(combinations):
        cfg = ModelConfig(
            n=n,
            mean_degree=mean_degree,
            network_type=network,
            intervention=strategy,
            heterogeneity=level,
            fact_checker_fraction=0.0 if strategy == "none" else 0.05,
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

    # mode="x" 防止覆盖已有文件。
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument(
        "--output",
        default="results/heterogeneity_experiment.csv"
    )
    args = parser.parse_args()

    df = run_factorial_experiment(
        args.output,
        n=args.n,
        repeats=args.repeats,
        max_steps=args.max_steps,
    )

    print(
        df.groupby(
            ["network_type", "heterogeneity", "intervention"]
        )["final_reach"].agg(["mean", "std", "count"]).round(4)
    )


if __name__ == "__main__":
    main()
