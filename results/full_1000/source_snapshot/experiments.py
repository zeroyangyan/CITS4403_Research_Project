from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy

from model import ModelConfig, run_simulation


NETWORKS = ["random", "small_world", "scale_free"]
LEVELS = ["homogeneous", "low", "high"]
STRATEGIES = ["none", "random", "centrality"]


def settings(suite):
    scenarios = [
        ("main", "baseline", {}, STRATEGIES)
    ]

    if suite == "full":
        # 5%预算复用主实验。
        for fraction in [0.01, 0.025, 0.075, 0.10]:
            scenarios.append((
                "budget",
                f"budget_{fraction}",
                {"fact_checker_fraction": fraction},
                STRATEGIES[1:],
            ))

        for probability in [0.10, 0.20]:
            scenarios.append((
                "sensitivity",
                f"adoption_{probability}",
                {"misinformation_prob": probability},
                STRATEGIES,
            ))

        for probability in [0.20, 0.50]:
            scenarios.append((
                "sensitivity",
                f"correction_{probability}",
                {"correction_prob": probability},
                STRATEGIES,
            ))

        # N=1000复用主实验。
        for n in [500, 5000]:
            scenarios.append((
                "structure",
                f"n_{n}",
                {"n": n},
                STRATEGIES,
            ))

        for degree in [4, 12]:
            scenarios.append((
                "structure",
                f"degree_{degree}",
                {"mean_degree": degree},
                STRATEGIES,
            ))

    return scenarios


def make_tasks(suite, repeats, base_seed, max_steps):
    tasks = []

    for experiment, scenario, changes, strategies in settings(suite):
        for network in NETWORKS:
            for level in LEVELS:
                for repeat in range(repeats):
                    configs = []

                    for strategy in strategies:
                        parameters = dict(
                            n=1000,
                            initial_believer_fraction=0.01,
                            mean_degree=8,
                            network_type=network,
                            heterogeneity=level,
                            intervention=strategy,
                            max_steps=max_steps,
                            seed=base_seed + repeat,
                            fact_checker_fraction=(
                                0.0 if strategy == "none" else 0.05
                            ),
                        )

                        parameters.update(changes)

                        parameters["initial_believers"] = max(
                            1,
                            round(
                                parameters["n"]
                                * parameters["initial_believer_fraction"]
                            ),
                        )

                        configs.append(parameters)

                    tasks.append((
                        experiment,
                        scenario,
                        repeat,
                        configs,
                    ))

    return tasks


def execute_task(task):
    experiment, scenario, repeat, configs = task
    rows = []
    histories = []

    for parameters in configs:
        cfg = ModelConfig(**parameters)
        history, metrics, _ = run_simulation(cfg)

        metrics.update(
            experiment=experiment,
            scenario=scenario,
            repeat=repeat,
        )
        rows.append(metrics)

        if experiment == "main" and repeat == 0:
            for item in history:
                histories.append(dict(
                    item,
                    network_type=cfg.network_type,
                    heterogeneity=cfg.heterogeneity,
                    intervention=cfg.intervention,
                    actual_n=metrics["actual_n"],
                    repeat=repeat,
                ))

    # 同一轮不同策略共享网络和潜在用户属性。
    assert len({row["graph_sha256"] for row in rows}) == 1
    assert len({row["attribute_sha256"] for row in rows}) == 1

    intervention_rows = [
        row for row in rows
        if row["intervention"] != "none"
    ]

    assert len({
        row["num_fact_checkers"]
        for row in intervention_rows
    }) <= 1

    return rows, histories


def environment_record():
    source = Path(__file__).resolve().parent

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=source,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip())

    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None

    return dict(
        command=sys.argv,
        python=sys.version,
        platform=platform.platform(),
        package_versions=dict(
            networkx=nx.__version__,
            numpy=np.__version__,
            pandas=pd.__version__,
            scipy=scipy.__version__,
        ),
        source_commit=commit,
        git_dirty=dirty,
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.glob("*.py"))
        },
        protocol="shared_graph_independent_eligible_initial_states_v1",
        disclaimer=(
            "Scenario choices and attribute distributions are synthetic "
            "modelling assumptions, not empirical calibration."
        ),
    )


def run_suite(
    output,
    suite="full",
    repeats=100,
    workers=8,
    base_seed=1000,
    max_steps=300,
):
    if repeats < 2 or workers < 1:
        raise ValueError("repeats >= 2 and workers >= 1 are required")

    out = Path(output)

    # 不覆盖已经存在的结果目录。
    out.mkdir(parents=True, exist_ok=False)

    snapshot = out / "source_snapshot"
    snapshot.mkdir()

    for path in Path(__file__).resolve().parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)

    tasks = make_tasks(
        suite,
        repeats,
        base_seed,
        max_steps,
    )

    total = sum(len(task[3]) for task in tasks)

    meta = environment_record()
    meta.update(
        status="running",
        suite=suite,
        repeats=repeats,
        workers=workers,
        base_seed=base_seed,
        max_steps=max_steps,
        planned_runs=total,
        started_utc=datetime.now(timezone.utc).isoformat(),
        baseline_n=1000,
        initial_believer_fraction=0.01,
        scale_comparison_nodes=[500, 1000, 5000],
        scenarios=[
            dict(
                experiment=experiment,
                scenario=scenario,
                changes=changes,
                strategies=strategies,
            )
            for experiment, scenario, changes, strategies in settings(suite)
        ],
    )

    metadata_path = out / "metadata.json"
    metadata_path.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    start = time.perf_counter()
    done = 0
    histories = []
    csv_path = out / "raw_results.csv"

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows, history in pool.map(
            execute_task,
            tasks,
            chunksize=3,
        ):
            pd.DataFrame(rows).to_csv(
                csv_path,
                mode="a",
                header=done == 0,
                index=False,
            )

            histories.extend(history)
            done += len(rows)

            if done % 300 == 0 or done == total:
                elapsed = time.perf_counter() - start
                print(
                    f"{done}/{total} runs, {elapsed:.1f}s",
                    flush=True,
                )

    pd.DataFrame(histories).to_csv(
        out / "representative_histories.csv",
        index=False,
    )

    saved = pd.read_csv(csv_path)

    if len(saved) != total:
        raise AssertionError(
            f"Expected {total} saved runs, found {len(saved)}"
        )

    sizes = saved.groupby([
        "experiment",
        "scenario",
        "network_type",
        "heterogeneity",
        "intervention",
    ]).size()

    if not sizes.eq(repeats).all():
        raise AssertionError("Incomplete experimental conditions")

    meta.update(
        status="completed",
        completed_runs=done,
        elapsed_seconds=time.perf_counter() - start,
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )

    metadata_path.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return csv_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suite",
        choices=["main", "full"],
        default="full",
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    output = args.output or str(
        Path(__file__).resolve().parent.parent
        / "results"
        / datetime.now().strftime("full_%Y%m%d_%H%M%S_%f")
    )

    csv_path = run_suite(
        output,
        args.suite,
        args.repeats,
        args.workers,
        args.base_seed,
        args.max_steps,
    )


    print(
        f"Completed: {csv_path.parent.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()