from pathlib import Path
import importlib.util
import json
import random
import sys
import tempfile

import numpy as np

from model import (
    ModelConfig,
    run_simulation,
    make_probabilities,
    STRENGTHS,
    _prob_at_least_one_success,
)
from experiments import run_suite, make_tasks
from plots import generate_plots, wilson


def main():
    assert wilson(0, 100)[0] == 0
    assert wilson(100, 100)[1] == 1

    # 概率范围、均值和异质性水平。
    for n in (100, 101):
        for mean in (0, 0.15, 0.35, 1):
            spreads = []

            for level in STRENGTHS:
                probabilities = make_probabilities(
                    list(range(n)),
                    mean,
                    level,
                    random.Random(43),
                )

                values = list(probabilities.values())

                assert all(0 <= value <= 1 for value in values)
                assert np.isclose(np.mean(values), mean)

                spreads.append(np.std(values))

            assert spreads[0] <= spreads[1] <= spreads[2]

    assert np.isclose(
        _prob_at_least_one_success(0.15, 3),
        1 - 0.85 ** 3,
    )

    count = 0
    state_columns = (
        "unaware",
        "believer",
        "corrected",
        "resistant",
    )

    for network in ("random", "small_world", "scale_free"):
        for level in STRENGTHS:
            for seed in (11, 12):
                configs = [
                    ModelConfig(
                        n=100,
                        network_type=network,
                        heterogeneity=level,
                        intervention=strategy,
                        seed=seed,
                        max_steps=25,
                    )
                    for strategy in ("none", "random", "centrality")
                ]

                results = []

                for cfg in configs:
                    history, metrics, graph = run_simulation(cfg)
                    history2, metrics2, _ = run_simulation(cfg)

                    assert history == history2
                    assert metrics == metrics2

                    fact_checkers = set(json.loads(
                        metrics["fact_checker_nodes"]
                    ))
                    resistant = set(json.loads(
                        metrics["resistant_nodes"]
                    ))
                    initial = set(json.loads(
                        metrics["initial_believer_nodes"]
                    ))

                    assert not (
                        fact_checkers & resistant
                        or fact_checkers & initial
                        or resistant & initial
                    )

                    assert len(initial) == max(
                        1,
                        round(
                            len(graph) * cfg.initial_believer_fraction
                        ),
                    )

                    assert (
                        metrics["extinction_time"] is None
                    ) == metrics["ended_with_active_believers"]

                    assert all(
                        sum(row[column] for column in state_columns)
                        == len(graph)
                        for row in history
                    )

                    assert all(
                        row["resistant"] == len(resistant)
                        and row["corrected"] >= len(fact_checkers)
                        for row in history
                    )

                    results.append(metrics)
                    count += 1

                assert len({
                    metrics["graph_sha256"]
                    for metrics in results
                }) == 1

                assert len({
                    metrics["attribute_sha256"]
                    for metrics in results
                }) == 1

                assert (
                    results[1]["num_fact_checkers"]
                    == results[2]["num_fact_checkers"]
                )

    # 若保留了原版参考文件，则检查同质基线兼容。
    baseline = (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "issue2_model.py"
    )

    if baseline.exists():
        spec = importlib.util.spec_from_file_location(
            "baseline_model",
            baseline,
        )
        legacy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = legacy
        spec.loader.exec_module(legacy)

        for strategy in ("none", "random", "centrality"):
            cfg = ModelConfig(
                n=100,
                initial_believers=5,
                initial_believer_fraction=None,
                intervention=strategy,
                seed=33,
                max_steps=30,
            )

            history, metrics, _ = run_simulation(cfg)

            old_cfg = legacy.ModelConfig(
                n=100,
                intervention=strategy,
                seed=33,
                max_steps=30,
            )
            old_history, old_metrics, _ = legacy.run_simulation(old_cfg)

            assert history == old_history

            for key in (
                "final_reach",
                "peak_believer_fraction",
                "time_to_peak",
                "extinction_time",
            ):
                assert metrics[key] == old_metrics[key]

    # 检查正式实验矩阵，不在这里运行全部实验。
    tasks = make_tasks("full", 100, 1000, 300)

    assert sum(len(task[3]) for task in tasks) == 31500

    assert {
        config["n"]
        for task in tasks
        for config in task[3]
    } == {500, 1000, 5000}

    for task in tasks:
        for config in task[3]:
            assert config["initial_believer_fraction"] == 0.01
            assert config["initial_believers"] == round(
                config["n"] * 0.01
            )

    # 实际检查5000节点传播及稀疏随机网络。
    large_cases = [
        ("random", 5000, 8),
        ("small_world", 5000, 8),
        ("scale_free", 5000, 8),
        ("random", 1000, 4),
    ]

    for network, n, degree in large_cases:
        for strategy in ("none", "random", "centrality"):
            cfg = ModelConfig(
                n=n,
                mean_degree=degree,
                network_type=network,
                intervention=strategy,
                seed=77,
                max_steps=5,
            )

            history, metrics, graph = run_simulation(cfg)

            assert metrics["num_initial_believers"] == max(
                1,
                round(len(graph) * 0.01),
            )

            expected_fact_checkers = (
                0
                if strategy == "none"
                else max(1, round(len(graph) * 0.05))
            )

            assert (
                metrics["num_fact_checkers"]
                == expected_fact_checkers
            )

            assert all(
                sum(row[column] for column in state_columns)
                == len(graph)
                for row in history
            )

    # 在临时目录运行小型主实验并检查统计与绘图。
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp) / "smoke"

        run_suite(
            out,
            suite="main",
            repeats=2,
            workers=2,
            max_steps=15,
        )
        generate_plots(
            out / "raw_results.csv",
            out / "figures",
        )

        assert (
            out / "figures" / "main_final_reach.png"
        ).exists()

        try:
            run_suite(
                out,
                suite="main",
                repeats=2,
                workers=1,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing results overwritten")

    print(
        f"PASS: {count} configurations repeated; "
        "probabilities, reproducibility, shared graphs, "
        "budgets, state invariants, experiment matrix, "
        "5000-node simulation, pipeline and overwrite protection."
    )


if __name__ == "__main__":
    main()