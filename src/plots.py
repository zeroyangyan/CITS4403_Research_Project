from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from model import ModelConfig, run_simulation


NETWORK_ORDER = ["random", "small_world", "scale_free"]
INTERVENTION_ORDER = ["none", "random", "centrality"]


def plot_main_results(csv_path: str, output_dir: str) -> None:
    """读取全因子实验CSV，并生成两张汇总柱状图和统计表。"""
    # 读取实验结果，并确保输出目录存在。
    df = pd.read_csv(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 绘图至少需要以下四列；若缺少则提前报错并指出缺失列。
    required = {
        "network_type",
        "intervention",
        "final_reach",
        "peak_believer_fraction",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # 图1：按“网络结构+干预策略”分组，计算最终传播范围的均值和标准差。
    summary = (
        df.groupby(["network_type", "intervention"])["final_reach"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # pivot把长表转换为适合分组柱状图的宽表：
    # 行是网络结构，列是干预策略，单元格是统计值。
    mean_pivot = summary.pivot(
        index="network_type", columns="intervention", values="mean"
    ).reindex(index=NETWORK_ORDER, columns=INTERVENTION_ORDER)
    std_pivot = summary.pivot(
        index="network_type", columns="intervention", values="std"
    ).reindex(index=NETWORK_ORDER, columns=INTERVENTION_ORDER)

    # yerr使用标准差作为误差线，capsize控制误差线端点宽度。
    ax = mean_pivot.plot(
        kind="bar",
        yerr=std_pivot,
        capsize=3,
        figsize=(9, 5.5),
    )
    ax.set_ylabel("Mean final reach")
    ax.set_xlabel("Network structure")
    ax.set_ylim(0, 1.05)
    ax.set_title("Misinformation reach by network and intervention")
    ax.legend(title="Intervention")
    plt.xticks(rotation=0)
    plt.tight_layout()
    # dpi=200让导出的PNG在报告中保持清晰。
    plt.savefig(out / "final_reach_bar.png", dpi=200)
    plt.close()

    # 图2：统计每组实验中相信者最高比例的均值和标准差。
    peak_summary = (
        df.groupby(["network_type", "intervention"])["peak_believer_fraction"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    peak_mean = peak_summary.pivot(
        index="network_type", columns="intervention", values="mean"
    ).reindex(index=NETWORK_ORDER, columns=INTERVENTION_ORDER)
    peak_std = peak_summary.pivot(
        index="network_type", columns="intervention", values="std"
    ).reindex(index=NETWORK_ORDER, columns=INTERVENTION_ORDER)

    # 使用与图1一致的网络和策略顺序，方便两张图相互比较。
    ax = peak_mean.plot(
        kind="bar",
        yerr=peak_std,
        capsize=3,
        figsize=(9, 5.5),
    )
    ax.set_ylabel("Mean peak believer fraction")
    ax.set_xlabel("Network structure")
    ax.set_ylim(0, 1.05)
    ax.set_title("Peak misinformation prevalence")
    ax.legend(title="Intervention")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "peak_believers_bar.png", dpi=200)
    plt.close()

    # 同时保存绘图所依据的汇总数据，便于报告引用和检查。
    summary.to_csv(out / "main_summary.csv", index=False)
    peak_summary.to_csv(out / "peak_summary.csv", index=False)


def plot_representative_time_series(output_dir: str, n: int = 300, seed: int = 42) -> None:
    """在同一个网络上绘制三种干预策略的代表性传播时间序列。

    三种策略使用相同的网络类型和随机种子，以提高可比性。
    这张图只用于直观展示；正式结论应来自多次重复实验，而不是单次运行。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 使用无标度网络，因为它最能体现高连接节点的重要性。
    network_type = "scale_free"
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # 分别运行无干预、随机核查和中心性核查，并画在同一坐标系中。
    for intervention in INTERVENTION_ORDER:
        cfg = ModelConfig(
            n=n,
            mean_degree=8,
            network_type=network_type,
            intervention=intervention,
            fact_checker_fraction=0.05 if intervention != "none" else 0.0,
            misinformation_prob=0.15,
            correction_prob=0.35,
            resistant_fraction=0.05,
            initial_believers=5,
            max_steps=150,
            seed=seed,
        )
        # history保存每一个时间步的状态人数；其他返回值在此处不需要。
        history, _, _ = run_simulation(cfg)
        hist = pd.DataFrame(history)
        # 除以n，把相信者人数转换为占总用户数的比例。
        ax.plot(hist["step"], hist["believer"] / cfg.n, label=intervention)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Believer fraction")
    ax.set_ylim(bottom=0)
    ax.set_title("Representative misinformation dynamics (scale-free network)")
    ax.legend(title="Intervention")
    plt.tight_layout()
    plt.savefig(out / "representative_time_series.png", dpi=200)
    plt.close()


def main() -> None:"""读取命令行参数，根据指定CSV生成结果图。"""
    parser = argparse.ArgumentParser(description="Create project result figures")
    parser.add_argument(
        "--input",
        default="../results/main_experiment.csv",
        help="Experiment CSV created by experiments.py",
    )
    parser.add_argument(
        "--output-dir",
        default="../results/figures",
        help="Directory in which figures and summaries are written",
    )
    parser.add_argument(
        "--time-series",
        action="store_true",
        help="Also create one representative time-series figure",
    )
    parser.add_argument("--time-series-n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 总是生成两张汇总柱状图；只有提供time-series时才生成时间序列图。
    plot_main_results(args.input, args.output_dir)
    if args.time_series:
        plot_representative_time_series(
            args.output_dir, n=args.time_series_n, seed=args.seed
        )

    print(f"Figures saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
