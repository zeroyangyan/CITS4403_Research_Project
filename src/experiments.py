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
    运行主要的全因子实验：
    3种网络结构 × 3种干预策略 × 多个随机种子重复实验。

    除网络结构和干预策略外，其余模型参数保持固定，
    这样可以更清楚地比较这两个因素对结果的影响。
    """
    # 需要比较的三个网络结构和三种事实核查策略。
    networks = ["random", "small_world", "scale_free"]
    interventions = ["none", "random", "centrality"]

    rows = []
    # itertools.product 会生成所有“网络×策略×重复编号”的组合。
    for network_type, intervention, rep in itertools.product(
        networks, interventions, range(repeats)
    ):
        # 无干预组的事实核查者比例为0；其他两组均使用5%，保证预算相同。
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
            # 相同 rep 在不同实验条件中使用相同种子，便于公平比较和复现。
            seed=1000 + rep,
        )
        # 只保存汇总指标，不保存每一步的历史记录和网络对象。
        _, metrics, _ = run_simulation(cfg)
        metrics["repeat"] = rep
        rows.append(metrics)

    # 将指标整理为表格，并保存为CSV。
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
    独立扩展实验：改变事实核查者比例（即干预预算），
    比较“随机选择”和“优先选择高连接用户”之间的差异。

    默认在无标度网络上进行，因为高连接节点在这种网络中最明显。
    """
    # 分别使用1%、2.5%、5%、7.5%和10%的事实核查者。
    fractions = [0.01, 0.025, 0.05, 0.075, 0.10]
    strategies = ["random", "centrality"]

    rows = []
    # 遍历所有“预算比例×选择策略×重复编号”的组合。
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
            # 使用与主实验不同的种子范围，避免混淆两组实验。
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
    """读取命令行参数，选择并运行主实验或预算扩展实验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["main", "budget"], default="main")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="../results/main_experiment.csv")
    args = parser.parse_args()

    # 根据实验的值调用相应的实验函数。
    if args.experiment == "main":
        df = run_factorial_experiment(args.output, n=args.n, repeats=args.repeats)
    else:
        df = run_budget_sweep(args.output, n=args.n, repeats=args.repeats)

    
    print(df.groupby(["network_type", "intervention"])["final_reach"]
          .agg(["mean", "std", "count"]))# 在终端显示每组最终传播范围的平均值、标准差和实验次数。


if __name__ == "__main__":
    main()
