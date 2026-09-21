"""CITS4403 Checkpoint2的一键演示程序。

为了方便展示，使用比最终报告更小的实验规模。
这些快速运行参数只能展示当前进度，不能直接作为最终结果。
"""
from pathlib import Path

from experiments import run_factorial_experiment
from plots import plot_main_results, plot_representative_time_series


def main() -> None:
    """运行小规模实验，并自动生成Checkpoint2所需的CSV与图表。"""
    # 根据当前脚本的位置确定项目根目录，避免依赖用户从哪个目录运行程序。
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results" / "checkpoint2"
    figures_dir = results_dir / "figures"
    csv_path = results_dir / "checkpoint2_experiment.csv"

    # 如果结果文件夹不存在，就自动建立。
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Running Checkpoint 2 demonstration experiment...")
    # 运行3种网络×3种策略，每个条件重复5次，共45次仿真。
    # n=200和repeats=5是为了让演示快速完成。
    df = run_factorial_experiment(
        str(csv_path),
        n=200,
        repeats=5,
        mean_degree=8,
        max_steps=100,
    )

    print("\nPreliminary mean final reach:")
    # 按网络结构和干预策略分组，输出最终传播范围的初步统计结果。
    print(
        df.groupby(["network_type", "intervention"])["final_reach"]
        .agg(["mean", "std", "count"])
        .round(3)
    )

    # 生成两张汇总柱状图和一张代表性时间序列图。
    plot_main_results(str(csv_path), str(figures_dir))
    plot_representative_time_series(str(figures_dir), n=200, seed=42)

    print("\nCheckpoint 2 outputs created:")
    print(f"  CSV:     {csv_path}")
    print(f"  Figures: {figures_dir}")
    # 每个条件应至少重复30次。
    print("\nFor the final report, run the full experiment with >=30 repeats per condition.")


if __name__ == "__main__":
    main()
