from datetime import datetime
from pathlib import Path

from experiments import run_factorial_experiment
from plots import plot_main_results


def main():
    # 每次使用新目录，保留之前的结果。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (
        Path(__file__).resolve().parent
        / "results"
        / f"demo_{timestamp}"
    )

    csv_path = output / "demo.csv"

    df = run_factorial_experiment(
        str(csv_path),
        n=200,
        repeats=5,
        mean_degree=8,
        max_steps=100,
    )

    plot_main_results(str(csv_path), str(output / "figures"))

    print(
        df.groupby(
            ["network_type", "heterogeneity", "intervention"]
        )["final_reach"].agg(["mean", "std", "count"]).round(3)
    )

    print(f"\nOutputs: {output}")
    print("This demonstration is not the final experiment.")


if __name__ == "__main__":
    main()