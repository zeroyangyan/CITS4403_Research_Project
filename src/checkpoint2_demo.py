"""One-command demonstration for CITS4403 Checkpoint 2.

This script intentionally uses a smaller experiment than the final report so that
it is convenient for a live checkpoint demonstration. Do not use these quick-run
settings as the final statistical experiment.
"""
from pathlib import Path

from experiments import run_factorial_experiment
from plots import plot_main_results, plot_representative_time_series


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results" / "checkpoint2"
    figures_dir = results_dir / "figures"
    csv_path = results_dir / "checkpoint2_experiment.csv"

    results_dir.mkdir(parents=True, exist_ok=True)

    print("Running Checkpoint 2 demonstration experiment...")
    df = run_factorial_experiment(
        str(csv_path),
        n=200,
        repeats=5,
        mean_degree=8,
        max_steps=100,
    )

    print("\nPreliminary mean final reach:")
    print(
        df.groupby(["network_type", "intervention"])["final_reach"]
        .agg(["mean", "std", "count"])
        .round(3)
    )

    plot_main_results(str(csv_path), str(figures_dir))
    plot_representative_time_series(str(figures_dir), n=200, seed=42)

    print("\nCheckpoint 2 outputs created:")
    print(f"  CSV:     {csv_path}")
    print(f"  Figures: {figures_dir}")
    print("\nFor the final report, run the full experiment with >=30 repeats per condition.")


if __name__ == "__main__":
    main()
