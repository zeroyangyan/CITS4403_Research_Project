from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_main_results(csv_path: str, output_dir: str) -> None:
    df = pd.read_csv(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Mean final reach by network and intervention
    summary = (
        df.groupby(["network_type", "intervention"])["final_reach"]
        .agg(["mean", "std"])
        .reset_index()
    )

    pivot = summary.pivot(index="network_type", columns="intervention", values="mean")
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Mean final reach")
    ax.set_xlabel("Network structure")
    ax.set_title("Misinformation reach by network and intervention")
    plt.tight_layout()
    plt.savefig(out / "final_reach_bar.png", dpi=200)
    plt.close()

    # 2. Mean peak believer fraction
    pivot2 = (
        df.groupby(["network_type", "intervention"])["peak_believer_fraction"]
        .mean()
        .unstack()
    )
    ax = pivot2.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Mean peak believer fraction")
    ax.set_xlabel("Network structure")
    ax.set_title("Peak misinformation prevalence")
    plt.tight_layout()
    plt.savefig(out / "peak_believers_bar.png", dpi=200)
    plt.close()

    summary.to_csv(out / "main_summary.csv", index=False)


if __name__ == "__main__":
    plot_main_results("../results/main_experiment.csv", "../results/figures")
