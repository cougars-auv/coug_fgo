#!/usr/bin/env python3
# Copyright (c) 2026 BYU FROST Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from common import ALGORITHMS, COLORS, NAME_MAPPING

plt.style.use(["science", "ieee"])

METRICS_CONFIG = [
    ("benchmark_ape_trans.csv", "APE Translation RMSE (m)", "ape_trans"),
    ("benchmark_ape_rot.csv", "APE Rotation RMSE (deg)", "ape_rot"),
    ("benchmark_rpe_trans.csv", "RPE Translation RMSE (m/m)", "rpe_trans"),
    ("benchmark_rpe_rot.csv", "RPE Rotation RMSE (deg/m)", "rpe_rot"),
]


def load_data(bags_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Collect benchmark CSV rows from all bags into per-metric DataFrames.

    :param bags_dir: Directory tree containing exported benchmark CSV files.
    :return: DataFrames of algorithm RMSE values keyed by CSV file name.
    """
    data_store = {cfg[0]: [] for cfg in METRICS_CONFIG}
    sorted_mapping = sorted(NAME_MAPPING.items(), key=lambda x: len(x[0]), reverse=True)

    for path in sorted(bags_dir.rglob("benchmark_*.csv")):
        if path.name not in data_store:
            continue
        try:
            df = pd.read_csv(path, index_col=0)
            for algo_key, row in df.iterrows():
                label = algo_key
                for key, val in sorted_mapping:
                    if key in str(algo_key):
                        label = val
                        break

                if label in ALGORITHMS:
                    data_store[path.name].append(
                        {"Algorithm": label, "RMSE": row["rmse"]}
                    )
        except Exception as e:
            print(f"Error reading {path}: {e}")

    return {k: pd.DataFrame(v) for k, v in data_store.items() if v}


def generate_plots(data_map: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    Save violin, box, and strip plots for each benchmark metric.

    :param data_map: DataFrames of algorithm RMSE values keyed by CSV name.
    :param output_dir: Directory to save the figures in.
    """
    for filename, label, prefix in METRICS_CONFIG:
        if filename not in data_map or data_map[filename].empty:
            continue

        df = data_map[filename]

        present_algos = [
            algo for algo in ALGORITHMS if algo in df["Algorithm"].unique()
        ]
        df["Algorithm"] = pd.Categorical(
            df["Algorithm"], categories=present_algos, ordered=True
        )
        df = df.sort_values("Algorithm")

        for plot_type in ["violin", "box", "strip"]:
            fig, ax = plt.subplots()

            if plot_type == "strip":
                sns.stripplot(
                    x="Algorithm",
                    y="RMSE",
                    data=df,
                    hue="Algorithm",
                    palette=COLORS,
                    legend=False,
                    ax=ax,
                    jitter=False,
                    marker="x",
                    linewidth=1.0,
                )
            else:
                plot_func = sns.violinplot if plot_type == "violin" else sns.boxplot
                kwargs = (
                    {"inner": "box", "linewidth": 0.5, "cut": 0}
                    if plot_type == "violin"
                    else {}
                )

                plot_func(
                    x="Algorithm",
                    y="RMSE",
                    data=df,
                    hue="Algorithm",
                    dodge=False,
                    palette=COLORS,
                    legend=False,
                    ax=ax,
                    **kwargs,
                )

            ax.set(title="", xlabel="", ylabel=label)
            save_path = output_dir / f"{prefix}_{plot_type}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: benchmark_plot.py <target_dir>")
        return

    target_dir = Path(sys.argv[1])
    if not target_dir.exists():
        print(f"Error: {target_dir} does not exist.")
        return

    generate_plots(load_data(target_dir), target_dir)
    print(f"Plots saved to {target_dir}")


if __name__ == "__main__":
    main()
