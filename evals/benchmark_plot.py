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

import glob
import os
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee"])

ALGORITHMS = [
    "FL-B",
    "iS2-B",
    "FL-LPI",
    "FL-TPI",
    "IEKF",
    # "UKF",
    # "EKF",
    "TM",
    "AQS",
    "DVL",
]
COLORS = {
    "FL-B": "#55A868",
    "iS2-B": "#DD8452",
    "FL-LPI": "#4C72B0",
    "FL-TPI": "#C44E52",
    "IEKF": "#8172B2",
    "UKF": "#937860",
    "EKF": "#DA8BC3",
    "TM": "#8C8C8C",
    "AQS": "#CCB974",
    "DVL": "#64B5CD",
}
NAME_MAPPING = {
    "global": "FL-B",
    "global_isam2": "iS2-B",
    "global_lpi": "FL-LPI",
    "global_tpi": "FL-TPI",
    "global_iekf": "IEKF",
    "global_ukf": "UKF",
    "global_ekf": "EKF",
    "global_tm": "TM",
    "global_aqs": "AQS",
    "dvl": "DVL",
}

METRICS_CONFIG = [
    ("benchmark_ape_trans.csv", "APE Translation RMSE (m)", "ape_trans"),
    ("benchmark_ape_rot.csv", "APE Rotation RMSE (deg)", "ape_rot"),
    ("benchmark_rpe_trans.csv", "RPE Translation RMSE (m/m)", "rpe_trans"),
    ("benchmark_rpe_rot.csv", "RPE Rotation RMSE (deg/m)", "rpe_rot"),
]


def load_data(bags_dir: Path) -> dict[str, pd.DataFrame]:
    data_store = {cfg[0]: [] for cfg in METRICS_CONFIG}
    files = glob.glob(os.path.join(bags_dir, "**", "benchmark_*.csv"), recursive=True)
    sorted_mapping = sorted(NAME_MAPPING.items(), key=lambda x: len(x[0]), reverse=True)

    for f in files:
        filename = os.path.basename(f)
        if filename in data_store:
            try:
                df = pd.read_csv(f, index_col=0)
                for algo_key, row in df.iterrows():
                    label = algo_key
                    for key, val in sorted_mapping:
                        if key in str(algo_key):
                            label = val
                            break

                    if label in ALGORITHMS:
                        data_store[filename].append(
                            {"Algorithm": label, "RMSE": row["rmse"]}
                        )
            except Exception as e:
                print(f"Error reading {f}: {e}")

    return {k: pd.DataFrame(v) for k, v in data_store.items() if v}


def generate_plots(data_map: dict[str, pd.DataFrame], output_dir: Path) -> None:
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
