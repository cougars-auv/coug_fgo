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
from common import COLORS
from rosbags.highlevel import AnyReader

plt.style.use(["science", "ieee"])

ALGORITHMS = ["FL-B", "iS2-B", "FL-LPI", "FL-TPI"]
NAME_MAPPING = {
    "factor_graph_node": "FL-B",
    "factor_graph_node_isam2": "iS2-B",
    "factor_graph_node_lpi": "FL-LPI",
    "factor_graph_node_tpi": "FL-TPI",
}


def load_data(bag_dir: Path, agent_name: str) -> pd.DataFrame:
    """
    Read the solver timing metrics for one agent from a bag.

    :param bag_dir: Path to the ROS 2 bag directory.
    :param agent_name: Agent namespace to read the metrics topics for.
    :return: DataFrame of per-message timing durations by algorithm.
    """
    timing_data = []
    try:
        with AnyReader([bag_dir]) as reader:
            available_topics = {c.topic: c for c in reader.connections}
            topics_to_read = [
                available_topics[f"/{agent_name}/{node}/metrics"]
                for node in NAME_MAPPING
                if f"/{agent_name}/{node}/metrics" in available_topics
            ]
            if not topics_to_read:
                return pd.DataFrame()

            topic_to_algo = {
                f"/{agent_name}/{node}/metrics": algo
                for node, algo in NAME_MAPPING.items()
            }
            for connection, _, rawdata in reader.messages(connections=topics_to_read):
                msg = reader.deserialize(rawdata, connection.msgtype)
                timing_data.append(
                    {
                        "Algorithm": topic_to_algo[connection.topic],
                        "Total": msg.total_duration,
                        "Smoother": msg.smoother_duration,
                        "Covariance": msg.cov_duration,
                    }
                )
    except Exception as e:
        print(f"Error reading bag {bag_dir}: {e}")

    return pd.DataFrame(timing_data)


def generate_plots(bag_dir: Path, output_dir: Path, agent_name: str) -> None:
    """
    Save violin and box plots of the solver timing for one agent.

    :param bag_dir: Path to the ROS 2 bag directory.
    :param output_dir: Directory to save the figures in.
    :param agent_name: Agent namespace to read the metrics topics for.
    """
    df = load_data(bag_dir, agent_name)

    if df.empty:
        return

    present_algos = [algo for algo in ALGORITHMS if algo in df["Algorithm"].unique()]
    df["Algorithm"] = pd.Categorical(
        df["Algorithm"], categories=present_algos, ordered=True
    )
    df = df.sort_values("Algorithm")

    df_melted = df.melt(
        id_vars=["Algorithm"],
        value_vars=["Total", "Smoother", "Covariance"],
        var_name="Timing Type",
        value_name="Duration (s)",
    )

    for plot_type in ["violin", "box"]:
        fig, ax = plt.subplots()

        plot_func = sns.violinplot if plot_type == "violin" else sns.boxplot
        kwargs = (
            {"inner": "box", "linewidth": 0.5, "cut": 0}
            if plot_type == "violin"
            else {}
        )

        plot_func(
            x="Timing Type",
            y="Duration (s)",
            hue="Algorithm",
            data=df_melted,
            palette=COLORS,
            ax=ax,
            **kwargs,
        )

        ax.set(title="", xlabel="", ylabel="Duration (s)")
        ax.legend(title=None)

        save_path = output_dir / f"{agent_name}_timing_{plot_type}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: timing_plot.py <target_dir>")
        return

    target_dir = Path(sys.argv[1])
    if not target_dir.exists():
        print(f"Error: {target_dir} does not exist.")
        return

    for evo_dir in target_dir.rglob("evo"):
        bag_dir = evo_dir.parent
        if not (bag_dir / "metadata.yaml").exists():
            continue
        for agent_dir in filter(Path.is_dir, evo_dir.iterdir()):
            generate_plots(bag_dir, bag_dir, agent_dir.name)

    print(f"Plots saved to bag directories in {target_dir}")


if __name__ == "__main__":
    main()
