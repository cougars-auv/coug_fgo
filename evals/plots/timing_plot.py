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

import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from rosbags.highlevel import AnyReader

plt.style.use(["science", "ieee"])

ALGORITHMS = ["FL-B", "iS2-B", "FL-LPI", "FL-TPI", "IEKF", "UKF", "EKF", "DVL"]
COLORS = {
    "FL-B": "#55A868",
    "iS2-B": "#DD8452",
    "FL-LPI": "#4C72B0",
    "FL-TPI": "#C44E52",
    "IEKF": "#8172B2",
    "UKF": "#937860",
    "EKF": "#DA8BC3",
    "DVL": "#64B5CD",
}
NAME_MAPPING = {
    "factor_graph_node": "FL-B",
    "factor_graph_node_isam2": "iS2-B",
    "factor_graph_node_lpi": "FL-LPI",
    "factor_graph_node_tpi": "FL-TPI",
}


def load_timing_data(bag_dir, agent_name):
    timing_data = []
    bag_path = Path(bag_dir)

    try:
        with AnyReader([bag_path]) as reader:
            available_topics = {c.topic: c for c in reader.connections}
            topics_to_read = []
            topic_to_algo = {}

            for node_name, algo in NAME_MAPPING.items():
                target_topic = f"/{agent_name}/{node_name}/metrics"
                if target_topic in available_topics:
                    topics_to_read.append(available_topics[target_topic])
                    topic_to_algo[target_topic] = algo

            if not topics_to_read:
                return pd.DataFrame()

            for connection, timestamp, rawdata in reader.messages(
                connections=topics_to_read
            ):
                msg = reader.deserialize(rawdata, connection.msgtype)
                algo = topic_to_algo[connection.topic]

                timing_data.append(
                    {
                        "Algorithm": algo,
                        "Total": msg.total_duration,
                        "Smoother": msg.smoother_duration,
                        "Covariance": msg.cov_duration,
                    }
                )
    except Exception as e:
        print(f"Error reading bag {bag_dir}: {e}")

    return pd.DataFrame(timing_data)


def plot_timing(bag_dir, output_dir, agent_name):
    df = load_timing_data(bag_dir, agent_name)

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

        save_path = Path(output_dir) / f"{agent_name}_timing_{plot_type}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    bags_root = (
        Path(os.environ.get("HOME", os.path.expanduser("~"))) / "cougars-dev" / "bags"
    )

    if not bags_root.exists():
        print(f"Error: {bags_root} does not exist.")
        return

    for evo_dir in bags_root.rglob("evo"):
        bag_dir = evo_dir.parent
        for agent_dir in evo_dir.iterdir():
            if agent_dir.is_dir():
                plot_timing(str(bag_dir), str(bag_dir), agent_dir.name)

    print(f"Plots saved to bag directories in {bags_root}")


if __name__ == "__main__":
    main()
