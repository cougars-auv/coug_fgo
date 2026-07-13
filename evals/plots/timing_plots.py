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

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from rosbags.highlevel import AnyReader

from utils import estimators, evo_tools

logger = logging.getLogger(__name__)

plt.style.use(["science", "ieee"])

COLORS = estimators.color_map()
ALGORITHMS = [e.label for e in estimators.timed_estimators()]
NAME_MAPPING = {e.node: e.label for e in estimators.timed_estimators()}


def read_timing_metrics(bag_dir: Path, agent_name: str) -> pd.DataFrame:
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
            topic_to_algo = {
                f"/{agent_name}/{node}/metrics": algo
                for node, algo in NAME_MAPPING.items()
            }
            topics_to_read = [
                available_topics[t] for t in topic_to_algo if t in available_topics
            ]
            if not topics_to_read:
                return pd.DataFrame()
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
        logger.warning(f"Could not read {bag_dir}: {e}")

    return pd.DataFrame(timing_data)


def render(target_dir: Path) -> None:
    """
    Save solver timing plots for every evaluated agent under a target directory.

    :param target_dir: A bag or directory of bags that has been evaluated.
    """
    for bag_dir, agent_dir in evo_tools.iter_evaluated_agents(target_dir):
        df = read_timing_metrics(bag_dir, agent_dir.name)
        if df.empty:
            continue

        present_algos = [
            algo for algo in ALGORITHMS if algo in df["Algorithm"].unique()
        ]
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
            fig.savefig(
                bag_dir / f"{agent_dir.name}_timing_{plot_type}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
