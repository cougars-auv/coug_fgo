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
import yaml
from rosbags.highlevel import AnyReader

from utils import estimators, evo_tools

logger = logging.getLogger(__name__)

plt.style.use(["science", "ieee"])

_FGO = estimators.timed_estimators()[0]
FGO_COLOR = _FGO.color
FGO_TOPIC = f"{_FGO.node}/metrics"


def _get_smoother_lag(bag_dir: Path, agent_name: str) -> float | None:
    """
    Read the smoother lag parameter from a bag's saved config files.

    :param bag_dir: Path to the ROS 2 bag directory.
    :param agent_name: Agent namespace used to select namespaced parameters.
    :return: The smoother lag in seconds, or None if it was not found.
    """
    config_paths = [
        bag_dir / "config" / "fleet" / "coug_fgo_params.yaml",
        bag_dir / "config" / f"{agent_name}_params.yaml",
    ]

    params: dict = {}
    for path in config_paths:
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            continue
        try:
            params.update(config["/**"]["ros__parameters"])
        except (KeyError, TypeError):
            pass
        try:
            params.update(config[f"/{agent_name}"]["**"]["ros__parameters"])
        except (KeyError, TypeError):
            pass

    if "smoother_lag" not in params:
        return None
    return float(params["smoother_lag"])


def _read_bag_durations(bag_dir: Path, agent_name: str) -> list[float]:
    """
    Read the total solver durations from a bag's metrics topic.

    :param bag_dir: Path to the ROS 2 bag directory.
    :param agent_name: Agent namespace to read the metrics topic for.
    :return: Total optimization durations in seconds.
    """
    topic = f"/{agent_name}/{FGO_TOPIC}"
    try:
        with AnyReader([bag_dir]) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            if not connections:
                return []
            return [
                float(reader.deserialize(rawdata, c.msgtype).total_duration)
                for c, _, rawdata in reader.messages(connections=connections)
            ]
    except Exception as e:
        logger.warning(f"Could not read {bag_dir}: {e}")
        return []


def _collect_durations_by_lag(target_dir: Path) -> pd.DataFrame:
    """
    Collect solver durations and smoother lags from all evaluated bags.

    :param target_dir: Directory tree containing evaluated bags.
    :return: DataFrame pairing each duration with its configured lag.
    """
    timing_data = []

    for bag_dir, agent_dir in evo_tools.iter_evaluated_agents(target_dir):
        lag = _get_smoother_lag(bag_dir, agent_dir.name)
        if lag is None:
            logger.warning(
                f"No smoother_lag found in {bag_dir} for {agent_dir.name}, skipping."
            )
            continue

        for duration in _read_bag_durations(bag_dir, agent_dir.name):
            timing_data.append(
                {
                    "Smoother Lag (s)": lag,
                    "Duration (s)": duration,
                }
            )

    return pd.DataFrame(timing_data)


def render(target_dir: Path) -> None:
    """
    Save violin and box plots of solver duration by smoother lag.

    :param target_dir: A bag or directory of bags that has been evaluated.
    """
    logger.info("Rendering lag plots...")
    df = _collect_durations_by_lag(target_dir)
    if df.empty:
        logger.error(f"No solver durations with smoother lags found for {target_dir}.")
        return

    lag_values = sorted(df["Smoother Lag (s)"].unique())
    df["Smoother Lag (s)"] = pd.Categorical(
        df["Smoother Lag (s)"], categories=lag_values, ordered=True
    )
    df = df.sort_values("Smoother Lag (s)")

    for plot_type in ["violin", "box"]:
        fig, ax = plt.subplots()

        plot_func = sns.violinplot if plot_type == "violin" else sns.boxplot
        kwargs = (
            {"inner": "box", "linewidth": 0.5, "cut": 0}
            if plot_type == "violin"
            else {}
        )

        plot_func(
            x="Smoother Lag (s)",
            y="Duration (s)",
            data=df,
            color=FGO_COLOR,
            ax=ax,
            **kwargs,
        )

        ax.set(title="", xlabel="Smoother Lag (s)", ylabel="Duration (s)")
        out_path = target_dir / f"lag_{plot_type}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot: {out_path}")
