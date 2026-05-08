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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from rosbags.highlevel import AnyReader
import yaml

plt.style.use(["science", "ieee"])

FGO_COLOR = "#55A868"
FGO_TOPIC = "factor_graph_node/metrics"


def get_smoother_lag(bag_dir: Path, agent_name: str) -> float | None:
    config_paths = [
        bag_dir / "config" / "fleet" / "coug_fgo_params.yaml",
        bag_dir / "config" / f"{agent_name}_params.yaml",
    ]

    params: dict = {}
    found_any = False
    for path in config_paths:
        if not path.exists():
            continue
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
            for key in ["/**", f"/{agent_name}/**"]:
                params.update(config.get(key, {}).get("ros__parameters", {}))
            found_any = True
        except Exception:
            continue

    if not found_any or "smoother_lag" not in params:
        return None
    return float(params["smoother_lag"])


def load_lag_data(target_dir: Path) -> pd.DataFrame:
    timing_data = []

    for evo_dir in target_dir.rglob("evo"):
        bag_dir = evo_dir.parent
        for agent_dir in evo_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            agent_name = agent_dir.name
            lag = get_smoother_lag(bag_dir, agent_name)
            if lag is None:
                print(f"No smoother_lag found in {bag_dir} for {agent_name}, skipping.")
                continue

            try:
                with AnyReader([bag_dir]) as reader:
                    connections = [
                        c
                        for c in reader.connections
                        if c.topic == f"/{agent_name}/{FGO_TOPIC}"
                    ]
                    if not connections:
                        continue
                    for connection, _, rawdata in reader.messages(
                        connections=connections
                    ):
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        timing_data.append(
                            {
                                "Smoother Lag (s)": lag,
                                "Duration (s)": msg.total_duration,  # type: ignore[union-attr]
                            }
                        )
            except Exception as e:
                print(f"Error reading bag {bag_dir}: {e}")

    return pd.DataFrame(timing_data)


def generate_plots(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
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

        save_path = output_dir / f"lag_{plot_type}.png"
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: lag_plot.py <target_dir>")
        return

    target_dir = Path(sys.argv[1])
    if not target_dir.exists():
        print(f"Error: {target_dir} does not exist.")
        return

    generate_plots(load_lag_data(target_dir), target_dir)
    print(f"Plots saved to {target_dir}")


if __name__ == "__main__":
    main()
