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

import argparse
import logging
from pathlib import Path

from utils import evaluation
from utils.log_setup import setup_logging

logger = logging.getLogger(__name__)

TARGET_DIR = Path.home() / "cougars-dev/bags/turtlmap_offline"
AGENTS = ["coug1sim", "coug2sim", "coug3sim", "blue1sim", "bluerov2", "turtlmap"]
EVO_FLAGS = ["--align"]  # , "--project_to_plane", "xy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=TARGET_DIR,
        help="A bag directory or a directory of bags to evaluate",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=AGENTS,
        help="AUV namespaces to evaluate",
    )
    parser.add_argument(
        "--evo-flags",
        default=" ".join(EVO_FLAGS),
        help="Extra evo flags forwarded to APE and RPE runs, e.g. "
        "--evo-flags='--align --project_to_plane xy'",
    )
    args = parser.parse_args()

    setup_logging()
    evaluation.evaluate_bags(args.target_dir, args.agents, args.evo_flags.split())


if __name__ == "__main__":
    main()
