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
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class UrdfTree:
    """
    Offline TF resolver parsed from a URDF or xacro description.

    :author: Nelson Durrant
    :date: July 2026
    """

    def __init__(self, urdf_path: str):
        """
        Parse the joint tree from a robot description file.

        :param urdf_path: Path to a URDF or xacro robot description.
        :raises ValueError: If a joint is missing its parent or child link.
        """
        path = Path(urdf_path)
        if path.suffix == ".xacro":
            import xacro

            xml_text = xacro.process_file(str(path)).toxml()
        else:
            xml_text = path.read_text()

        self._joints: dict[str, tuple[str, np.ndarray, Rotation]] = {}
        self._links: set[str] = set()
        for joint in ET.fromstring(xml_text).findall("joint"):
            parent_el, child_el = joint.find("parent"), joint.find("child")
            if parent_el is None or child_el is None:
                raise ValueError(
                    f"URDF joint '{joint.attrib.get('name', '?')}' is missing a "
                    "parent or child link."
                )
            parent = parent_el.attrib["link"]
            child = child_el.attrib["link"]
            origin = joint.find("origin")
            attrib = origin.attrib if origin is not None else {}
            pos = np.array([float(v) for v in attrib.get("xyz", "0 0 0").split()])
            rpy = [float(v) for v in attrib.get("rpy", "0 0 0").split()]
            self._joints[child] = (parent, pos, Rotation.from_euler("xyz", rpy))
            self._links.update((parent, child))

    def lookup(
        self, target_frame: str, source_frame: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the static transform between two frames in the tree.

        :param target_frame: Frame to express the transform in.
        :param source_frame: Frame whose pose is being looked up.
        :return: Position and xyzw quaternion of the source in the target.
        """
        target_pos, target_rot = self._root_tf(target_frame)
        source_pos, source_rot = self._root_tf(source_frame)
        pos = target_rot.inv().apply(source_pos - target_pos)
        rot = target_rot.inv() * source_rot
        return pos, rot.as_quat()

    def _root_tf(self, frame: str) -> tuple[np.ndarray, Rotation]:
        """
        Accumulate the fixed transform from the URDF root to a frame.

        :param frame: Frame name (frame_prefix stripped before lookup).
        :return: Position and rotation of the frame in the root link.
        :raises KeyError: If the frame is not present in the URDF.
        """
        link = frame.split("/")[-1]  # Strip robot_state_publisher frame_prefix
        if link not in self._links:
            raise KeyError(f"Frame '{frame}' not found in the URDF.")

        pos, rot = np.zeros(3), Rotation.identity()
        while link in self._joints:
            link, j_pos, j_rot = self._joints[link]
            pos = j_pos + j_rot.apply(pos)
            rot = j_rot * rot
        return pos, rot


def resolve_urdf_path(namespace: str, config_paths: list[str]) -> str | None:
    """
    Find the URDF file referenced by the coug_description_launch parameters.

    :param namespace: AUV namespace used to select namespaced parameters.
    :param config_paths: Parameter YAML files to search for a urdf_file entry.
    :return: Absolute path to the URDF file, or None if it was not found.
    """

    def read_urdf_file(yaml_path: Path, top_keys: list[str]) -> str | None:
        """
        Read the urdf_file parameter from one YAML file, if present.

        :param yaml_path: Path to the parameter YAML file.
        :param top_keys: Top-level namespace keys to try, in order.
        :return: The urdf_file value, or None if it was not found.
        """
        try:
            data = yaml.safe_load(yaml_path.read_text())
        except OSError:
            return None
        except yaml.YAMLError as e:
            logger.error(f"Could not parse params file {yaml_path}: {e}")
            return None
        for top_key in top_keys:
            try:
                return data[top_key]["coug_description_launch"]["ros__parameters"][
                    "urdf_file"
                ]
            except (KeyError, TypeError):
                continue
        return None

    urdf_file = None
    for path in map(Path, config_paths):
        urdf_file = read_urdf_file(path, [f"/{namespace}", "/**"]) or urdf_file
    if urdf_file is None:
        for path in map(Path, config_paths):
            for config_dir in (path.parent / "fleet", path.parent):
                fleet_path = config_dir / "coug_description_params.yaml"
                urdf_file = urdf_file or read_urdf_file(fleet_path, ["/**"])
    if urdf_file is None:
        logger.error("No urdf_file parameter found in any config.")
        return None

    urdf_dirs = [
        Path.home() / "cougars-dev/ros2_ws/src/coug_description/coug_description/urdf",
        Path.home()
        / "cougars-dev/ros2_ws/install/coug_description/share/coug_description/urdf",
    ]
    try:
        from ament_index_python.packages import get_package_share_directory

        urdf_dirs.insert(
            0, Path(get_package_share_directory("coug_description")) / "urdf"
        )
    except Exception:
        pass

    for urdf_dir in urdf_dirs:
        if (urdf_dir / urdf_file).is_file():
            return str(urdf_dir / urdf_file)
    logger.error(f"Could not locate urdf_file '{urdf_file}' in any URDF directory.")
    return None
