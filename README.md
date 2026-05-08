# 🌊 CoUGARs Factor Graph Odometry

[![ROS 2 Build & Test](https://github.com/cougars-auv/coug_fgo/actions/workflows/ros2_build_and_test.yml/badge.svg)](https://github.com/cougars-auv/coug_fgo/actions/workflows/ros2_build_and_test.yml)
[![Docker Build](https://github.com/cougars-auv/coug_fgo/actions/workflows/docker_build.yml/badge.svg)](https://github.com/cougars-auv/coug_fgo/actions/workflows/docker_build.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/cougars-auv/coug_fgo/main.svg)](https://results.pre-commit.ci/latest/github/cougars-auv/coug_fgo/main)
[![codecov](https://codecov.io/gh/cougars-auv/coug_fgo/graph/badge.svg?token=OFBQI4IOT1)](https://codecov.io/gh/cougars-auv/coug_fgo)

## 🤝 Contributing

- **Create a Branch:** Create a new branch using the format `name/feature` (e.g., `nelson/repo-docs`).

- **Make Changes:** Develop and debug your new feature. Add good documentation.

  > If you need to add dependencies, update the `package.xml`, `Dockerfile`, `cougars.repos`, or `dependencies.repos` in your branch and test building the image locally. The CI will automatically build and push the new image to Docker Hub upon merge.

- **Sync Frequently:** Regularly rebase your branch against `main` (or merge `main` into your branch) to prevent conflicts.

- **Submit a PR:** Open a pull request, ensure required tests pass, and merge once approved.

## 📚 Citations

Please cite our relevant publications if you find this repository useful for your research:

### CoUGARs
```bibtex
@misc{durrant2025lowcostmultiagentfleetacoustic,
  title={Low-cost Multi-agent Fleet for Acoustic Cooperative Localization Research},
  author={Nelson Durrant and Braden Meyers and Matthew McMurray and Clayton Smith and Brighton Anderson and Tristan Hodgins and Kalliyan Velasco and Joshua G. Mangelson},
  year={2025},
  eprint={2511.08822},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2511.08822},
}
```
