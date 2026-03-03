# 🌊 CoUGARs Factor Graph Odometry

[![ROS 2 CI](https://github.com/cougars-auv/coug_fgo/actions/workflows/ros2_ci.yml/badge.svg)](https://github.com/cougars-auv/coug_template/actions/workflows/ros2_ci.yml)
[![Docker CI](https://github.com/cougars-auv/coug_fgo/actions/workflows/docker_ci.yml/badge.svg)](https://github.com/cougars-auv/coug_template/actions/workflows/docker_ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/cougars-auv/cougars_fgo/main.svg)](https://results.pre-commit.ci/latest/github/cougars-auv/cougars_fgo/main)

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
