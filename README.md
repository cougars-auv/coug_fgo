# 🌊 CoUGARs Factor Graph Odometry

[![ROS 2 Build & Test](https://github.com/cougars-auv/coug_fgo/actions/workflows/ros2_build_and_test.yml/badge.svg)](https://github.com/cougars-auv/coug_fgo/actions/workflows/ros2_build_and_test.yml)
[![Docker Build](https://github.com/cougars-auv/coug_fgo/actions/workflows/docker_build.yml/badge.svg)](https://github.com/cougars-auv/coug_fgo/actions/workflows/docker_build.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/cougars-auv/coug_fgo/main.svg)](https://results.pre-commit.ci/latest/github/cougars-auv/coug_fgo/main)
[![codecov](https://codecov.io/gh/cougars-auv/coug_fgo/graph/badge.svg?token=OFBQI4IOT1)](https://codecov.io/gh/cougars-auv/coug_fgo)

## 🤝 Contributing

- **Create a Branch:** Create a new branch using the format `name/feature` (e.g., `nelson/repo-docs`).

- **Make Changes:** Develop and debug your new feature. Add good documentation.

  > If you need to add dependencies, update the `package.xml`, the Dockerfiles under `.docker/`, `cougars.repos`, or `dependencies.repos` in your branch and test building the image locally.

- **Sync Frequently:** Regularly integrate the latest changes from `main` into your branch (via rebase or merge) to prevent future conflicts.

- **Submit a PR:** Open a pull request, ensure required tests pass, and merge once approved. Upon merge to `main`, GitHub Actions will automatically build and push updated images to Docker Hub with any new dependencies.

## 📦 Releasing

We adhere to the **Semantic Versioning (SemVer 2.0.0)** standard to release new versions of this repository:
> Given a version number **`MAJOR.MINOR.PATCH`**, increment the:
> - **MAJOR** version when you make incompatible API changes
> - **MINOR** version when you add functionality in a backward compatible manner
> - **PATCH** version when you make backward compatible bug fixes

- **Tag and Push:** Create and push a version tag (e.g., `v1.2.3`) on your release commit:

  ```bash
  git tag v1.2.3
  git push origin v1.2.3
  ```

  Pushing the tag automatically opens a draft GitHub Release with auto-generated notes.

- **Publish a GitHub Release:** Review the draft release in GitHub and click **Publish**.

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
