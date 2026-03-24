// Copyright (c) 2026 BYU FROST Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file gtsam_conversions.hpp
 * @brief Utility functions for converting std containers to GTSAM math types.
 * @author Nelson Durrant
 * @date Jan 2026
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

#include <array>
#include <vector>

namespace coug_fgo::utils {

/**
 * @brief Converts a 9-element covariance array (row-major) to a GTSAM Matrix33.
 * @param cov The input 3x3 covariance array.
 * @return The resulting gtsam::Matrix33.
 */
inline gtsam::Matrix33 toGtsam(const std::array<double, 9>& cov) {
  gtsam::Matrix33 m;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m(i, j) = cov[i * 3 + j];
    }
  }
  return m;
}

/**
 * @brief Converts a 36-element covariance array (row-major) to a GTSAM Matrix66.
 * @param cov The input 6x6 covariance array.
 * @return The resulting gtsam::Matrix66.
 */
inline gtsam::Matrix66 toGtsam(const std::array<double, 36>& cov) {
  gtsam::Matrix66 m;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      m(i, j) = cov[i * 6 + j];
    }
  }
  return m;
}

/**
 * @brief Converts a std::vector to a GTSAM Vector (dynamic size).
 * @param v The input vector.
 * @return The resulting gtsam::Vector.
 */
inline gtsam::Vector toGtsam(const std::vector<double>& v) {
  gtsam::Vector gtsam_v(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    gtsam_v(i) = v[i];
  }
  return gtsam_v;
}

/**
 * @brief Converts a vector of sigmas to a squared diagonal covariance matrix (dynamic size).
 * @param sigmas The input vector of standard deviations.
 * @return The resulting gtsam::Matrix (diagonal with sigmas squared).
 */
inline gtsam::Matrix toGtsamSquaredDiagonal(const std::vector<double>& sigmas) {
  gtsam::Vector squared(sigmas.size());
  for (size_t i = 0; i < sigmas.size(); ++i) {
    squared(i) = sigmas[i] * sigmas[i];
  }
  return squared.asDiagonal();
}

/**
 * @brief Converts a vector of values to a diagonal matrix (dynamic size).
 * @param diag_elements The input vector of diagonal elements.
 * @return The resulting gtsam::Matrix.
 */
inline gtsam::Matrix toGtsamDiagonal(const std::vector<double>& diag_elements) {
  gtsam::Vector v(diag_elements.size());
  for (size_t i = 0; i < diag_elements.size(); ++i) {
    v(i) = diag_elements[i];
  }
  return v.asDiagonal();
}

}  // namespace coug_fgo::utils
