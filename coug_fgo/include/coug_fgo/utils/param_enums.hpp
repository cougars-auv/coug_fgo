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
 * @file param_enums.hpp
 * @brief Strongly-typed enums for string-valued node parameters.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <stdexcept>
#include <string>

namespace coug_fgo::utils {

enum class SolverType { kIncrementalFixedLagSmoother, kIsam2, kLevenbergMarquardt };

enum class RobustKernel { kNone, kHuber, kTukey };

enum class KeyframeSource { kNone, kDvl, kDepth, kTimer };

/**
 * @brief Parses a string into a SolverType enum.
 * @param s The string to parse.
 * @return The parsed SolverType.
 * @throws std::invalid_argument If the string does not match a valid solver type.
 */
inline SolverType parseSolverType(const std::string& s) {
  if (s == "IncrementalFixedLagSmoother") return SolverType::kIncrementalFixedLagSmoother;
  if (s == "ISAM2") return SolverType::kIsam2;
  if (s == "LevenbergMarquardt") return SolverType::kLevenbergMarquardt;
  throw std::invalid_argument("Unknown solver_type: " + s);
}

/**
 * @brief Parses a string into a RobustKernel enum.
 * @param s The string to parse.
 * @return The parsed RobustKernel.
 * @throws std::invalid_argument If the string does not match a valid robust kernel type.
 */
inline RobustKernel parseRobustKernel(const std::string& s) {
  if (s == "None") return RobustKernel::kNone;
  if (s == "Huber") return RobustKernel::kHuber;
  if (s == "Tukey") return RobustKernel::kTukey;
  throw std::invalid_argument("Unknown robust_kernel: " + s);
}

/**
 * @brief Parses a string into a KeyframeSource enum.
 * @param s The string to parse.
 * @return The parsed KeyframeSource.
 * @throws std::invalid_argument If the string does not match a valid keyframe source.
 */
inline KeyframeSource parseKeyframeSource(const std::string& s) {
  if (s == "None") return KeyframeSource::kNone;
  if (s == "DVL") return KeyframeSource::kDvl;
  if (s == "Depth") return KeyframeSource::kDepth;
  if (s == "Timer") return KeyframeSource::kTimer;
  throw std::invalid_argument("Unknown keyframe source: " + s);
}

}  // namespace coug_fgo::utils
