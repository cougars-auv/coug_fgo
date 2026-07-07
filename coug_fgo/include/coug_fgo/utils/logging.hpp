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
 * @file logging.hpp
 * @brief Log sink types for the ROS-independent core.
 * @author Nelson Durrant
 * @date July 2026
 */

#pragma once

#include <functional>
#include <string>

namespace coug_fgo::utils {

/**
 * @enum LogLevel
 * @brief Severity levels for core log messages.
 */
enum class LogLevel { kDebug, kInfo, kWarn, kError };

/**
 * @brief Log message sink, injected by the ROS node or the Python bindings.
 */
using LogCallback = std::function<void(LogLevel, const std::string&)>;

}  // namespace coug_fgo::utils
