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
 * @file thread_safe_queue.hpp
 * @brief Utility for thread-safe queue operations.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <chrono>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace coug_fgo::utils {

/**
 * @class ThreadSafeQueue
 * @brief Utility for thread-safe queue operations.
 * @tparam T Pointer-like message type exposing a `timestamp` member (e.g. a shared_ptr).
 */
template <typename T>
class ThreadSafeQueue {
 public:
  /**
   * @brief Pushes a new item onto the queue.
   * @param value The item to push.
   */
  void push(const T& value) {
    std::scoped_lock lock(mutex_);
    queue_.push_back(value);
    last_msg_time_ = value->timestamp;
    last_arrival_ = std::chrono::steady_clock::now();
  }

  /**
   * @brief Drains all items from the queue, leaving it empty.
   * @return A deque containing all items that were in the queue.
   */
  std::deque<T> drain() {
    std::scoped_lock lock(mutex_);
    return std::exchange(queue_, {});
  }

  /**
   * @brief Checks if the queue is empty.
   * @return True if the queue is empty, false otherwise.
   */
  bool empty() const {
    std::scoped_lock lock(mutex_);
    return queue_.empty();
  }

  /**
   * @brief Gets the number of items in the queue.
   * @return The size of the queue.
   */
  size_t size() const {
    std::scoped_lock lock(mutex_);
    return queue_.size();
  }

  /**
   * @brief Gets the timestamp of the last item added to the queue (persists across drains).
   * @return The timestamp as a double (seconds), or nullopt if no item has ever been pushed.
   */
  std::optional<double> getLastTime() const {
    std::scoped_lock lock(mutex_);
    return last_msg_time_;
  }

  /**
   * @brief Gets the wall-clock time since the last item arrived.
   * @return Seconds since the last arrival, or nullopt if no item has ever been pushed.
   */
  std::optional<double> secondsSinceLastArrival() const {
    std::scoped_lock lock(mutex_);
    if (!last_arrival_.has_value()) {
      return std::nullopt;
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - *last_arrival_).count();
  }

  /**
   * @brief Restores items to the front of the queue.
   * @param items The items to restore.
   */
  void restore(const std::deque<T>& items) {
    std::scoped_lock lock(mutex_);
    queue_.insert(queue_.begin(), items.begin(), items.end());
  }

 private:
  mutable std::mutex mutex_;
  std::deque<T> queue_;
  std::optional<double> last_msg_time_;
  std::optional<std::chrono::steady_clock::time_point> last_arrival_;
};

}  // namespace coug_fgo::utils
