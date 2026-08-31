// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 invariant I2 and §8.4–§8.5 event storage.
#pragma once
#include <cstddef>
#include <cstdint>
namespace tilemega::runtime {
struct alignas(16) EventSlot { std::uint64_t value = 0; std::uint64_t epoch = 0; };
class EventBuffer {
 public:
  EventBuffer() = default;
  explicit EventBuffer(std::size_t count);
  ~EventBuffer();
  EventBuffer(EventBuffer const&) = delete;
  EventBuffer& operator=(EventBuffer const&) = delete;
  EventBuffer(EventBuffer&&) noexcept;
  EventBuffer& operator=(EventBuffer&&) noexcept;
  EventSlot* data() const { return data_; }
  std::size_t size() const { return size_; }
 private:
  EventSlot* data_ = nullptr;
  std::size_t size_ = 0;
};
}  // namespace tilemega::runtime
