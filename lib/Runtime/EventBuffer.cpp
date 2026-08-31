// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Runtime/EventBuffer.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <utility>
namespace tilemega::runtime {
EventBuffer::EventBuffer(std::size_t count) : size_(count) {
  if (count && cudaMalloc(reinterpret_cast<void**>(&data_),
                          count * sizeof(EventSlot)) != cudaSuccess)
    throw std::runtime_error("cudaMalloc EventBuffer failed");
  if (count) cudaMemset(data_, 0, count * sizeof(EventSlot));
}
EventBuffer::~EventBuffer() { if (data_) cudaFree(data_); }
EventBuffer::EventBuffer(EventBuffer&& other) noexcept { *this = std::move(other); }
EventBuffer& EventBuffer::operator=(EventBuffer&& other) noexcept {
  if (this != &other) { if (data_) cudaFree(data_); data_ = other.data_; size_ = other.size_; other.data_ = nullptr; other.size_ = 0; }
  return *this;
}
}  // namespace tilemega::runtime
