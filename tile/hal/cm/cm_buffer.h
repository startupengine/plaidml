// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/cm/cm_runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

// Represents a chunk of cm memory.
class cmBuffer : public hal::Buffer {
 public:
  // Casts a hal::Buffer to a Buffer, throwing an exception if the supplied
  // hal::Buffer isn't an cm buffer, or if
  // it's a buffer for a different context.
  static std::shared_ptr<cmBuffer> Downcast(const std::shared_ptr<hal::Buffer>& buffer);
  static cmBuffer* Downcast(hal::Buffer* buffer);

  virtual void SetKernelArg(CmKernel* kernel, std::size_t index) = 0;

  virtual void* base() const { return nullptr; }
  virtual CmBufferUP* mem() const { return nullptr; }

  virtual void ReleaseDeviceBuffer() {}
  std::uint64_t size() const { return size_; }

 protected:
  explicit cmBuffer(std::uint64_t size);

 private:
  const std::uint64_t size_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
