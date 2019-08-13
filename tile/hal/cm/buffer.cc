// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/buffer.h"

#include <memory>

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

std::shared_ptr<cmBuffer> cmBuffer::Downcast(const std::shared_ptr<hal::Buffer>& buffer) {
  std::shared_ptr<cmBuffer> buf = std::dynamic_pointer_cast<cmBuffer>(buffer);
  return buf;
}

cmBuffer* cmBuffer::Downcast(hal::Buffer* buffer) {
  cmBuffer* buf = dynamic_cast<cmBuffer*>(buffer);
  return buf;
}

cmBuffer::cmBuffer(std::uint64_t size) : size_{size} {}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
