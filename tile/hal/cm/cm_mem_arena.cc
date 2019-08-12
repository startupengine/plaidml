// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/cm_mem_arena.h"

#include <utility>

#include "base/util/error.h"
#include "tile/hal/cm/cm_mem_buffer.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

CMMemArena::CMMemArena(std::shared_ptr<cmDeviceState> device_state, std::uint64_t size)
    : device_state_{device_state}, size_{size} {}

std::shared_ptr<hal::Buffer> CMMemArena::MakeBuffer(std::uint64_t offset, std::uint64_t size) {
  if (size_ < offset || size_ < size || size_ < (offset + size)) {
    throw error::OutOfRange{"Requesting memory outside arena bounds"};
  }

  void* base = CM_ALIGNED_MALLOC(size, 0x1000);
  memset(base, 0, size);

  CmBufferUP* pCmBuffer;
  cm_result_check(device_state_->cmdev()->CreateBufferUP(size, base, pCmBuffer));

  return std::make_shared<CMMemBuffer>(device_state_, size, pCmBuffer, base);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
