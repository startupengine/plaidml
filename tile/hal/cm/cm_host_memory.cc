// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/cm_host_memory.h"

#include <utility>

#include "tile/hal/cm/cm_mem_arena.h"
#include "tile/hal/cm/cm_mem_buffer.h"
#include "tile/hal/cm/cm_runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

cmHostMemory::cmHostMemory(std::shared_ptr<cmDeviceState> device_state) : device_state_{std::move(device_state)} {}

cmHostMemory::~cmHostMemory() {
  // std::cout << "(DBG) cmHostMemory::~cmHostMemory()" << std::endl;
}

std::shared_ptr<hal::Buffer> cmHostMemory::MakeBuffer(std::uint64_t size, BufferAccessMask /* access */) {
  uint64_t buf_alignment_overflow_size = 3 * sizeof(float);
  size = (size >= 16) ? size : 16;
  size += buf_alignment_overflow_size;
  void* void_buf_ = CM_ALIGNED_MALLOC(size, 0x1000);

  memset(void_buf_, 0, size);

  CmBufferUP* pCmBuffer;
  cm_result_check(device_state_->cmdev()->CreateBufferUP(size, void_buf_, pCmBuffer));
  // std::cout << "(DBG) cmHostMemory::MakeBuffer CreateBufferUP + new
  // CMMemBuffer pCmBuffer=" << pCmBuffer <<
  // std::endl;
  return std::make_shared<CMMemBuffer>(device_state_, size, pCmBuffer, void_buf_);
}

std::shared_ptr<hal::Arena> cmHostMemory::MakeArena(std::uint64_t size, BufferAccessMask /* access */) {
  return std::make_shared<CMMemArena>(device_state_, size);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
