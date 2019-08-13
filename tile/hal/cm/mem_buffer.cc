// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/mem_buffer.h"

#include <execinfo.h>
#include <utility>

#include "tile/hal/cm/err.h"
#include "tile/hal/cm/event.h"
#include "tile/hal/cm/result.h"
#include "tile/hal/cm/runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

CMMemBuffer::CMMemBuffer(std::shared_ptr<cmDeviceState> device_state, std::uint64_t size, CmBufferUP* pCmBuffer,
                         void* base)
    : cmBuffer{size}, device_state_{device_state}, pCmBuffer_{pCmBuffer}, base_{base} {}

CMMemBuffer::~CMMemBuffer() {
  // std::cout << "(DBG) CMMemBuffer::~CMMemBuffer() pCmBuffer_=" << pCmBuffer_
  // << std::endl;
  if (pCmBuffer_) {
    cm_result_check(device_state_->cmdev()->DestroyBufferUP(pCmBuffer_));
  }
}

void CMMemBuffer::ReleaseDeviceBuffer() {
  if (pCmBuffer_) {
    cm_result_check(device_state_->cmdev()->DestroyBufferUP(pCmBuffer_));
  }
}

void CMMemBuffer::SetKernelArg(CmKernel* kernel, std::size_t index) {
  if (pCmBuffer_ == nullptr) {
    // std::cout << "(DBG) CMMemBuffer::SetKernelArg CreateBufferUP " <<
    // std::endl;
    cm_result_check(device_state_->cmdev()->CreateBufferUP(this->size(), base_, pCmBuffer_));
  }
  // std::cout << "(DBG) CMMemBuffer::SetKernelArg pCmBuffer_=" << pCmBuffer_ <<
  // std::endl;
  SurfaceIndex* BUFFER;
  pCmBuffer_->GetIndex(BUFFER);
  kernel->SetKernelArg(index, sizeof(SurfaceIndex), BUFFER);
}

boost::future<void*> CMMemBuffer::MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  // std::cout << "(DBG) CMMemBuffer::MapCurrent pCmBuffer_=" << pCmBuffer_ <<
  // std::endl;
  return boost::make_ready_future(base_);
}

boost::future<void*> CMMemBuffer::MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  // std::cout << "(DBG) CMMemBuffer::MapDiscard pCmBuffer_=" << pCmBuffer_ <<
  // std::endl;
  return boost::make_ready_future(base_);
}

std::shared_ptr<hal::Event> CMMemBuffer::Unmap(const context::Context& ctx) {
  // std::cout << "(DBG) CMMemBuffer::Unmap pCmBuffer_=" << pCmBuffer_ <<
  // std::endl;

  context::Activity activity{ctx, "tile::hal::cm::Buffer::Unmap"};
  CmEvent* e;
  return std::make_shared<cmEvent>(activity.ctx(), device_state_, std::move(e), device_state_->cm_queue_);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
