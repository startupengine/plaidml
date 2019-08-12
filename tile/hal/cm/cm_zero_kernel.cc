// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/cm_zero_kernel.h"

#include "base/util/error.h"
#include "tile/hal/cm/cm_event.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

cmZeroKernel::cmZeroKernel(const std::shared_ptr<cmDeviceState>& device_state, const lang::KernelInfo& kinfo,
                           context::proto::ActivityID kid)
    : device_state_{device_state}, kinfo_{kinfo}, kid_(kid) {}

std::shared_ptr<hal::Event> cmZeroKernel::Run(const context::Context& ctx,
                                              const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                              const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                              bool enable_profiling) {
  const auto& queue = device_state_->cm_queue_;
  auto deps = cmEvent::Downcast(dependencies, queue);
  IVLOG(4, "Running zero-fill memory " << kinfo_.kname);

  if (params.size() != 1) {
    throw error::Internal("Zero-memory operation invoked with a memory region count != 1");
  }

  cmBuffer* buf = cmBuffer::Downcast(params[0].get());
  IVLOG(4, "  Buffer: " << buf);

  context::Activity activity{ctx, "tile::hal::cm::Buffer::Fill"};
  proto::RunInfo rinfo;
  *rinfo.mutable_kernel_id() = kid_;
  activity.AddMetadata(rinfo);

  CmEvent* done;
  return std::make_shared<cmEvent>(activity.ctx(), device_state_, done, queue);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
