// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/cm/cm_buffer.h"
#include "tile/hal/cm/cm_device_state.h"
#include "tile/hal/cm/cm_runtime.h"
#include "tile/hal/cm/emitcm.h"
#include "tile/hal/cm/kernel.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class cmComputeKernel final : public Kernel {
 public:
  cmComputeKernel(std::shared_ptr<cmDeviceState> device_state, CmKernel* kernel, const lang::KernelInfo& info,
                  context::proto::ActivityID kernel_id, const std::shared_ptr<Emit>& cm);

  ~cmComputeKernel();

  std::shared_ptr<hal::Event> Run(const context::Context& ctx, const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

  static int count;

 private:
  std::mutex mu_;
  std::shared_ptr<cmDeviceState> device_state_;
  CmKernel* kernel_;
  CmTask* pKernelArray_;
  CmThreadGroupSpace* pts_;
  lang::KernelInfo ki_;
  context::proto::ActivityID kernel_id_;
  std::shared_ptr<Emit> cm_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
