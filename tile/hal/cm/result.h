// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "tile/base/hal.h"
#include "tile/hal/cm/device_state.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

struct cmResultInfo {
  ulong queued_time;
  ulong submit_time;
  ulong start_time;
  ulong end_time;
  int status;
  std::chrono::high_resolution_clock::duration execution_duration;
};

// Implements hal::Result in terms of cm events.
class cmResult final : public hal::Result {
 public:
  cmResult(const context::Context& ctx, std::shared_ptr<cmDeviceState> device_state, CmEvent* event);

  std::chrono::high_resolution_clock::duration GetDuration() const final;
  void LogStatistics() const final;

 private:
  context::Context ctx_;
  std::shared_ptr<cmDeviceState> device_state_;
  CmEvent* event_;
  mutable std::unique_ptr<cmResultInfo> info_;
  mutable std::once_flag once_;
};

class cmKernelResult final : public hal::Result {
 public:
  cmKernelResult(const context::Context& ctx, std::shared_ptr<cmDeviceState> device_state, CmEvent* event,
                 const lang::KernelInfo& ki);

  std::chrono::high_resolution_clock::duration GetDuration() const final;
  void LogStatistics() const final;

 private:
  context::Context ctx_;
  std::shared_ptr<cmDeviceState> device_state_;
  CmEvent* event_;
  mutable std::unique_ptr<cmResultInfo> info_;
  mutable std::once_flag once_;

  lang::KernelInfo ki_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
