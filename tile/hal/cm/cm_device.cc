// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/cm_device.h"

#include <utility>

#include "base/util/compat.h"
#include "tile/hal/cm/cm_compiler.h"
#include "tile/hal/cm/cm_executor.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

cmDevice::cmDevice(const context::Context& ctx, CmDevice* pCmDev, proto::DeviceInfo dinfo)
    : device_state_{std::make_shared<cmDeviceState>(ctx, pCmDev, std::move(dinfo))},
      compiler_{std::make_unique<cmCompiler>(device_state_)},
      executor_{std::make_unique<cmExecutor>(device_state_)} {}

void cmDevice::Initialize(const hal::proto::HardwareSettings& settings) { device_state_->Initialize(); }

std::string cmDevice::description() {  //
  return device_state()->info().vendor() + " " + device_state()->info().name() + " (CM)";
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
