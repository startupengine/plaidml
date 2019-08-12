// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/cm_device_set.h"

#include <string>
#include <utility>

#include <boost/regex.hpp>

#include "base/util/compat.h"
#include "base/util/error.h"
#include "tile/hal/cm/cm_err.h"
#include "tile/hal/cm/cm_runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

cmDeviceSet::cmDeviceSet(const context::Context& ctx) {
  context::Activity platform_activity{ctx, "tile::hal::cm::Platform"};
  proto::PlatformInfo pinfo;
  pinfo.set_name("CM");

  context::Activity device_activity{platform_activity.ctx(), "tile::hal::cm::Device"};
  proto::DeviceInfo info;
  info.set_platform_name(pinfo.name());
  info.set_name("Intel GPU");
  info.set_mem_base_addr_align(0x1000);

  device_activity.AddMetadata(info);
  *info.mutable_platform_id() = platform_activity.ctx().activity_id();

  CmDevice* pCmDev = NULL;
  UINT version = 0;
  cm_result_check(::CreateCmDevice(pCmDev, version));
  if (version < CM_1_0) {
    throw std::runtime_error(std::string("The runtime API version is later than runtime DLL version "));
  }

  auto dev = std::make_shared<cmDevice>(device_activity.ctx(), pCmDev, std::move(info));

  std::shared_ptr<cmDevice> first_dev;
  first_dev = dev;

  devices_.emplace_back(std::move(dev));

  host_memory_ = std::make_unique<cmHostMemory>(first_dev->device_state());
}

const std::vector<std::shared_ptr<hal::Device>>& cmDeviceSet::devices() { return devices_; }

Memory* cmDeviceSet::host_memory() { return host_memory_.get(); }

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
