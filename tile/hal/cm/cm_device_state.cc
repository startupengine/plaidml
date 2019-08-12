// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/cm_device_state.h"

#include <string>
#include <utility>
#include <vector>

#include "base/util/error.h"
#include "tile/hal/cm/cm_err.h"
#include "tile/hal/cm/cm_runtime.h"
#include "tile/hal/util/selector.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {
namespace {

cmDeviceState::cmQueueStruct MakeQueue(CmDevice* pCmDev) {
  cmDeviceState::cmQueueStruct result;

  // Create a task queue
  pCmDev->InitPrintBuffer();
  CmQueue* pCmQueue = NULL;
  pCmDev->CreateQueue(pCmQueue);

  result.pCmQueue_ = pCmQueue;
  return result;
}

}  // namespace

void cmDeviceState::Flush() const { cm_result_check(pCmDev_->FlushPrintBuffer()); }

cmDeviceState::cmDeviceState(const context::Context& ctx, CmDevice* pCmDev, proto::DeviceInfo dinfo)
    : cm_queue_{std::unique_ptr<cmQueueStruct>()},
      pCmDev_{pCmDev},
      info_{std::move(dinfo)},
      clock_{},
      id_{ctx.activity_id()} {}

cmDeviceState::~cmDeviceState() { cm_result_check(::DestroyCmDevice(pCmDev_)); }
void cmDeviceState::Initialize() { cm_queue_ = std::make_unique<cmQueueStruct>(MakeQueue(pCmDev_)); }

void cmDeviceState::FlushCommandQueue() { Flush(); }

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
