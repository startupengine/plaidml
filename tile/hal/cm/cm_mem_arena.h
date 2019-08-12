// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/hal/cm/cm_buffer.h"
#include "tile/hal/cm/cm_device_state.h"
#include "tile/hal/cm/cm_err.h"
#include "tile/hal/cm/cm_mem_buffer.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

// An Arena implemented using a cl_mem object.
class CMMemArena final : public hal::Arena {
 public:
  CMMemArena(std::shared_ptr<cmDeviceState> device_state, std::uint64_t size);

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t offset, std::uint64_t size) final;

 private:
  std::shared_ptr<cmDeviceState> device_state_;
  std::uint64_t size_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
