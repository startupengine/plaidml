#pragma once

#include <vector>

#include "tile/base/hal.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

void OptimizeKernel(const lang::KernelInfo& ki, bool cl_khr_fp16, const hal::proto::HardwareSettings& settings);

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
