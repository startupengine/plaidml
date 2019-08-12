// Copyright 2017-2018 Intel Corporation.

#include "base/util/compat.h"
#include "base/util/env.h"
#include "base/util/factory.h"
#include "tile/hal/cm/cm_driver.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

[[gnu::unused]] char reg = []() -> char {
  FactoryRegistrar<hal::Driver>::Instance()->Register(
      "cm",                                                                         //
      [](const context::Context& ctx) { return std::make_unique<cmDriver>(ctx); },  //
#ifdef __APPLE__
      FactoryPriority::DEFAULT);
#else
      FactoryPriority::HIGH);
#endif
  return 0;
}();

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
