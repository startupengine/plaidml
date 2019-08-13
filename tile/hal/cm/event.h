// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/cm/buffer.h"
#include "tile/hal/cm/device_state.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

// Implements hal::Event in terms of cm events.
class cmEvent final : public hal::Event {
 public:
  // Casts a hal::Event to an Event, throwing an exception if the supplied
  // hal::Event isn't an
  // cm event, or if it's an event for a different context.
  static std::shared_ptr<cmEvent> Downcast(const std::shared_ptr<hal::Event>& event);

  // Casts a vector of hal::Event to a vector of cl_event, throwing an exception
  // if the supplied
  // hal::Event objects aren't cm events, or if any are events for a different
  // context. This
  // method does not retain the returned events; callers must hold onto the
  // input vector to keep the
  // events alive.
  static std::vector<CmEvent*> Downcast(const std::vector<std::shared_ptr<hal::Event>>& events,
                                        const std::unique_ptr<cmDeviceState::cmQueueStruct>& queue);

  // Returns a future that waits for all of the supplied events to complete.
  static boost::future<std::vector<std::shared_ptr<hal::Result>>> WaitFor(
      const std::vector<std::shared_ptr<hal::Event>>& events, std::shared_ptr<cmDeviceState> device_state);

  cmEvent(const context::Context& ctx, std::shared_ptr<cmDeviceState> device_state, CmEvent* cm_event,
          const std::unique_ptr<cmDeviceState::cmQueueStruct>& queue);

  cmEvent(const context::Context& ctx, std::shared_ptr<cmDeviceState> device_state, CmEvent* cm_event,
          const std::unique_ptr<cmDeviceState::cmQueueStruct>& queue, const std::shared_ptr<hal::Result>& result);

  ~cmEvent() final;

  boost::shared_future<std::shared_ptr<hal::Result>> GetFuture() final;

 private:
  struct cmFutureState {
    std::mutex mu;
    bool completed = false;
    std::shared_ptr<cmFutureState> self;  // Set iff clSetEventCallback is in flight
    std::shared_ptr<hal::Result> result;
    boost::promise<std::shared_ptr<hal::Result>> prom;
  };

  static void EventComplete(CmEvent* evt, int32_t status, void* data);

  context::Context ctx_;
  const std::unique_ptr<cmDeviceState::cmQueueStruct>& queue_;
  std::mutex mu_;
  bool started_ = false;
  CmEvent* cm_event_;
  std::shared_ptr<cmFutureState> state_;
  boost::shared_future<std::shared_ptr<hal::Result>> fut_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
