// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/result.h"

#include <google/protobuf/util/time_util.h>

#include <utility>

#include "base/util/compat.h"
#include "base/util/env.h"
#include "base/util/error.h"
#include "tile/hal/cm/err.h"
#include "tile/lang/semprinter.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

namespace {

namespace gpu = google::protobuf::util;

void LogActivity(const context::Context& ctx, std::shared_ptr<cmDeviceState> device_state, const cmResultInfo& info) {
  // auto queued_dur = gpu::TimeUtil::NanosecondsToDuration(info.queued_time);
  auto submit_dur = gpu::TimeUtil::NanosecondsToDuration(info.submit_time);
  auto start_dur = gpu::TimeUtil::NanosecondsToDuration(info.start_time);
  auto end_dur = gpu::TimeUtil::NanosecondsToDuration(info.end_time);
  // device_state->clock().LogActivity(ctx, "tile::hal::cm::HostQueue",
  // queued_dur, submit_dur);
  device_state->clock().LogActivity(ctx, "tile::hal::cm::DevQueue", submit_dur, start_dur);
  device_state->clock().LogActivity(ctx, "tile::hal::cm::Executing", start_dur, end_dur);
}

std::unique_ptr<cmResultInfo> MakeResultInfo(CmEvent* event) {
  if (!event) {
    // std::cout << "(DBG) MakeResultInfo No associated event information" <<
    // std::endl;
    return std::make_unique<cmResultInfo>();
  }

  auto info = std::make_unique<cmResultInfo>();

  void* pValue;
  cm_result_check(
      event->GetProfilingInfo(CM_EVENT_PROFILING_SUBMIT, sizeof(info->submit_time), &pValue, &info->submit_time));
  cm_result_check(
      event->GetProfilingInfo(CM_EVENT_PROFILING_HWSTART, sizeof(info->start_time), &pValue, &info->start_time));
  cm_result_check(event->GetProfilingInfo(CM_EVENT_PROFILING_HWEND, sizeof(info->end_time), &pValue, &info->end_time));

  info->execution_duration = std::chrono::nanoseconds(info->end_time - info->start_time);
  // std::cout << "(DBG TIME) end_time=" << info->end_time << "  start_time=" <<
  // info->start_time << std::endl;

  UINT64 execution_time = 0;
  cm_result_check(event->GetExecutionTime(execution_time));
  // std::cout << "(DBG TIME) Kernel linear execution time is " <<
  // execution_time << " nanoseconds" << std::endl;

  return info;
}

}  // namespace

cmResult::cmResult(const context::Context& ctx, std::shared_ptr<cmDeviceState> device_state, CmEvent* event)
    : ctx_{ctx}, device_state_{std::move(device_state)}, event_{std::move(event)} {}

std::chrono::high_resolution_clock::duration cmResult::GetDuration() const {
  std::call_once(once_, [this]() { info_ = MakeResultInfo(event_); });
  return info_->execution_duration;
}

void cmResult::LogStatistics() const {
  std::call_once(once_, [this]() { info_ = MakeResultInfo(event_); });
  if (info_->status < 0) {
    LOG(ERROR) << "Event " << event_ << " failed with: ";
  } else {
    auto duration = info_->execution_duration.count();
    VLOG(2) << "Result: dur=" << duration;
    LogActivity(ctx_, device_state_, *info_);
  }
}

cmKernelResult::cmKernelResult(const context::Context& ctx, std::shared_ptr<cmDeviceState> device_state, CmEvent* event,
                               const lang::KernelInfo& ki)
    : ctx_{ctx}, device_state_{std::move(device_state)}, event_{std::move(event)}, ki_(ki) {}

std::chrono::high_resolution_clock::duration cmKernelResult::GetDuration() const {
  std::call_once(once_, [this]() { info_ = MakeResultInfo(event_); });
  return info_->execution_duration;
}

void cmKernelResult::LogStatistics() const {
  std::call_once(once_, [this]() { info_ = MakeResultInfo(event_); });
  if (info_->status < 0) {
    LOG(ERROR) << "Kernel " << ki_.kname << " failed with: ";

    sem::Print emit_debug(*ki_.kfunc);
    LOG(ERROR) << "Generic debug kernel:";
    LOG(ERROR) << ki_.comments;
    LOG(ERROR) << emit_debug.str();

  } else {
    auto duration = info_->execution_duration.count();
    if (duration == 0) {
      // Prevent division by 0
      duration = 1;
    }
    if (env::Get("PLAIDML_DUMP_TIMES") == "1") {
      std::string rcom = ki_.comments;
      if (rcom.size() > 2 && rcom[0] == '/' && rcom[1] == '/') {
        rcom = rcom.substr(2, rcom.size() - 2);
      }
      if (rcom.size() > 1 && rcom[rcom.size() - 1] == '\n') {
        rcom = rcom.substr(0, rcom.size() - 1);
      }
      for (size_t i = 0; i < rcom.size(); i++) {
        if (rcom[i] == '\n') rcom[i] = '\t';
      }
      std::cout << duration << "\t" << ki_.kname << "\t" << rcom << "\n";
    }
    VLOG(3) << ki_.comments;
    // VLOG(1) << "Ran " << ki_.kname << ": dur=" << duration << " GFL/s=" <<
    // ki_.tot_flops / duration
    // << " GBP/s=" << ki_.tot_bytes / duration;
    LogActivity(ctx_, device_state_, *info_);
  }
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
