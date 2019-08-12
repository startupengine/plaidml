// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/cm_compute_kernel.h"

#include <utility>

#include "base/util/env.h"
#include "base/util/error.h"
#include "base/util/uuid.h"
#include "tile/hal/cm/cm_buffer.h"
#include "tile/hal/cm/cm_err.h"
#include "tile/hal/cm/cm_event.h"
#include "tile/hal/cm/cm_mem_buffer.h"
#include "tile/hal/cm/cm_result.h"
#include "tile/hal/cm/cm_runtime.h"
#include "tile/hal/cm/emitcm.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

int cmComputeKernel::count = 0;

void write_buffer_file(int byte_size, void* tmp, std::string filename) {
  float* buffer = reinterpret_cast<float*>(tmp);
  FILE* stream = NULL;
  int float_size = byte_size / sizeof(float);
  stream = fopen(filename.c_str(), "w");
  for (int i = 0; i < float_size; i++) {
    fprintf(stream, "%.6g\n", *buffer);
    buffer++;
  }
  fclose(stream);
}

cmComputeKernel::cmComputeKernel(std::shared_ptr<cmDeviceState> device_state, CmKernel* kernel,
                                 const lang::KernelInfo& info, context::proto::ActivityID kernel_id,
                                 const std::shared_ptr<Emit>& cm)
    : device_state_{device_state}, kernel_{std::move(kernel)}, ki_(info), kernel_id_(kernel_id), cm_{cm} {
  pKernelArray_ = nullptr;
  pts_ = nullptr;
}

cmComputeKernel::~cmComputeKernel() {
  auto pCmDev = device_state_->cmdev();
  cm_result_check(pCmDev->DestroyTask(pKernelArray_));
  cm_result_check(pCmDev->DestroyKernel(kernel_));
  cm_result_check(pCmDev->DestroyThreadGroupSpace(pts_));
}

unsigned int max_divisor(unsigned int a, unsigned int uplimit) {
  while (a % uplimit != 0 && uplimit > 0) {
    uplimit--;
  }
  return uplimit;
}

std::shared_ptr<hal::Event> cmComputeKernel::Run(const context::Context& ctx,
                                                 const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                                 const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                                 bool enable_profiling) {
  const auto& queue = device_state_->cm_queue_;
  std::lock_guard<std::mutex> lock{mu_};

  auto pCmQueue = device_state_->cm_queue_->pCmQueue_;

  for (std::size_t i = 0; i < params.size(); ++i) {
    cmBuffer* buf = cmBuffer::Downcast(params[i].get());
    VLOG(4) << "  Param " << i << ": " << buf << " size=" << buf->size();
    CMMemBuffer* membuf = dynamic_cast<CMMemBuffer*>(buf);
    if (i == 0) {
      membuf->clean_base_();
    }
    buf->SetKernelArg(kernel_, i);
  }

  context::Activity activity{ctx, "tile::hal::cm::Kernel::Run"};
  if (ctx.is_logging_events()) {
    proto::RunInfo rinfo;
    *rinfo.mutable_kernel_id() = kernel_id_;
    activity.AddMetadata(rinfo);
  }

  auto pCmDev = device_state_->cmdev();
  cm_result_check(pCmDev->CreateTask(pKernelArray_));
  cm_result_check(pKernelArray_->AddKernel(kernel_));

  unsigned int g[3], nthreads[3];

  size_t v_size = 16;
  if (cm_->vector_size == "8") {
    v_size = 8;
  }

  if (cm_->vector_size == "4") {
    v_size = 4;
  }

  const unsigned int threads_num = 48;

  if (cm_->use_global_id) {
    int max_threads_num = threads_num;
    for (int i = 0; i < 3; i++) {
      nthreads[i] = ki_.gwork[i];

      g[i] = max_divisor(nthreads[i], max_threads_num);
      max_threads_num /= g[i];
    }
  } else {
    for (int i = 0; i < 3; i++) {
      nthreads[i] = (ki_.gwork[i] % v_size == 0) ? ki_.gwork[i] / v_size : ki_.gwork[i];
      if (ki_.lwork[i]) {
        g[i] = (ki_.lwork[i] % v_size == 0) ? ki_.lwork[i] / v_size : ki_.lwork[i];
      } else {
        g[i] = max_divisor(nthreads[i], threads_num);
      }
    }
  }

  cm_result_check(pCmDev->CreateThreadGroupSpaceEx(g[0], g[1], g[2], nthreads[0] / g[0], nthreads[1] / g[1],
                                                   nthreads[2] / g[2], pts_));

  CmEvent* done = NULL;
  cm_result_check(pCmQueue->EnqueueWithGroup(pKernelArray_, done, pts_));

  CM_STATUS s;
  for (done->GetStatus(s); s != CM_STATUS_FINISHED; done->GetStatus(s)) {
  }

  for (std::size_t i = 0; i < params.size(); ++i) {
    cmBuffer* buf = cmBuffer::Downcast(params[i].get());
    buf->ReleaseDeviceBuffer();
  }

  auto result = std::make_shared<cmKernelResult>(activity.ctx(), device_state_, done, ki_);
  return std::make_shared<cmEvent>(activity.ctx(), device_state_, std::move(done), queue, std::move(result));
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
