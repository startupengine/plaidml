// Copyright 2017-2018 Intel Corporation.

#include <cstdint>
#include <mutex>
#include <vector>

#include "base/util/compat.h"
#include "base/util/error.h"
#include "base/util/logging.h"
#include "tile/base/hal.h"
#include "tile/hal/cm/buffer.h"
#include "tile/hal/cm/device_state.h"
#include "tile/hal/cm/event.h"
#include "tile/hal/cm/executor.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

namespace {

// Class declarations

class cmSharedArena final : public Arena, public std::enable_shared_from_this<cmSharedArena> {
 public:
  cmSharedArena(const std::shared_ptr<cmDeviceState>& device_state, std::uint64_t size);
  virtual ~cmSharedArena();

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t offset, std::uint64_t size) final;

  std::shared_ptr<cmDeviceState> device_state() const { return device_state_; }

 private:
  // A lock to guard clSVMAlloc/clSVMFree calls.  This shouldn't be necessary,
  // but
  // it turns out we see crashes without it.
  static std::mutex svm_mu;

  std::shared_ptr<cmDeviceState> device_state_;
  void* base_ = nullptr;
  std::uint64_t size_;
};

class cmSharedBuffer final : public cmBuffer {
 public:
  cmSharedBuffer(std::shared_ptr<cmSharedArena> arena, void* base, std::uint64_t size);

  void SetKernelArg(CmKernel* kernel, std::size_t index) final;

  boost::future<void*> MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  boost::future<void*> MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  std::shared_ptr<hal::Event> Unmap(const context::Context& ctx) final;

  void* base() const final { return base_; }

 private:
  std::shared_ptr<cmSharedArena> arena_;
  void* base_ = nullptr;
};

class cmSharedMemory final : public Memory {
 public:
  explicit cmSharedMemory(const std::shared_ptr<cmDeviceState>& device_state);

  std::uint64_t size_goal() const final {
    // TODO: Actually query the system physical memory size.
    return 16 * std::giga::num;
  }

  BufferAccessMask AllowedAccesses() const final { return BufferAccessMask::ALL; }

  std::size_t ArenaBufferAlignment() const final { return device_state_->info().mem_base_addr_align(); }

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t size, BufferAccessMask access) final;

  std::shared_ptr<hal::Arena> MakeArena(std::uint64_t size, BufferAccessMask access) final;

 private:
  std::shared_ptr<cmDeviceState> device_state_;
};

// SharedArena implementation

std::mutex cmSharedArena::svm_mu;

cmSharedArena::cmSharedArena(const std::shared_ptr<cmDeviceState>& device_state, std::uint64_t size)
    : device_state_{device_state}, size_{size} {}

cmSharedArena::~cmSharedArena() {}

std::shared_ptr<hal::Buffer> cmSharedArena::MakeBuffer(std::uint64_t offset, std::uint64_t size) {
  if (size_ < offset || size_ < size || size_ < (offset + size)) {
    throw error::OutOfRange{"Requesting memory outside arena bounds"};
  }
  return std::make_shared<cmSharedBuffer>(shared_from_this(), static_cast<char*>(base_) + offset, size);
}

// SharedBuffer implementation

cmSharedBuffer::cmSharedBuffer(std::shared_ptr<cmSharedArena> arena, void* base, std::uint64_t size)
    : cmBuffer{size}, arena_{std::move(arena)}, base_{base} {}

void cmSharedBuffer::SetKernelArg(CmKernel* kernel, std::size_t index) {}

boost::future<void*> cmSharedBuffer::MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  VLOG(4) << "OCL SharedBuffer MapCurrent: waiting this: " << this;
  return cmEvent::WaitFor(deps, arena_->device_state()).then([
    this, base = base_
  ](boost::shared_future<std::vector<std::shared_ptr<hal::Result>>> f) {
    f.get();
    return base;
  });
}

boost::future<void*> cmSharedBuffer::MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  // We need to wait for the dependencies to resolve; once that's happened, we
  // might as
  // well map the current memory.
  return MapCurrent(deps);
}

std::shared_ptr<hal::Event> cmSharedBuffer::Unmap(const context::Context& ctx) {
  CmEvent* cm_event;
  return std::make_shared<cmEvent>(ctx, arena_->device_state(), cm_event, arena_->device_state()->cm_queue_);
}

// SharedMemory implementation

cmSharedMemory::cmSharedMemory(const std::shared_ptr<cmDeviceState>& device_state) : device_state_{device_state} {}

std::shared_ptr<hal::Buffer> cmSharedMemory::MakeBuffer(std::uint64_t size, BufferAccessMask access) {
  return MakeArena(size, access)->MakeBuffer(0, size);
}

std::shared_ptr<hal::Arena> cmSharedMemory::MakeArena(std::uint64_t size, BufferAccessMask /* access */) {
  return std::make_shared<cmSharedArena>(device_state_, size);
}

}  // namespace

// Implements Executor::InitSharedMemory on systems that support the
// shared memory cm APIs, by enabling shared memory if the
// underlying hardware supports it.
void cmExecutor::InitSharedMemory() {
  if (!device_state_->info().host_unified_memory()) {
    return;
  }

  for (auto cap : device_state_->info().svm_capability()) {
    if (cap != proto::SvmCapability::FineGrainBuffer) {
      continue;
    }
    VLOG(3) << "Enabling cm fine-grain SVM memory";

    shared_memory_ = std::make_unique<cmSharedMemory>(device_state_);
    break;
  }
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
