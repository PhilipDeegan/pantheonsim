// Host-GPU mode, the part that talks to the machine's real driver (docs/perf-design/09-host-gpu-spike.md). Off unless the program
// runs with VGPU_HOST_GPU=exact or =relaxed (docs/perf-design/12-host-gpu-relaxed.md); the decisions are in vgpu/host_gpu.hpp.
#pragma once

#include <cstdint>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/host_gpu.hpp"
#include "vgpu/host_gpu_tol.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/ptx/ast.hpp"

namespace vgpu::hostgpu {

struct Stats {
  uint64_t offloaded = 0, simulated = 0;
  uint64_t refused_kernel = 0, refused_size = 0, refused_launch = 0, failed = 0;
  // VGPU_HOST_GPU_VERIFY: launches run on both, by whether the kernel passed the exact scan and whether the bytes were equal.
  uint64_t pushed_bytes = 0, pulled_bytes = 0;   // moved to and from the card
  uint64_t same_exact_set = 0, differ_exact_set = 0, same_other = 0, differ_other = 0;
  // The machine-readable lines (host_gpu_relaxed:, host_gpu_refusals:). `card_runs` counts every launch that ran on the card, the ones a verify run
  // repeats in the simulator included; the seconds are wall time of the card path (copies, launch, wait) and of the simulator's launches.
  uint64_t card_runs = 0;
  double card_seconds = 0, sim_seconds = 0;
  // Launches kept in the simulator for their kernel's code, by why (vgpu/host_gpu.hpp RelaxedRefusal); `refused_exact` is exact mode's count of
  // launches whose kernel relaxed mode would admit with its default switches.
  uint64_t refused_never = 0, refused_atomics = 0, refused_off = 0, refused_exact = 0;
  TolReport tol;   // relaxed mode's verify: every compared launch
};

class Runtime {
 public:
  // The runtime for a device's launches, or null when the mode is off or could not be started (said once on stderr unless
  // VGPU_QUIET). Only device 0 of the simulated machine is served: its window is the one reserved on the card.
  static Runtime* for_device(const DeviceProfile& profile, int ordinal);

  // Runs the launch on the card if it is eligible and worth it, and returns true; false leaves the launch to the simulator, with
  // the simulator's memory untouched. `sim_seconds`: how long the simulator took for a launch it did run (feeds the estimate).
  bool try_launch(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args,
                  MemoryManager& mem);
  // `mem`, when the mode is verifying, is the simulator's memory after the launch it just ran.
  // Called before the simulator runs a launch: whatever only the card holds comes back, and the card's copy is no longer valid (the launch may write anything).
  void before_simulated(MemoryManager& mem);
  void note_simulated(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, double sim_seconds, const MemoryManager* mem = nullptr,
                      const std::vector<std::vector<uint8_t>>* args = nullptr);

  struct Impl;
  ~Runtime();

 private:
  Runtime();
  Impl* impl_;
};

}  // namespace vgpu::hostgpu
