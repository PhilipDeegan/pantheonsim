#pragma once
// What the host kernel is doing for a simulator process, for finding out why a run is slower on one kernel than on another
// (pantheonsim #551: a CI job 2-3x slower after the runner image moved from kernel 6.17 to 7.0, with the same simulator build).
//
//   VGPU_HOST_STATS=1      one line on stderr when the process exits
//   VGPU_HOST_STATS=30     the same, and a line every 30 seconds while it runs, flushed, so a long run that is slow or stuck
//                          shows where it is even when the program's own output is block-buffered
//   VGPU_HOST_STATS_FILE=  append the lines to this file instead of stderr
//
// A line reads, for example,
//   vgpu-host-stats t=120.0s cpu_user=95.1s cpu_sys=21.7s minflt=4783885 majflt=0 vol_cs=9018 invol_cs=88 threads=5 \
//     thp_kb=0 madvise_dontneed=100000 arena_maps=3 pool_threads=4 pool_calls=88123 kernel=7.0.0-1017-azure cpus=4 thp=madvise
// A large cpu_sys against cpu_user, or minflt growing by millions, points at the kernel (page faults, mmap/madvise);
// invol_cs growing fast with the pool's threads busy points at scheduling; neither points at the simulator's own work.
// `strace -f -c -p PID` and `perf stat -p PID` give the per-syscall and per-event view of the same process.
#include <cstdint>
#include <string>

namespace vgpu {

// One line, from the process's own counters (getrusage, /proc/self/status, /proc/self/smaps_rollup, the allocator's and the
// worker pool's counts). `elapsed_s` is the process's age.
std::string host_stats_line(double elapsed_s);

// Reads VGPU_HOST_STATS once and, when it is set, arranges the lines above. Cheap and idempotent; called when a device is made.
void host_stats_start();

}  // namespace vgpu
