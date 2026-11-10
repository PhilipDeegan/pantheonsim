// See vgpu/host_stats.hpp.
#include "vgpu/host_stats.hpp"

#include <sys/resource.h>
#include <sys/utsname.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#include "vgpu/exec/worker_pool.hpp"
#include "vgpu/host_cpus.hpp"
#include "vgpu/memory_backing.hpp"

namespace vgpu {
namespace {

using Clock = std::chrono::steady_clock;
const Clock::time_point g_start = Clock::now();

double secs(const timeval& t) { return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_usec) * 1e-6; }

// The number after `key` on the line of `path` that starts with it (0 when there is none).
uint64_t field(const char* path, const std::string& key) {
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line))
    if (line.compare(0, key.size(), key) == 0) return std::strtoull(line.c_str() + key.size(), nullptr, 10);
  return 0;
}

// "madvise" out of "always [madvise] never".
std::string thp_mode() {
  std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled");
  std::string line;
  if (!std::getline(f, line)) return "unknown";
  const auto a = line.find('['), b = line.find(']');
  return a != std::string::npos && b != std::string::npos && b > a ? line.substr(a + 1, b - a - 1) : line;
}

std::mutex g_out_mu;

void emit(const std::string& line) {
  std::lock_guard<std::mutex> g(g_out_mu);
  std::FILE* f = stderr;
  bool close = false;
  if (const char* p = std::getenv("VGPU_HOST_STATS_FILE"); p && *p) {
    if (std::FILE* o = std::fopen(p, "a")) {
      f = o;
      close = true;
    }
  }
  std::fprintf(f, "%s\n", line.c_str());
  std::fflush(f);
  if (close) std::fclose(f);
}

double age() { return std::chrono::duration<double>(Clock::now() - g_start).count(); }

// The periodic reporter. It sleeps on a condition variable so the exit handler can stop it at once.
struct Sampler {
  std::mutex mu;
  std::condition_variable cv;
  bool stop = false;
  std::thread th;
};
Sampler* g_sampler = nullptr;

void at_exit() {
  if (g_sampler) {
    {
      std::lock_guard<std::mutex> g(g_sampler->mu);
      g_sampler->stop = true;
    }
    g_sampler->cv.notify_all();
    if (g_sampler->th.joinable()) g_sampler->th.join();
  }
  emit(host_stats_line(age()));
}

}  // namespace

std::string host_stats_line(double elapsed_s) {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  utsname un{};
  uname(&un);
  const exec::WorkerPoolStats pool = exec::worker_pool_stats();
  const backing::KernelCalls kc = backing::kernel_calls();
  char buf[768];
  std::snprintf(buf, sizeof buf,
                "vgpu-host-stats t=%.1fs cpu_user=%.1fs cpu_sys=%.1fs minflt=%ld majflt=%ld vol_cs=%ld invol_cs=%ld threads=%llu "
                "thp_kb=%llu madvise_dontneed=%llu arena_maps=%llu pool_threads=%zu pool_calls=%llu kernel=%s cpus=%u thp=%s",
                elapsed_s, secs(ru.ru_utime), secs(ru.ru_stime), ru.ru_minflt, ru.ru_majflt, ru.ru_nvcsw, ru.ru_nivcsw,
                static_cast<unsigned long long>(field("/proc/self/status", "Threads:")),
                static_cast<unsigned long long>(field("/proc/self/smaps_rollup", "AnonHugePages:")),
                static_cast<unsigned long long>(kc.madvise_dontneed), static_cast<unsigned long long>(kc.arena_maps), pool.created,
                static_cast<unsigned long long>(pool.calls), un.release, host_cpus(), thp_mode().c_str());
  return buf;
}

void host_stats_start() {
  static std::once_flag once;
  std::call_once(once, [] {
    const char* e = std::getenv("VGPU_HOST_STATS");
    if (!e || !e[0] || e[0] == '0') return;
    const long every = std::strtol(e, nullptr, 10);   // "1" and a word both mean the exit line only
    std::atexit(at_exit);
    if (every > 1) {
      g_sampler = new Sampler;   // leaked: the exit handler stops and joins the thread, and the object outlives it
      g_sampler->th = std::thread([every] {
        std::unique_lock<std::mutex> l(g_sampler->mu);
        while (!g_sampler->stop) {
          if (g_sampler->cv.wait_for(l, std::chrono::seconds(every), [] { return g_sampler->stop; })) break;
          l.unlock();
          emit(host_stats_line(age()));
          l.lock();
        }
      });
    }
  });
}

}  // namespace vgpu
