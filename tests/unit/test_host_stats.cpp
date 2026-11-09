// VGPU_HOST_STATS (vgpu/host_stats.hpp): the line has the kernel-facing counters, the counters move when the process does what
// they count, and the environment variable arranges an exit line and, with a period, lines while the process runs.
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/host_stats.hpp"
#include "vtest.hpp"

#if defined(__SANITIZE_THREAD__)
#define VGPU_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define VGPU_TEST_TSAN 1
#endif
#endif

namespace {

long number_after(const std::string& line, const std::string& key) {
  const auto at = line.find(" " + key + "=");
  return at == std::string::npos ? -1 : std::strtol(line.c_str() + at + key.size() + 2, nullptr, 10);
}

// A child that sets the variables, starts the reporter, lives `ms` and exits normally (so atexit handlers run). Its lines.
std::vector<std::string> run_child(const char* value, int ms) {
  char path[] = "/tmp/vgpu-host-stats-XXXXXX";
  const int fd = mkstemp(path);
  close(fd);
  std::fflush(stdout);   // (a child would write the parent's buffered lines again)
  const pid_t pid = fork();
  if (pid == 0) {
    setenv("VGPU_HOST_STATS", value, 1);
    setenv("VGPU_HOST_STATS_FILE", path, 1);
    vgpu::host_stats_start();
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    std::exit(0);
  }
  int st = 0;
  waitpid(pid, &st, 0);
  std::vector<std::string> lines;
  std::ifstream f(path);
  for (std::string l; std::getline(f, l);) lines.push_back(l);
  unlink(path);
  return lines;
}

}  // namespace

VTEST(host_stats_line_has_the_counters) {
  const std::string a = vgpu::host_stats_line(1.0);
  for (const char* k : {"minflt", "majflt", "vol_cs", "invol_cs", "threads", "thp_kb", "madvise_dontneed", "arena_maps", "pool_threads",
                        "pool_calls", "cpus"})
    VCHECK(number_after(a, k) >= 0);
  VCHECK_CONTAINS(a, "kernel=");
  VCHECK_CONTAINS(a, "thp=");
  // Touching fresh pages is counted as page faults.
  const long before = number_after(a, "minflt");
  std::vector<char> big(64u << 20);
  for (size_t i = 0; i < big.size(); i += 4096) big[i] = 1;
  const long after = number_after(vgpu::host_stats_line(1.0), "minflt");
  VCHECK(after - before >= 10000);
}

#ifndef VGPU_TEST_TSAN
VTEST(host_stats_env_writes_an_exit_line) {
  const auto lines = run_child("1", 50);
  VCHECK_EQ(lines.size(), size_t{1});
  if (!lines.empty()) VCHECK_CONTAINS(lines[0], "vgpu-host-stats ");
}

VTEST(host_stats_period_adds_lines_while_running) {
  const auto lines = run_child("1", 50);
  const auto periodic = run_child("1x", 50);   // a word is the exit line only too
  VCHECK_EQ(periodic.size(), lines.size());
  const auto more = run_child("2", 2700);      // every 2 s for 2.7 s: one periodic line, then the exit line
  VCHECK_EQ(more.size(), size_t{2});
}

VTEST(host_stats_unset_or_zero_is_silent) {
  VCHECK_EQ(run_child("0", 50).size(), size_t{0});
}
#endif

VTEST_MAIN
