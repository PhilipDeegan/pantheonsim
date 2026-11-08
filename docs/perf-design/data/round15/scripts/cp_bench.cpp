// What a copy-and-patch compile of a JIT unit would cost, as a micro-benchmark (not an implementation): for N ops, copy a stencil of S bytes into an RW mapping and patch H
// 32-bit holes in it (register-row offsets, a constant, a branch target), then mprotect the mapping RX. Also times the mmap and mprotect themselves.
// Usage: cp_bench [N...]   (default 100 1500 20000)
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Clock = std::chrono::steady_clock;
static double us_since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
  std::vector<unsigned> ns;
  for (int i = 1; i < argc; ++i) ns.push_back(static_cast<unsigned>(std::strtoul(argv[i], nullptr, 10)));
  if (ns.empty()) ns = {100, 1500, 20000};
  // A library of 400 stencils of 96..640 bytes (the sizes this JIT's compiled op bodies have: see the study), each with 1 to 4 holes at fixed offsets.
  struct Stencil { std::vector<uint8_t> code; std::vector<uint16_t> holes; };
  std::vector<Stencil> lib(400);
  uint64_t x = 88172645463325252ull;
  const auto rnd = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
  for (Stencil& s : lib) {
    s.code.resize(96 + rnd() % 545);
    for (auto& b : s.code) b = static_cast<uint8_t>(rnd());
    s.code.back() = 0xc3;
    const unsigned h = 1 + static_cast<unsigned>(rnd() % 4);
    for (unsigned i = 0; i < h; ++i) s.holes.push_back(static_cast<uint16_t>(rnd() % (s.code.size() - 8)));
  }
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  for (unsigned n : ns) {
    std::vector<uint32_t> pick(n);
    for (auto& p : pick) p = static_cast<uint32_t>(rnd() % lib.size());
    size_t total = 0;
    for (unsigned p : pick) total += lib[p].code.size();
    const size_t map_len = (total + page - 1) / page * page;
    double best_copy = 1e18, best_total = 1e18, best_map = 1e18, best_prot = 1e18;
    for (int rep = 0; rep < 9; ++rep) {
      const auto t0 = Clock::now();
      uint8_t* m = static_cast<uint8_t*>(mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
      const double t_map = us_since(t0);
      const auto t1 = Clock::now();
      size_t off = 0;
      uint32_t patch_val = 0x1000;
      for (unsigned p : pick) {
        const Stencil& s = lib[p];
        std::memcpy(m + off, s.code.data(), s.code.size());
        for (uint16_t h : s.holes) {
          patch_val += 32;
          std::memcpy(m + off + h, &patch_val, 4);
        }
        off += s.code.size();
      }
      const double t_copy = us_since(t1);
      const auto t2 = Clock::now();
      const int rc = mprotect(m, map_len, PROT_READ | PROT_EXEC);
      const double t_prot = us_since(t2);
      if (rc != 0) std::perror("mprotect");
      best_copy = std::min(best_copy, t_copy);
      best_map = std::min(best_map, t_map);
      best_prot = std::min(best_prot, t_prot);
      best_total = std::min(best_total, us_since(t0));
      munmap(m, map_len);
    }
    std::printf("N=%-6u stencil bytes %-9zu mmap %.1f us  copy+patch %.1f us (%.1f ns/op)  mprotect %.1f us  total %.1f us\n", n, total, best_map, best_copy,
                best_copy * 1000.0 / n, best_prot, best_total);
  }
  return 0;
}
