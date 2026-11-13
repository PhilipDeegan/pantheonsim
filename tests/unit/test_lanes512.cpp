// The binary64 kernel of the AVX-512 table (src/simd/lanes512.inc: dop_m, DADD / DMUL / DFMA on register pairs, sixteen lanes to a step) against
// the AVX2 one on the same inputs: the same bits in every lane, nothing written outside the result rows, with the result rows a source pair and
// apart from them, at two alignments, in the four rounding modes. It is also held to a definition written here lane by lane, so that a mistake
// the two tables shared would show.
//
// The comparison needs a host that runs both tables. On any other host the executable prints a line starting with "SKIP:" and CMake reports the
// test as skipped (SKIP_REGULAR_EXPRESSION), not as passed. VGPU_SIMD does not matter here: the tables are asked for by level (kernels_for).
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "vgpu/simd/simd.hpp"
#include "vtest.hpp"

using namespace vgpu;
using simd::Kernels;
using simd::Level;

namespace {

const Kernels& K2() { return simd::kernels_for(Level::Avx2); }
const Kernels& K5() { return simd::kernels_for(Level::Avx512); }

struct Gen {
  std::mt19937_64 rng{0x5eed51250ull};
};

// A buffer of rows with guard words around each: `at(i)` is row i, 64 lanes long, `off` words past a 64-byte boundary.
struct Rows {
  static constexpr unsigned kGuard = 16, kLen = 64, kStep = kGuard + kLen + kGuard + 16;
  std::vector<uint32_t> mem;
  unsigned off;
  Rows(unsigned rows, unsigned off_) : mem(rows * kStep + 64, 0xdeadbeefu), off(off_) {}
  uint32_t* at(unsigned i) {
    const uintptr_t base = (reinterpret_cast<uintptr_t>(mem.data()) + 63) & ~uintptr_t{63};
    return reinterpret_cast<uint32_t*>(base) + i * kStep + kGuard + off;
  }
  bool guards_intact(unsigned i, unsigned used) {
    const uint32_t* p = at(i);
    for (unsigned g = 1; g <= kGuard; ++g)
      if (p[-static_cast<int>(g)] != 0xdeadbeefu) return false;
    for (unsigned g = used; g < kLen + kGuard; ++g)
      if (p[g] != 0xdeadbeefu) return false;
    return true;
  }
};

const int kRounding[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};

bool same(const uint32_t* x, const uint32_t* y, unsigned n) { return std::memcmp(x, y, n * sizeof(uint32_t)) == 0; }

}  // namespace

VTEST(the_avx512_table_has_its_own_dop_m) {
  // (Otherwise the comparison below would be the AVX2 kernel against itself.)
  VCHECK(K5().level == Level::Avx512);
  VCHECK(K2().level == Level::Avx2);
  VCHECK(K5().dop_m != K2().dop_m);
  // The other lane kernels have no 512-bit form (round sp50): the Avx512 table's are the AVX2 functions.
  VCHECK(K5().iadd3 == K2().iadd3 && K5().ffma == K2().ffma && K5().fmul == K2().fmul && K5().blend_m == K2().blend_m && K5().mufu == K2().mufu);
}

VTEST(dop_m_gives_the_avx2_bits_in_every_rounding_mode) {
  // binary64 lanes as register pairs: sums, products and fused multiply-adds with every |x| / -x modifier, NaNs of every kind in every operand
  // (the result is then simd::dop_nan of the operands as the registers hold them), infinities whose sum is a new NaN, masks of every shape, and
  // result rows that are source rows.
  Gen g;
  const uint64_t edges[] = {0x0000000000000000ull, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull, 0x7ff0000000000000ull,
                            0xfff0000000000000ull, 0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff0000000000001ull, 0xfff0000000000001ull,
                            0x7ff8000000012345ull, 0xfff4000000054321ull, 0x7fffffffffffffffull, 0x0000000000000001ull, 0x000fffffffffffffull,
                            0x0010000000000000ull, 0x7fefffffffffffffull, 0xffefffffffffffffull, 0x3ff0000000000001ull, 0x3ca0000000000000ull,
                            0x4340000000000000ull, 0x3fd5555555555555ull};
  const unsigned n_edges = sizeof edges / sizeof edges[0];
  const auto value = [&](unsigned style) -> uint64_t {
    switch (style % 4) {
      case 0: return g.rng();
      case 1: return edges[g.rng() % n_edges];
      case 2: return std::bit_cast<uint64_t>(static_cast<double>(static_cast<int>(g.rng() % 2001) - 1000) / 8.0);
      default: return (g.rng() & 3) ? std::bit_cast<uint64_t>(static_cast<double>(static_cast<int64_t>(g.rng() % 2000001) - 1000000) * 1e-3) : edges[g.rng() % n_edges];
    }
  };
  const auto is_nan = [](uint64_t u) { return (u & 0x7fffffffffffffffull) > 0x7ff0000000000000ull; };
  const int before = std::fegetround();
  unsigned nan_rule_lanes = 0, masked_lanes = 0;
  for (int mode : kRounding) {
    std::fesetround(mode);
    for (unsigned round = 0; round < 400; ++round) {
      const unsigned kind = round % 3, mods = (round / 3) % 8 == 0 ? 0u : static_cast<unsigned>(g.rng() % 64), off = round % 5 == 0 ? 3u : 0u;
      const uint32_t mask = round % 7 == 0 ? 0xffffffffu : round % 7 == 1 ? 0u : round % 7 == 2 ? 0xffff0000u : round % 7 == 3 ? 0x0000ffffu : static_cast<uint32_t>(g.rng());
      // rows: 0/1 a lo/hi, 2/3 b, 4/5 c, 6/7 d (the destination, when it is not a source pair)
      Rows m2(8, off), m5(8, off);
      uint64_t A[32], B[32], C[32], D0[32];
      for (unsigned l = 0; l < 32; ++l) {
        A[l] = value(round + l), B[l] = value(round / 2 + 1), C[l] = value(round / 3 + 2), D0[l] = g.rng();
        if (round % 11 == 0 && l % 5 == 0) B[l] = A[l] ^ 0x8000000000000000ull;   // a - a, inf - inf
        const uint64_t vals[4] = {A[l], B[l], C[l], D0[l]};
        for (unsigned k = 0; k < 4; ++k) {
          m2.at(2 * k)[l] = static_cast<uint32_t>(vals[k]);
          m2.at(2 * k + 1)[l] = static_cast<uint32_t>(vals[k] >> 32);
        }
      }
      for (unsigned i = 0; i < 8; ++i) std::memcpy(m5.at(i), m2.at(i), 32 * sizeof(uint32_t));
      const unsigned alias = (round / 2) % 4;   // 0: d apart, 1: d is a, 2: d is b, 3: d is c
      const unsigned drow = alias ? 2 * (alias - 1) : 6;
      const auto call = [&](const Kernels& k, Rows& m) { k.dop_m(kind, mods, m.at(drow), m.at(drow + 1), m.at(0), m.at(1), m.at(2), m.at(3), m.at(4), m.at(5), mask); };
      uint64_t was[32];
      for (unsigned l = 0; l < 32; ++l) was[l] = uint64_t{m2.at(drow)[l]} | (uint64_t{m2.at(drow + 1)[l]} << 32);
      call(K2(), m2);
      call(K5(), m5);
      for (unsigned i = 0; i < 8; ++i) {
        VCHECK(same(m2.at(i), m5.at(i), 32));
        VCHECK(m5.guards_intact(i, 32));
      }
      for (unsigned l = 0; l < 32; ++l) {
        const uint64_t got = uint64_t{m5.at(drow)[l]} | (uint64_t{m5.at(drow + 1)[l]} << 32);
        if (!((mask >> l) & 1)) {
          VCHECK_EQ(got, was[l]);
          ++masked_lanes;
          continue;
        }
        const auto mod = [&](uint64_t u, unsigned k) { return ((mods >> (2 * k)) & 1 ? u & 0x7fffffffffffffffull : u) ^ ((mods >> (2 * k + 1)) & 1 ? 0x8000000000000000ull : 0); };
        uint64_t want;
        if (is_nan(A[l]) || is_nan(B[l]) || (kind == 2 && is_nan(C[l]))) {
          want = simd::dop_nan(kind, A[l], B[l], kind == 2 ? C[l] : 0);
          ++nan_rule_lanes;
        } else {
          volatile double x = std::bit_cast<double>(mod(A[l], 0)), y = std::bit_cast<double>(mod(B[l], 1)), z = std::bit_cast<double>(mod(C[l], 2));
          volatile double v = kind == 0 ? x + y : kind == 1 ? x * y : std::fma(x, y, z);
          want = std::bit_cast<uint64_t>(static_cast<double>(v));
        }
        VCHECK_EQ(got, want);
      }
    }
  }
  std::fesetround(before);
  VCHECK(nan_rule_lanes > 1000);   // (the rare path is not rare here)
  VCHECK(masked_lanes > 1000);
}

int main(int argc, char** argv) {
  if (!simd::supported(Level::Avx512) || !simd::supported(Level::Avx2)) {
    std::printf("SKIP: this host has no AVX-512 F/BW/DQ/VL/VNNI: the 512-bit dop_m cannot be run against the AVX2 one here\n");
    return 0;
  }
  return ::vtest::run_all(argc, argv);
}
