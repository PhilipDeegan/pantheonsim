// The 32-lane ALU kernels of the AVX-512 table (src/simd/lanes512.inc: iadd3, imad, lop3, ffma, fadd, fmul, icmp_mask, blend_m, iadd3x, imad_wide, mufu, dop_m), its
// two warp-addresses kernels (warp_rows, warp_run) and the 512-bit forms of the scheduler's lane-address functions (src/simd/lanes.cpp) against
// the AVX2 ones on the same inputs: the same bits in every lane, nothing written outside the result row, with the result row one of the source
// rows and apart from them, at any alignment, in the four rounding modes for the float ones. Each is also held to a definition written here lane
// by lane, so that a mistake the two tables shared would show.
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

#include "vgpu/exec/sass_lanes.hpp"
#include "vgpu/simd/simd.hpp"
#include "vtest.hpp"

using namespace vgpu;
using simd::Kernels;
using simd::Level;

namespace {

const Kernels& K2() { return simd::kernels_for(Level::Avx2); }
const Kernels& K5() { return simd::kernels_for(Level::Avx512); }

// Values that sit on every edge the kernels have: carries, signs, NaNs of each kind (quiet, signalling, with a sign, with a payload), infinities,
// subnormals, the largest and smallest normal numbers, and numbers whose sum or product rounds differently in each rounding mode.
const uint32_t kEdges[] = {
    0u,          1u,          2u,          0x7fffffffu, 0x80000000u, 0x80000001u, 0xffffffffu, 0xfffffffeu, 0x0000ffffu, 0x00010000u,
    0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff800001u, 0x7fc12345u, 0x7fffffffu, 0x7f800000u, 0xff800000u, 0x00000001u, 0x80000001u,
    0x007fffffu, 0x00800000u, 0x7f7fffffu, 0xff7fffffu, 0x3f800000u, 0xbf800000u, 0x3f800001u, 0x33800000u, 0x3effffffu, 0x4b800000u,
    31u,         32u,         33u,         63u,         64u,         0x55555555u, 0xaaaaaaaau, 0x12345678u};
constexpr unsigned kNEdges = sizeof kEdges / sizeof kEdges[0];

struct Gen {
  std::mt19937_64 rng{0x5eed51250ull};
  uint32_t word(unsigned style) {
    switch (style % 4) {
      case 0: return static_cast<uint32_t>(rng());
      case 1: return kEdges[rng() % kNEdges];
      case 2: return std::bit_cast<uint32_t>(static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 8.0f);   // small floats: exact sums, ties
      default: return (rng() & 1) ? kEdges[rng() % kNEdges] : static_cast<uint32_t>(rng());
    }
  }
  void row(uint32_t* p, unsigned n, unsigned style) {
    for (unsigned i = 0; i < n; ++i) p[i] = word(style);
  }
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

uint32_t canon(float f) { return f != f ? 0x7fffffffu : std::bit_cast<uint32_t>(f); }
float f(uint32_t u) { return std::bit_cast<float>(u); }

const int kRounding[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};

// Runs `body(style, off, alias)` over the input styles, three alignments and the aliasing forms 0 (apart), 1 (r is a), 2 (r is b), 3 (r is c).
template <class F>
void sweep(unsigned rounds, unsigned aliases, F body) {
  for (unsigned round = 0; round < rounds; ++round)
    for (unsigned off : {0u, 1u, 5u})
      for (unsigned alias = 0; alias < aliases; ++alias) body(round, off, alias);
}

bool same(const uint32_t* x, const uint32_t* y, unsigned n) { return std::memcmp(x, y, n * sizeof(uint32_t)) == 0; }

// One three-source kernel through both tables: sources a, b, c (rows 0 to 2), the results in rows 3 (AVX2) and 4 (AVX-512), or over the source
// row `alias` names (each table then gets its own copy of the sources). Returns the AVX-512 row; fails if the rows differ or a guard moved.
template <class Call>
const uint32_t* both3(Gen& g, unsigned style, unsigned off, unsigned alias, uint32_t* a, uint32_t* b, uint32_t* c, Rows& m2, Rows& m5, Call call) {
  g.row(m2.at(0), 32, style);
  g.row(m2.at(1), 32, style + 1);
  g.row(m2.at(2), 32, style + 3);
  for (unsigned i = 0; i < 3; ++i) std::memcpy(m5.at(i), m2.at(i), 32 * sizeof(uint32_t));
  std::memcpy(a, m2.at(0), 32 * sizeof(uint32_t));
  std::memcpy(b, m2.at(1), 32 * sizeof(uint32_t));
  std::memcpy(c, m2.at(2), 32 * sizeof(uint32_t));
  uint32_t* const r2 = alias ? m2.at(alias - 1) : m2.at(3);
  uint32_t* const r5 = alias ? m5.at(alias - 1) : m5.at(3);
  call(K2(), r2, m2.at(0), m2.at(1), m2.at(2));
  call(K5(), r5, m5.at(0), m5.at(1), m5.at(2));
  VCHECK(same(r2, r5, 32));
  for (unsigned i = 0; i < 3; ++i) {
    VCHECK(m2.guards_intact(i, 32));
    VCHECK(m5.guards_intact(i, 32));
    if (alias != i + 1) VCHECK(same(m5.at(i), m2.at(i), 32));   // (a source row that is not the result row is not written)
  }
  if (!alias) VCHECK(m5.guards_intact(3, 32));
  (void)off;
  return r5;
}

}  // namespace

VTEST(the_avx512_table_has_its_own_lane_kernels) {
  // (Otherwise every comparison below would be the AVX2 kernel against itself.)
  VCHECK(K5().level == Level::Avx512);
  VCHECK(K2().level == Level::Avx2);
  VCHECK(K5().iadd3 != K2().iadd3);
  VCHECK(K5().imad != K2().imad);
  VCHECK(K5().lop3 != K2().lop3);
  VCHECK(K5().ffma != K2().ffma);
  VCHECK(K5().fadd != K2().fadd);
  VCHECK(K5().fmul != K2().fmul);
  VCHECK(K5().icmp_mask != K2().icmp_mask);
  VCHECK(K5().blend_m != K2().blend_m);
  VCHECK(K5().iadd3x != K2().iadd3x);
  VCHECK(K5().dop_m != K2().dop_m);
  VCHECK(K5().imad_wide != K2().imad_wide);
  VCHECK(K5().mufu != K2().mufu);
}

VTEST(iadd3_and_imad_give_the_avx2_bits) {
  Gen g;
  uint32_t a[32], b[32], c[32];
  const uint32_t adds[] = {0u, 1u, 0xffffffffu, 0x80000000u, 0x12345u};
  sweep(40, 4, [&](unsigned round, unsigned off, unsigned alias) {
    Rows m2(4, off), m5(4, off);
    const uint32_t add = adds[round % 5];
    const uint32_t* r = both3(g, round, off, alias, a, b, c, m2, m5,
                              [&](const Kernels& k, uint32_t* r_, const uint32_t* a_, const uint32_t* b_, const uint32_t* c_) { k.iadd3(r_, a_, b_, c_, add); });
    for (unsigned l = 0; l < 32; ++l) VCHECK_EQ(r[l], a[l] + b[l] + c[l] + add);
    Rows n2(4, off), n5(4, off);
    r = both3(g, round, off, alias, a, b, c, n2, n5,
              [&](const Kernels& k, uint32_t* r_, const uint32_t* a_, const uint32_t* b_, const uint32_t* c_) { k.imad(r_, a_, b_, c_); });
    for (unsigned l = 0; l < 32; ++l) VCHECK_EQ(r[l], a[l] * b[l] + c[l]);
  });
}

VTEST(lop3_gives_the_avx2_bits_for_every_table) {
  Gen g;
  uint32_t a[32], b[32], c[32];
  for (uint32_t lut = 0; lut < 256; ++lut)
    sweep(3, 4, [&](unsigned round, unsigned off, unsigned alias) {
      Rows m2(4, off), m5(4, off);
      const uint32_t* r = both3(g, round, off, alias, a, b, c, m2, m5,
                                [&](const Kernels& k, uint32_t* r_, const uint32_t* a_, const uint32_t* b_, const uint32_t* c_) { k.lop3(r_, a_, b_, c_, lut); });
      for (unsigned l = 0; l < 32; ++l) {
        uint32_t want = 0;
        for (unsigned bit = 0; bit < 32; ++bit) {
          const unsigned t = (((a[l] >> bit) & 1) << 2) | (((b[l] >> bit) & 1) << 1) | ((c[l] >> bit) & 1);
          want |= ((lut >> t) & 1u) << bit;
        }
        VCHECK_EQ(r[l], want);
      }
    });
}

VTEST(ffma_fadd_fmul_give_the_avx2_bits_in_every_rounding_mode) {
  Gen g;
  uint32_t a[32], b[32], c[32];
  const int before = std::fegetround();
  for (int mode : kRounding) {
    std::fesetround(mode);
    sweep(60, 4, [&](unsigned round, unsigned off, unsigned alias) {
      Rows m2(4, off), m5(4, off);
      const uint32_t* r = both3(g, round, off, alias, a, b, c, m2, m5,
                                [&](const Kernels& k, uint32_t* r_, const uint32_t* a_, const uint32_t* b_, const uint32_t* c_) { k.ffma(r_, a_, b_, c_); });
      for (unsigned l = 0; l < 32; ++l) VCHECK_EQ(r[l], canon(std::fmaf(f(a[l]), f(b[l]), f(c[l]))));
      if (alias == 3) return;   // (two sources)
      Rows n2(4, off), n5(4, off);
      r = both3(g, round, off, alias, a, b, c, n2, n5,
                [&](const Kernels& k, uint32_t* r_, const uint32_t* a_, const uint32_t* b_, const uint32_t*) { k.fadd(r_, a_, b_); });
      for (unsigned l = 0; l < 32; ++l) {
        volatile float x = f(a[l]), y = f(b[l]);
        volatile float s = x + y;
        VCHECK_EQ(r[l], canon(s));
      }
      Rows p2(4, off), p5(4, off);
      r = both3(g, round, off, alias, a, b, c, p2, p5,
                [&](const Kernels& k, uint32_t* r_, const uint32_t* a_, const uint32_t* b_, const uint32_t*) { k.fmul(r_, a_, b_); });
      for (unsigned l = 0; l < 32; ++l) {
        volatile float x = f(a[l]), y = f(b[l]);
        volatile float s = x * y;
        VCHECK_EQ(r[l], canon(s));
      }
    });
  }
  std::fesetround(before);
}

VTEST(every_edge_value_in_every_lane_of_the_float_kernels) {
  // One edge value in one lane, the rest ordinary numbers: a NaN, an infinity or a subnormal in lane 15, 16 or 31 must not leak into its neighbours.
  uint32_t a[32], b[32], c[32], r2[32], r5[32];
  for (unsigned e = 0; e < kNEdges; ++e)
    for (unsigned lane = 0; lane < 32; ++lane)
      for (unsigned which = 0; which < 3; ++which) {
        for (unsigned l = 0; l < 32; ++l) {
          a[l] = std::bit_cast<uint32_t>(1.5f + static_cast<float>(l));
          b[l] = std::bit_cast<uint32_t>(-0.75f * static_cast<float>(l + 1));
          c[l] = std::bit_cast<uint32_t>(3.0f);
        }
        (which == 0 ? a : which == 1 ? b : c)[lane] = kEdges[e];
        K2().ffma(r2, a, b, c);
        K5().ffma(r5, a, b, c);
        VCHECK(same(r2, r5, 32));
        K2().fadd(r2, a, b);
        K5().fadd(r5, a, b);
        VCHECK(same(r2, r5, 32));
        K2().fmul(r2, a, b);
        K5().fmul(r5, a, b);
        VCHECK(same(r2, r5, 32));
        for (unsigned l = 0; l < 32; ++l) VCHECK(r5[l] == 0x7fffffffu || (r5[l] & 0x7fffffffu) <= 0x7f800000u);   // a NaN is the canonical one
      }
}

VTEST(icmp_mask_gives_the_avx2_bits) {
  Gen g;
  for (unsigned round = 0; round < 200; ++round)
    for (unsigned off : {0u, 1u, 5u})
      for (unsigned n : {32u, 64u}) {
        Rows m(2, off);
        g.row(m.at(0), n, round);
        g.row(m.at(1), n, round + 1);
        const uint32_t *a = m.at(0), *b = m.at(1);
        if (round % 3 == 0)
          for (unsigned l = 0; l < n; l += 1 + round % 5) m.at(1)[l] = a[l];   // equal lanes
        for (unsigned cmp = 0; cmp < 6; ++cmp)
          for (bool sgn : {false, true}) {
            const uint64_t got2 = K2().icmp_mask(cmp, sgn, a, b, n), got5 = K5().icmp_mask(cmp, sgn, a, b, n);
            uint64_t want = 0;
            for (unsigned l = 0; l < n; ++l) {
              const uint32_t x = a[l], y = b[l];
              const int32_t xs = static_cast<int32_t>(x), ys = static_cast<int32_t>(y);
              const bool t = sgn ? (cmp == 0 ? xs == ys : cmp == 1 ? xs != ys : cmp == 2 ? xs < ys : cmp == 3 ? xs <= ys : cmp == 4 ? xs > ys : xs >= ys)
                                 : (cmp == 0 ? x == y : cmp == 1 ? x != y : cmp == 2 ? x < y : cmp == 3 ? x <= y : cmp == 4 ? x > y : x >= y);
              want |= uint64_t{t} << l;
            }
            VCHECK_EQ(got5, got2);
            VCHECK_EQ(got5, want);
          }
        VCHECK(m.guards_intact(0, n));
        VCHECK(m.guards_intact(1, n));
      }
}

VTEST(blend_m_gives_the_avx2_bits_for_every_shape_of_mask) {
  Gen g;
  std::vector<uint32_t> masks = {0u, 0xffffffffu, 0x0000ffffu, 0xffff0000u, 0x00008000u, 0x00010000u, 0x80000000u, 1u, 0x7fffffffu, 0xfffffffeu, 0x55555555u, 0xaaaaaaaau};
  for (unsigned l = 0; l < 32; ++l) masks.push_back(1u << l), masks.push_back(~(1u << l));
  for (unsigned i = 0; i < 200; ++i) masks.push_back(static_cast<uint32_t>(g.rng()));
  for (uint32_t mask : masks)
    for (unsigned off : {0u, 1u, 5u}) {
      Rows m2(2, off), m5(2, off);
      g.row(m2.at(0), 32, 0);
      g.row(m2.at(1), 32, 1);
      uint32_t d0[32], r[32];
      std::memcpy(d0, m2.at(0), sizeof d0);
      std::memcpy(r, m2.at(1), sizeof r);
      std::memcpy(m5.at(0), d0, sizeof d0);
      std::memcpy(m5.at(1), r, sizeof r);
      K2().blend_m(m2.at(0), m2.at(1), mask);
      K5().blend_m(m5.at(0), m5.at(1), mask);
      VCHECK(same(m2.at(0), m5.at(0), 32));
      for (unsigned l = 0; l < 32; ++l) VCHECK_EQ(m5.at(0)[l], ((mask >> l) & 1) ? r[l] : d0[l]);
      VCHECK(same(m5.at(1), r, 32));
      VCHECK(m5.guards_intact(0, 32));
      VCHECK(m5.guards_intact(1, 32));
    }
}

VTEST(iadd3x_gives_the_avx2_bits_and_carries) {
  Gen g;
  uint32_t a[32], b[32], c[32];
  // (An addend of all ones with a carry in wraps in the 32-bit number the AVX2 kernel makes of the three; the kernels agree there too, and the
  // definition below is held only to the addends that do not wrap.)
  const uint32_t adds[] = {0u, 1u, 2u, 0x7fffffffu, 0x80000000u, 0xfffffffdu, 0xfffffffeu, 0xffffffffu};
  sweep(120, 4, [&](unsigned round, unsigned off, unsigned alias) {
    Rows m2(4, off), m5(4, off);
    const uint32_t add = adds[round % 8];
    const uint32_t cin_a = round % 7 == 0 ? 0u : round % 7 == 1 ? 0xffffffffu : static_cast<uint32_t>(g.rng());
    const uint32_t cin_b = round % 11 == 0 ? 0xffffffffu : round % 11 < 4 ? 0u : static_cast<uint32_t>(g.rng());
    uint32_t ge1[2] = {0x11111111u, 0x11111111u}, ge2[2] = {0x22222222u, 0x22222222u};
    unsigned call = 0;
    const uint32_t* r = both3(g, round, off, alias, a, b, c, m2, m5, [&](const Kernels& k, uint32_t* r_, const uint32_t* a_, const uint32_t* b_, const uint32_t* c_) {
      k.iadd3x(r_, a_, b_, c_, add, cin_a, cin_b, &ge1[call], &ge2[call]);
      ++call;
    });
    VCHECK_EQ(ge1[1], ge1[0]);
    VCHECK_EQ(ge2[1], ge2[0]);
    for (unsigned l = 0; l < 32; ++l) {
      const uint32_t ca = (cin_a >> l) & 1u, cb = (cin_b >> l) & 1u;
      const uint64_t sum = uint64_t{a[l]} + b[l] + c[l] + add + ca + cb;
      VCHECK_EQ(r[l], static_cast<uint32_t>(sum));
      if (uint64_t{add} + ca + cb <= 0xffffffffull) {
        VCHECK_EQ((ge1[1] >> l) & 1u, static_cast<uint32_t>((sum >> 32) >= 1));
        VCHECK_EQ((ge2[1] >> l) & 1u, static_cast<uint32_t>((sum >> 32) >= 2));
      }
    }
  });
}

VTEST(imad_wide_gives_the_avx2_bits_and_carries) {
  Gen g;
  unsigned carries = 0;
  for (unsigned round = 0; round < 3000; ++round) {
    const unsigned off = round % 3 == 0 ? 0u : round % 7;
    // rows: a, b, clo, chi, rlo, rhi
    Rows m2(6, off), m5(6, off);
    for (unsigned i = 0; i < 4; ++i) g.row(m2.at(i), 32, round + i);
    if (round % 5 == 0)   // an addend near the top, so that the sum carries out
      for (unsigned l = 0; l < 32; ++l) m2.at(3)[l] = 0xffffffffu - (g.rng() % 3), m2.at(2)[l] = 0xffffffffu - static_cast<uint32_t>(g.rng() % 5);
    for (unsigned i = 0; i < 4; ++i) std::memcpy(m5.at(i), m2.at(i), 32 * sizeof(uint32_t));
    uint32_t a[32], b[32], cl[32], ch[32];
    std::memcpy(a, m2.at(0), sizeof a), std::memcpy(b, m2.at(1), sizeof b), std::memcpy(cl, m2.at(2), sizeof cl), std::memcpy(ch, m2.at(3), sizeof ch);
    const bool sgn = round % 2;
    const uint32_t cin = (round / 2) % 3 == 0 ? 0u : (round / 2) % 3 == 1 ? 0xffffffffu : static_cast<uint32_t>(g.rng());
    const bool want_carry = (round / 6) % 2;
    // The destination pair is apart from the sources, or the addend's own rows (IMAD.WIDE into its addend register pair).
    const unsigned drow = (round / 12) % 2 ? 2 : 4;
    uint32_t c2 = 0x33333333u, c5 = 0x44444444u;
    K2().imad_wide(m2.at(drow), m2.at(drow + 1), m2.at(0), m2.at(1), m2.at(2), m2.at(3), sgn, cin, want_carry ? &c2 : nullptr);
    K5().imad_wide(m5.at(drow), m5.at(drow + 1), m5.at(0), m5.at(1), m5.at(2), m5.at(3), sgn, cin, want_carry ? &c5 : nullptr);
    for (unsigned i = 0; i < 6; ++i) {
      VCHECK(same(m2.at(i), m5.at(i), 32));
      VCHECK(m5.guards_intact(i, 32));
    }
    if (want_carry) VCHECK_EQ(c5, c2);
    else VCHECK_EQ(c5, 0x44444444u);   // (not written when it is not asked for)
    for (unsigned l = 0; l < 32; ++l) {
      const uint64_t prod = sgn ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a[l])) * static_cast<int32_t>(b[l])) : uint64_t{a[l]} * b[l];
      const unsigned __int128 full = static_cast<unsigned __int128>(prod) + (uint64_t{cl[l]} | (uint64_t{ch[l]} << 32)) + ((cin >> l) & 1u);
      VCHECK_EQ(m5.at(drow)[l], static_cast<uint32_t>(full));
      VCHECK_EQ(m5.at(drow + 1)[l], static_cast<uint32_t>(static_cast<uint64_t>(full) >> 32));
      if (want_carry) {
        VCHECK_EQ((c5 >> l) & 1u, static_cast<uint32_t>((full >> 64) != 0));
        carries += (c5 >> l) & 1u;
      }
    }
  }
  VCHECK(carries > 1000);
}

VTEST(mufu_gives_the_avx2_bits_in_every_rounding_mode) {
  Gen g;
  const int before = std::fegetround();
  for (int mode : kRounding) {
    std::fesetround(mode);
    for (unsigned round = 0; round < 400; ++round)
      for (unsigned kind : {4u, 5u, 8u}) {
        const unsigned off = round % 4;
        Rows m2(2, off), m5(2, off);
        g.row(m2.at(0), 32, round);
        if (round % 6 == 0)
          for (unsigned l = 0; l < 32; ++l) m2.at(0)[l] = kEdges[(round + l * 7 + l / 8) % kNEdges];
        if (round % 6 == 1)
          for (unsigned l = 0; l < 32; ++l) m2.at(0)[l] = std::bit_cast<uint32_t>(std::ldexp(1.0f + static_cast<float>(g.rng() % 4096) / 4096.0f, static_cast<int>(g.rng() % 300) - 150));
        std::memcpy(m5.at(0), m2.at(0), 32 * sizeof(uint32_t));
        uint32_t a[32];
        std::memcpy(a, m2.at(0), sizeof a);
        const unsigned drow = round % 2;   // the result apart from the source, or over it
        K2().mufu(kind, m2.at(drow), m2.at(0));
        K5().mufu(kind, m5.at(drow), m5.at(0));
        for (unsigned i = 0; i < 2; ++i) {
          VCHECK(same(m2.at(i), m5.at(i), 32));
          VCHECK(m5.guards_intact(i, 32));
        }
        for (unsigned l = 0; l < 32; ++l) {
          const uint32_t in = (a[l] & 0x7f800000u) ? a[l] : a[l] & 0x80000000u;
          volatile double x = static_cast<double>(std::bit_cast<float>(in));
          volatile double v = kind == 4 ? 1.0 / x : kind == 5 ? 1.0 / std::sqrt(x) : std::sqrt(x);
          volatile float fl = static_cast<float>(v);
          const float res = fl;
          const uint32_t u = std::bit_cast<uint32_t>(res);
          const uint32_t want = res != res ? 0x7fffffffu : (u & 0x7f800000u) ? u : u & 0x80000000u;
          VCHECK_EQ(m5.at(drow)[l], want);
        }
      }
  }
  std::fesetround(before);
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

VTEST(warp_rows_and_warp_run_give_the_avx2_answers) {
  const simd::WarpRowsFn rows2 = simd::testing::warp_rows_at(Level::Avx2), rows5 = simd::testing::warp_rows_at(Level::Avx512);
  const simd::WarpRunFn run2 = simd::testing::warp_run_at(Level::Avx2), run5 = simd::testing::warp_run_at(Level::Avx512);
  VCHECK(rows2 && rows5 && rows2 != rows5);
  VCHECK(run2 && run5 && run2 != run5);
  Gen g;
  unsigned progressions = 0, runs = 0, in_window = 0, not_runs = 0;
  for (unsigned round = 0; round < 6000; ++round) {
    uint32_t lo[32], hi[32];
    const unsigned bytes_of[] = {0, 1, 2, 4, 8};
    const unsigned bytes = bytes_of[round % 5];
    const uint64_t strides[] = {0, 1, 2, 4, 8, 12, 4096, 0x100000000ull, 0xfffffffffffffffcull, 0x7fffffffull};
    const uint64_t stride = round % 3 == 0 ? bytes : strides[g.rng() % 10];
    // A base whose low word wraps inside the warp in some rounds, and whose 64 bits wrap in a few.
    const uint64_t bases[] = {g.rng(), 0x00000000fffffff0ull, 0x7fffffffffffff00ull, 0xffffffffffffffe0ull, 0x0000000100000000ull - 4 * stride, 0x12345678ull};
    const uint64_t base = bases[round % 6];
    for (unsigned l = 0; l < 32; ++l) {
      const uint64_t a = base + l * stride;
      lo[l] = static_cast<uint32_t>(a);
      hi[l] = static_cast<uint32_t>(a >> 32);
    }
    if (round % 4 == 1) {   // one lane off the stride: every lane and both words in turn
      const unsigned l = (round / 4) % 32;
      ((round / 128) % 2 ? hi : lo)[l] += 1u << (round % 31);
    } else if (round % 4 == 2) {
      for (unsigned l = 0; l < 32; ++l) lo[l] = g.word(round), hi[l] = g.word(round + 1) & 0xffu;
    }
    const uint64_t add = round % 2 ? g.rng() : (g.rng() & 0xfff);
    const uint64_t a0 = (uint64_t{hi[0]} << 32 | lo[0]) + add;
    const uint64_t windows[] = {a0, a0 - 0xffffffffull, a0 + 1, a0 - 0x100000000ull, g.rng(), a0 - 16 * stride, 0};
    const uint64_t window = windows[round % 7];
    for (unsigned form = 0; form < 3; ++form) {   // lo and hi; lo alone; neither
      const uint32_t* const plo = form == 2 ? nullptr : lo;
      const uint32_t* const phi = form == 0 ? hi : nullptr;
      alignas(64) uint64_t ad2[32 + 2], ad5[32 + 2];
      ad2[32] = ad5[32] = 0xfeedfacecafebeefull;
      simd::WarpRowsInfo i2{}, i5{};
      rows2(plo, phi, add, window, ad2, i2);
      rows5(plo, phi, add, window, ad5, i5);
      VCHECK(std::memcmp(ad2, ad5, 32 * sizeof(uint64_t)) == 0);
      VCHECK_EQ(ad5[32], 0xfeedfacecafebeefull);
      VCHECK_EQ(i5.a0, i2.a0);
      VCHECK_EQ(i5.stride, i2.stride);
      VCHECK_EQ(i5.in_window, i2.in_window);
      VCHECK_EQ(i5.progression, i2.progression);
      // The definition, lane by lane.
      bool prog = true, win = false;
      for (unsigned l = 0; l < 32; ++l) {
        const uint64_t a = plo ? ((phi ? uint64_t{phi[l]} << 32 : 0) | plo[l]) + add : add;
        VCHECK_EQ(ad5[l], a);
        prog = prog && a == ad5[0] + l * (ad5[1] - ad5[0]);
        win = win || (a - window) >> 32 == 0;
      }
      VCHECK_EQ(i5.progression, prog);
      VCHECK_EQ(i5.in_window, win);
      progressions += prog;
      in_window += win;
      if (form == 2) continue;
      const bool r2 = run2(plo, phi, bytes), r5 = run5(plo, phi, bytes);
      VCHECK_EQ(r5, r2);
      bool want = true;
      for (unsigned l = 0; l < 32; ++l) {
        const uint64_t a = (phi ? uint64_t{phi[l]} << 32 : 0) | plo[l], first = (phi ? uint64_t{phi[0]} << 32 : 0) | plo[0];
        const uint64_t expect = first + uint64_t{l} * bytes;
        want = want && (phi ? a == expect : (a == (expect & 0xffffffffull) && expect >> 32 == 0));
      }
      VCHECK_EQ(r5, want);
      runs += r5;
      not_runs += !r5;
    }
  }
  VCHECK(progressions > 500 && runs > 200 && not_runs > 200 && in_window > 500);
}

VTEST(the_lane_address_functions_give_the_avx2_answers) {
  // pick_group_diverged, set_pcs_partial and select_pcs: the 512-bit forms against the 256-bit ones and the plain loops, on warps whose lanes
  // stand at few or many addresses, with every shape of the masks, addresses at both ends of the range, at any alignment.
  namespace sl = vgpu::exec::sasslanes;
  Gen g;
  const uint64_t ends[] = {0, 1, 0x7fffffffffffffffull, 0x8000000000000000ull, 0xffffffffffffffffull, 0xfffffffffffffffeull, 0x100000000ull, 0xffffffffull};
  for (unsigned round = 0; round < 20000; ++round) {
    alignas(64) uint64_t store[3][32 + 9];
    const unsigned off = round % 3 == 0 ? 0 : round % 8;
    uint64_t* pc[3] = {store[0] + off, store[1] + off, store[2] + off};
    const unsigned distinct = 1 + round % 6;
    uint64_t at[6];
    for (unsigned k = 0; k < distinct; ++k) at[k] = round % 4 == 0 ? ends[g.rng() % 8] : round % 4 == 1 ? 0x1000 + 16 * (g.rng() % 8) : g.rng();
    for (unsigned k = 0; k < 3; ++k) store[k][off + 32] = 0xfeedfacecafebeefull;
    for (unsigned l = 0; l < 32; ++l) pc[0][l] = pc[1][l] = pc[2][l] = at[g.rng() % distinct];
    uint32_t run = round % 5 == 0 ? 0xffffffffu : round % 5 == 1 ? 1u << (round % 32) : static_cast<uint32_t>(g.rng());
    if (round % 9 == 0) run &= static_cast<uint32_t>(g.rng());
    if (!run) run = 0x80000000u;
    const sl::Mask levels_run = run;
    uint64_t lo[3] = {7, 7, 7};
    sl::Mask grp[3] = {0, 0, 0};
    VCHECK(sl::testing::pick_group_at(Level::Base, pc[0], levels_run, lo[0], grp[0]));
    VCHECK(sl::testing::pick_group_at(Level::Avx2, pc[1], levels_run, lo[1], grp[1]));
    VCHECK(sl::testing::pick_group_at(Level::Avx512, pc[2], levels_run, lo[2], grp[2]));
    VCHECK_EQ(lo[2], lo[1]);
    VCHECK_EQ(grp[2], grp[1]);
    VCHECK_EQ(lo[2], lo[0]);
    VCHECK_EQ(grp[2], grp[0]);
    VCHECK(grp[2] != 0 && (grp[2] & ~levels_run) == 0);
    const sl::Mask group = round % 2 ? grp[2] : static_cast<sl::Mask>(static_cast<uint32_t>(g.rng()) & (round % 7 == 0 ? 0u : ~0u));
    const sl::Mask take = round % 6 == 0 ? group : round % 6 == 1 ? sl::Mask{0} : static_cast<sl::Mask>(static_cast<uint32_t>(g.rng()));
    const uint64_t target = g.rng(), next = round % 10 == 0 ? ends[g.rng() % 8] : g.rng();
    if (round % 2) {
      VCHECK(sl::testing::set_pcs_at(Level::Base, pc[0], group, target));
      VCHECK(sl::testing::set_pcs_at(Level::Avx2, pc[1], group, target));
      VCHECK(sl::testing::set_pcs_at(Level::Avx512, pc[2], group, target));
    } else {
      VCHECK(sl::testing::select_pcs_at(Level::Base, pc[0], group, take, target, next));
      VCHECK(sl::testing::select_pcs_at(Level::Avx2, pc[1], group, take, target, next));
      VCHECK(sl::testing::select_pcs_at(Level::Avx512, pc[2], group, take, target, next));
    }
    VCHECK(std::memcmp(pc[2], pc[1], 32 * sizeof(uint64_t)) == 0);
    VCHECK(std::memcmp(pc[2], pc[0], 32 * sizeof(uint64_t)) == 0);
    for (unsigned k = 0; k < 3; ++k) VCHECK_EQ(store[k][off + 32], 0xfeedfacecafebeefull);
  }
}

int main(int argc, char** argv) {
  if (!simd::supported(Level::Avx512) || !simd::supported(Level::Avx2)) {
    std::printf("SKIP: this host has no AVX-512 F/BW/DQ/VL/VNNI: the 512-bit lane kernels cannot be run against the AVX2 ones here\n");
    return 0;
  }
  return ::vtest::run_all(argc, argv);
}
