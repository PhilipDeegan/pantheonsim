// The AVX2 counting of the sectors of a warp whose lanes' sectors never go backwards (vgpu/simd/simd.hpp, simd::sector_up_kernel; memcount's
// sectors_monotone<true> is the scalar form), against a definition written from scratch here (the set of 32-byte sectors the bytes of every active lane
// touch), and through the entry points the engines call (distinct_sectors_uncapped, distinct_sectors).
//
// What the kernel promises: for lanes that are one run of consecutive lanes, a width of 1 to 32 bytes, addresses under 2^62, and lanes whose first
// sector is never below the last sector of the lane before, it counts, and the count is the definition's. Anything else it declines (false, `out`
// untouched) and the scalar forms count. Both halves are checked: every warp the kernel must take (it takes it), and every one it must not
// (it declines), so a kernel that went quietly off for some shape, or took a warp it did not understand, fails here. The kernel is null below AVX2
// (VGPU_SIMD=base; this test is also registered under that variable) and the entry points are checked either way.
#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include "vgpu/exec/mem_counters.hpp"
#include "vgpu/simd/simd.hpp"
#include "vtest.hpp"

namespace mc = vgpu::exec::memcount;

namespace {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  uint64_t below(uint64_t n) { return next() % n; }
};

// The sectors, from the bytes (unsigned 64-bit arithmetic that does not wrap for any address the kernel takes; for the others the definition is only used
// where the scalar entry points agree with it: addresses a lane's last byte does not wrap past 2^64).
uint64_t definition(const uint64_t* a, uint64_t m, unsigned lanes, uint64_t span) {
  std::set<uint64_t> sectors;
  for (unsigned l = 0; l < lanes; ++l)
    if (m >> l & 1)
      for (uint64_t s = a[l] / 32; s <= (a[l] + span - 1) / 32; ++s) sectors.insert(s);
  return sectors.size();
}

// Whether the kernel must take the warp.
bool must_take(const uint64_t* a, uint64_t m, uint64_t span) {
  if (span < 1 || span > 32 || m == 0) return false;
  const unsigned lo = static_cast<unsigned>(__builtin_ctzll(m));
  const uint64_t run = m >> lo;
  if (run & (run + 1)) return false;
  const unsigned n = 64 - static_cast<unsigned>(__builtin_clzll(run));
  bool have_prev = false;
  uint64_t prev = 0;
  for (unsigned i = 0; i < n; ++i) {
    const uint64_t x = a[lo + i];
    if (x >> 62) return false;
    const uint64_t f = x >> 5, e = (x + span - 1) >> 5;
    if (have_prev && f < prev) return false;
    have_prev = true;
    prev = e;
  }
  return true;
}

long g_taken = 0, g_declined = 0;

void check(const uint64_t* a, uint64_t m, uint64_t span) {
  if (!m) return;
  const bool wraps = [&] {
    for (unsigned l = 0; l < 64; ++l)
      if ((m >> l & 1) && a[l] + span - 1 < a[l]) return true;
    return false;
  }();
  if (const auto k = vgpu::simd::sector_up_kernel()) {
    uint64_t out = 777;
    const bool took = k(a, m, span, out);
    VCHECK_EQ(took, must_take(a, m, span));
    if (took) {
      VCHECK_EQ(out, definition(a, m, 64, span));
      ++g_taken;
    } else {
      VCHECK_EQ(out, 777ull);
      ++g_declined;
    }
  }
  if (!wraps && span >= 1 && span <= 256) {
    const uint64_t want = definition(a, m, 64, span);
    VCHECK_EQ(mc::distinct_sectors_uncapped(a, m, 64, span), static_cast<uint32_t>(want));
    VCHECK_EQ(mc::distinct_sectors(a, m, 64, span, SIZE_MAX), static_cast<uint32_t>(want));
  }
}

uint64_t random_run_mask(Rng& r, unsigned lanes) {
  const unsigned n = 1 + static_cast<unsigned>(r.below(lanes));
  const unsigned lo = static_cast<unsigned>(r.below(lanes - n + 1));
  return (n >= 64 ? ~0ull : ((1ull << n) - 1)) << lo;
}

uint64_t random_mask(Rng& r, unsigned lanes) {
  switch (r.below(6)) {
    case 0: return (lanes >= 64 ? ~0ull : ((1ull << lanes) - 1));
    case 1: return random_run_mask(r, lanes);
    case 2: return random_run_mask(r, lanes);
    case 3: return (r.next() | 1) & (lanes >= 64 ? ~0ull : ((1ull << lanes) - 1));
    case 4: return 0x5555555555555555ull & (lanes >= 64 ? ~0ull : ((1ull << lanes) - 1));
    default: return (1ull << r.below(lanes));
  }
}

}  // namespace

VTEST(runs_of_consecutive_words_in_two_to_four_segments) {
  // A block-partitioned access: k segments of lanes on consecutive elements, the segments far apart, adjacent, overlapping or sharing a sector.
  Rng r(1);
  for (int i = 0; i < 600000; ++i) {
    const uint64_t span = r.below(3) ? 4 : (r.below(2) ? 8 : 1 + r.below(32));
    uint64_t a[64];
    const unsigned lanes = r.below(8) ? 32 : 64;
    const uint64_t base = r.below(4) == 0 ? r.below(200) : (r.below(2) ? 0x7000'0000'0000ull : (1ull << 33) + r.below(1 << 20));
    unsigned l = 0;
    uint64_t at = base + (r.below(2) ? r.below(32) : 0);
    while (l < lanes) {
      const unsigned len = 1 + static_cast<unsigned>(r.below(lanes));
      const uint64_t step = r.below(5) ? span : span * (1 + r.below(3));
      for (unsigned j = 0; j < len && l < lanes; ++j, ++l) a[l] = at + j * step;
      // the next segment: below, in the same sector, just after, or far
      switch (r.below(5)) {
        case 0: at = at + (r.below(2) ? 64 : 12288) + r.below(32); break;
        case 1: at = at + len * step - span + r.below(span); break;   // (overlaps the end of this one)
        case 2: at = at + len * step; break;
        case 3: at = at > 1000 ? at - 1000 : at + 7; break;           // (goes back: declined)
        default: at = at + len * step + 32 * r.below(40); break;
      }
    }
    const uint64_t m = random_mask(r, lanes);
    if (lanes == 64) for (unsigned q = 32; q < 64; ++q) if (!(m >> q & 1)) a[q] = r.next();
    for (unsigned q = 0; q < lanes; ++q) if (!(m >> q & 1) && r.below(2)) a[q] = r.next() >> r.below(40);   // (idle lanes hold anything)
    check(a, m, span);
  }
}

VTEST(strided_and_jittered_ascending_gathers) {
  Rng r(2);
  for (int i = 0; i < 500000; ++i) {
    uint64_t a[64];
    const uint64_t span = r.below(2) ? 4 : 1 + r.below(32);
    const uint64_t base = r.below(3) == 0 ? r.below(100) : 0x7000'0000'0000ull + r.below(1 << 24);
    const uint64_t stride = r.below(4) == 0 ? 0 : 1 + r.below(r.below(2) ? 40 : 700);
    uint64_t at = base;
    for (unsigned l = 0; l < 64; ++l) {
      a[l] = at;
      at += stride + (r.below(3) == 0 ? r.below(40) : 0);
      if (r.below(40) == 0 && at > 64) at -= 1 + r.below(64);   // (a lane that steps back, sometimes inside its own sector)
    }
    check(a, random_mask(r, 32), span);
    check(a, random_mask(r, 64), span);
  }
}

VTEST(the_first_lanes_sector_is_counted_whatever_sector_it_is) {
  // The lane with nothing before it adds its own sectors: sector 0 included (a stand-in of 0 for "none" would lose one), and the last sector of the
  // address space the kernel takes.
  Rng r(3);
  for (uint64_t first : {0ull, 1ull, 31ull, 32ull, 33ull, 63ull, 64ull, 4095ull, (1ull << 62) - 64, (1ull << 62) - 33, (1ull << 62) - 32, (1ull << 62) - 1, 1ull << 62, (1ull << 62) + 5}) {
    for (uint64_t span : {1ull, 2ull, 4ull, 8ull, 16ull, 31ull, 32ull, 33ull}) {
      for (unsigned n : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 16u, 17u, 31u, 32u}) {
        uint64_t a[64];
        for (unsigned l = 0; l < 64; ++l) a[l] = first + l * span + (r.below(2) ? 0 : r.below(3));
        check(a, ((1ull << n) - 1) << r.below(33 - n), span);
        check(a, (1ull << n) - 1, span);
      }
    }
  }
  // Lanes that do not run, in front, between and after the run, hold addresses that would break the count if they were read.
  for (int i = 0; i < 20000; ++i) {
    uint64_t a[64];
    for (auto& x : a) x = r.next() >> r.below(50);
    const unsigned n = 1 + static_cast<unsigned>(r.below(32)), lo = static_cast<unsigned>(r.below(33 - n));
    for (unsigned l = 0; l < n; ++l) a[lo + l] = 0x1000 + 4 * l + 12288 * (l / 16);
    check(a, ((1ull << n) - 1) << lo, 4);
  }
}

VTEST(every_run_of_lanes_and_every_mask_of_a_few) {
  // Every run [lo, lo + n) of a 32-lane and a 64-lane warp on a rising address pattern with a jump, at every width that matters; every mask of 12 lanes.
  Rng r(4);
  for (unsigned lanes : {32u, 64u}) {
    for (unsigned lo = 0; lo < lanes; ++lo) {
      for (unsigned n = 1; lo + n <= lanes; ++n) {
        for (uint64_t span : {1ull, 4ull, 16ull, 32ull}) {
          uint64_t a[64];
          for (unsigned l = 0; l < 64; ++l) a[l] = 0x10000 + span * l + (l >= lo + n / 2 ? 4096 : 0) + 5;
          check(a, ((n >= 64 ? ~0ull : (1ull << n) - 1)) << lo, span);
        }
      }
    }
  }
  for (uint64_t mask = 1; mask < (1u << 12); ++mask) {
    uint64_t a[64];
    for (unsigned l = 0; l < 64; ++l) a[l] = 0x10000 + 8 * l + (l >= 6 ? 4096 : 0);
    check(a, mask << (mask & 3), 8);
    for (unsigned l = 0; l < 64; ++l) a[l] = 0x10000 + 4096 * (l % 3) + 4 * l;
    check(a, mask, 4);
  }
}

VTEST(lanes_that_go_backwards_even_a_little_are_declined_and_counted) {
  Rng r(5);
  for (int i = 0; i < 300000; ++i) {
    uint64_t a[64];
    const uint64_t span = r.below(2) ? 4 : 1 + r.below(32);
    uint64_t at = 0x7000'0000'0000ull + r.below(1000);
    for (unsigned l = 0; l < 32; ++l) {
      a[l] = at;
      at += r.below(3) ? span : span * (1 + r.below(4));
    }
    // one pair of lanes swapped, one lane moved back by a byte, a sector, or many sectors
    const unsigned x = 1 + static_cast<unsigned>(r.below(31));
    switch (r.below(4)) {
      case 0: std::swap(a[x], a[x - 1]); break;
      case 1: a[x] = a[x - 1] > 0 ? a[x - 1] - 1 : a[x - 1]; break;
      case 2: a[x] = a[x - 1] >= span + 32 ? a[x - 1] - span - 32 : a[x - 1]; break;
      default: a[x] = a[x - 1] >= 1 << 14 ? a[x - 1] - (1 << 14) : a[x]; break;
    }
    check(a, random_run_mask(r, 32) | (r.below(2) ? 0xFFFFFFFFull : 0), span);
  }
}

VTEST(widths_outside_the_kernel_and_addresses_near_the_top_go_to_the_scalar_forms) {
  Rng r(6);
  for (int i = 0; i < 200000; ++i) {
    uint64_t a[64];
    const uint64_t span = r.below(2) ? 33 + r.below(300) : 1 + r.below(32);
    const uint64_t base = r.below(3) == 0 ? (1ull << 62) - 200 + r.below(400) : r.below(2) ? ~0ull - 5000 + r.below(4000) : r.below(1 << 20);
    for (unsigned l = 0; l < 32; ++l) a[l] = base + l * (1 + r.below(2) * 3) * 4;
    check(a, random_run_mask(r, 32), span);
  }
}

VTEST(the_kernel_was_taken_and_declined_many_times) {
  if (vgpu::simd::sector_up_kernel()) {
    VCHECK(g_taken > 200000);
    VCHECK(g_declined > 200000);
  } else {
    VCHECK_EQ(g_taken, 0L);
  }
}

VTEST_MAIN
