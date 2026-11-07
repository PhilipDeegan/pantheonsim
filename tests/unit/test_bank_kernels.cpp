// The kernels that count the bank passes of a warp that the closed forms and the table of shapes do not take (vgpu/exec/mem_counters.hpp,
// vgpu/simd/simd.hpp):
//
//   * simd::bank_window_kernel()      AVX2: every word the warp touches within 128 consecutive words, counted in registers (rows of 32 words,
//                                     a bit a bank, the bit-sliced maximum of four counts); null below AVX2, so the checks of it are skipped
//                                     under VGPU_SIMD=base (the test is also registered with that variable set);
//   * detail::bank_passes_max_lanes   1 to 4 byte lanes in one word each: the passes are the most lanes any bank has, when the bank that has that many has
//                                     that many different words (otherwise it declines);
//   * detail::bank_passes_slots       a bank keeps its first four distinct words; a fifth gives the warp up;
//   * sass_bank_passes_widened        the closed forms of a stride, then the two above, then the table;
//   * sass_bank_passes_offsets        all of it behind the table of shapes (`memo`), whatever state the table is in.
//
// Each is compared with a definition written from scratch here (the words the bytes of every active lane cover, the most distinct words any
// one of 32 banks is asked for), over the shapes the engines make (a window of words of every width, the partners of a bitonic network, strides, rows,
// lanes on one word, broadcasts), random warps in the millions, every mask of the structured ones, and the edges of every bound the kernels
// have: the 128 words of the window (spread 127 and 128), the fourth distinct word of a bank and the fifth, the first and last row, a lane whose
// bytes run into the next word at every byte, an offset within 16 of 2^32, and idle lanes that hold anything at all.
//
// The kernels may decline (return false, count nothing): what they must never do is count wrong, and decline a warp they are made for. So where a
// kernel returns true it must equal the definition, and it must return true exactly when the definition's own conditions (below) hold.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
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

// What the passes are, from the words (each lane covers the words its bytes cover; 64-bit arithmetic).
struct Def {
  uint32_t passes = 1;
  unsigned most_words = 0;     // the most distinct words in one bank
  uint64_t lowest = ~0ull, highest = 0;   // the lowest first word and the highest last word over the active lanes
};
Def definition(const uint32_t* o, uint32_t mask, uint32_t bytes) {
  std::array<uint64_t, 32 * 17> words;
  size_t n = 0;
  Def d;
  for (unsigned l = 0; l < 32; ++l) {
    if (!(mask >> l & 1)) continue;
    const uint64_t first = uint64_t{o[l]} / 4, last = (uint64_t{o[l]} + bytes - 1) / 4;
    d.lowest = std::min(d.lowest, first);
    d.highest = std::max(d.highest, last);
    for (uint64_t w = first; w <= last; ++w) words[n++] = w;
  }
  std::sort(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(n));
  n = static_cast<size_t>(std::unique(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(n)) - words.begin());
  unsigned per[32] = {};
  for (size_t i = 0; i < n; ++i) ++per[words[i] % 32];
  for (unsigned c : per) {
    d.most_words = std::max(d.most_words, c);
    d.passes = std::max<uint32_t>(d.passes, c);
  }
  return d;
}

bool have_window() { return vgpu::simd::bank_window_kernel() != nullptr; }

// One warp, every entry point. `idle` is what the lanes that do not run hold (they are in `o` as given: the caller filled them).
long g_window_taken = 0, g_window_declined = 0, g_slots_taken = 0, g_slots_declined = 0, g_max_taken = 0, g_max_declined = 0;

// When bank_passes_max_lanes must take a warp: 1 to 4 bytes, no lane runs into a second word, and the most lanes in a bank is 1, or at most 8 with a bank of that many
// lanes on that many different words.
bool max_lanes_expected(const uint32_t* o, uint32_t mask, uint32_t bytes) {
  if (bytes < 1 || bytes > 4) return false;
  unsigned lanes[32] = {};
  for (unsigned l = 0; l < 32; ++l) {
    if (!(mask >> l & 1)) continue;
    if ((o[l] & 3u) + bytes > 4u) return false;
    ++lanes[(o[l] / 4) % 32];
  }
  unsigned most = 0;
  for (unsigned c : lanes) most = std::max(most, c);
  if (most == 1) return true;
  if (most > 8) return false;
  for (unsigned b = 0; b < 32; ++b) {
    if (lanes[b] != most) continue;
    std::vector<uint32_t> words;
    for (unsigned l = 0; l < 32; ++l)
      if ((mask >> l & 1) && (o[l] / 4) % 32 == b) words.push_back(o[l] / 4);
    std::sort(words.begin(), words.end());
    if (std::adjacent_find(words.begin(), words.end()) == words.end()) return true;
  }
  return false;
}
void check(const uint32_t* o, uint32_t mask, uint32_t bytes) {
  if (!mask) return;
  const Def d = definition(o, mask, bytes);
  VCHECK_EQ(mc::sass_bank_passes_widened(o, mask, bytes), d.passes);
  {
    mc::BankMemo memo;
    VCHECK_EQ(mc::sass_bank_passes_offsets(memo, o, mask, bytes), d.passes);
    memo.skip32 = 10;   // (the table left alone: the kept warp, then the same counting)
    VCHECK_EQ(mc::sass_bank_passes_offsets(memo, o, mask, bytes), d.passes);
    VCHECK_EQ(mc::sass_bank_passes_offsets(memo, o, mask, bytes), d.passes);
  }
  uint32_t out = 12345;
  uint32_t top = 0;
  for (unsigned l = 0; l < 32; ++l)
    if (mask >> l & 1) top = std::max(top, o[l]);
  const bool no_wrap = top <= 0xFFFFFFF0u;
  // The slots: they take every warp whose banks hold at most four distinct words each (and no wrap, 1 to 16 bytes).
  {
    const bool expect = bytes >= 1 && bytes <= 16 && no_wrap && d.most_words <= 4;
    const bool took = mc::detail::bank_passes_slots(o, mask, bytes, out);
#if defined(__SSE2__)
    VCHECK_EQ(took, expect);
#endif
    if (took) {
      VCHECK_EQ(out, d.passes);
      ++g_slots_taken;
    } else {
      ++g_slots_declined;
    }
  }
  // The most lanes: every warp whose fullest bank has that many different words (1 to 4 bytes, nothing running into a second word).
  {
    out = 12345;
    const bool expect = max_lanes_expected(o, mask, bytes);
    const bool took = mc::detail::bank_passes_max_lanes(o, mask, bytes, out);
#if defined(__SSE2__)
    VCHECK_EQ(took, expect);
#endif
    if (took) {
      VCHECK_EQ(out, d.passes);
      ++g_max_taken;
    } else {
      ++g_max_declined;
      VCHECK_EQ(out, 12345u);
    }
  }
  // The AVX2 window: every warp whose words lie within 128 consecutive words (no wrap, 1 to 16 bytes).
  if (const auto k = vgpu::simd::bank_window_kernel()) {
    out = 12345;
    const bool expect = bytes >= 1 && bytes <= 16 && no_wrap && d.highest - d.lowest <= 127;
    const bool took = k(o, mask, bytes, out);
    VCHECK_EQ(took, expect);
    if (took) {
      VCHECK_EQ(out, d.passes);
      ++g_window_taken;
    } else {
      ++g_window_declined;
      VCHECK_EQ(out, 12345u);   // (declined: nothing written)
    }
  }
}

uint32_t random_mask(Rng& r) {
  switch (r.below(7)) {
    case 0: return 0xFFFFFFFFu;
    case 1: return 1u << r.below(32);
    case 2: return static_cast<uint32_t>(r.next()) | 1u;
    case 3: return (1u << (1 + r.below(31))) - 1;
    case 4: return 0xFFFFFFFFu << r.below(31);
    case 5: return r.below(2) ? 0x55555555u : 0xAAAAAAAAu;
    default: return static_cast<uint32_t>(r.next() & r.next()) | (1u << r.below(32));
  }
}

// Lanes that do not run hold anything (a register's old value): a word, the first lane's, zero or noise.
void fill_idle(Rng& r, uint32_t* o, uint32_t mask) {
  for (unsigned l = 0; l < 32; ++l) {
    if (mask >> l & 1) continue;
    switch (r.below(4)) {
      case 0: o[l] = 0; break;
      case 1: o[l] = static_cast<uint32_t>(r.next()); break;
      case 2: o[l] = 0xFFFFFFFFu; break;
      default: break;   // (as the family made it)
    }
  }
}

const uint32_t kWidths[] = {1, 2, 3, 4, 5, 6, 8, 12, 16};

}  // namespace

VTEST(random_warps_in_a_window_of_every_width) {
  Rng r(1);
  const uint64_t windows[] = {1, 2, 3, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 160, 256, 1000, 4000};
  for (int i = 0; i < 1500000; ++i) {
    const uint32_t bytes = kWidths[r.below(sizeof kWidths / sizeof kWidths[0])];
    const uint64_t win = windows[r.below(sizeof windows / sizeof windows[0])];
    const uint32_t base = r.below(3) == 0 ? static_cast<uint32_t>(r.below(256)) : r.below(2) ? 4 * static_cast<uint32_t>(32 * r.below(1 << 14)) : static_cast<uint32_t>(r.below(1 << 24));
    uint32_t o[32];
    for (auto& x : o) {
      x = base + 4 * static_cast<uint32_t>(r.below(win));
      if (bytes != 4 && r.below(2)) x += static_cast<uint32_t>(r.below(4));
    }
    const uint32_t mask = random_mask(r);
    fill_idle(r, o, mask);
    check(o, mask, bytes);
  }
}

VTEST(random_scatters_and_lanes_on_a_few_words) {
  Rng r(7);
  for (int i = 0; i < 1000000; ++i) {
    const uint32_t bytes = r.below(4) == 0 ? kWidths[r.below(sizeof kWidths / sizeof kWidths[0])] : static_cast<uint32_t>(1 + r.below(4));
    uint32_t o[32];
    const uint32_t base = r.below(2) ? 0u : static_cast<uint32_t>(r.below(1 << 20)) * 4;
    if (r.below(2)) {
      // a gather over a wide array (a histogram's bytes, a radix sort's digits)
      const uint64_t words = 64 + r.below(r.below(2) ? 1024 : 70000);
      for (auto& x : o) x = base + 4 * static_cast<uint32_t>(r.below(words)) + (bytes < 4 ? static_cast<uint32_t>(r.below(4 - bytes + 1)) : 0u);
    } else {
      // lanes on a few words (a pool of k of them, some in one bank)
      const unsigned k = 2 + static_cast<unsigned>(r.below(12));
      uint32_t pool[16];
      for (unsigned q = 0; q < k; ++q) pool[q] = base + 4 * (static_cast<uint32_t>(r.below(3)) * 32 + static_cast<uint32_t>(r.below(r.below(2) ? 3 : 40)) + 32 * static_cast<uint32_t>(r.below(6)));
      for (auto& x : o) x = pool[r.below(k)] + (bytes < 4 ? static_cast<uint32_t>(r.below(4 - bytes + 1)) : 0u);
    }
    const uint32_t mask = random_mask(r);
    fill_idle(r, o, mask);
    check(o, mask, bytes);
  }
  // The fullest bank: nine lanes (declined), eight (taken), eight with a repeated word in the first fullest bank and none in the second.
  for (unsigned lanes_in_bank : {1u, 2u, 3u, 7u, 8u, 9u, 10u, 32u}) {
    uint32_t o[32];
    for (unsigned l = 0; l < 32; ++l) o[l] = l < lanes_in_bank ? 4 * (5 + 32 * l) : 4 * (l - lanes_in_bank + 6 + 64 * 1000);   // (the rest, a bank each or sharing)
    check(o, 0xFFFFFFFFu, 4);
    for (unsigned l = 0; l < 32; ++l) o[l] = l < lanes_in_bank ? 4 * (5 + 32 * (l % 3)) : 4 * (l + 500);
    check(o, 0xFFFFFFFFu, 4);
  }
  for (unsigned dup = 0; dup < 4; ++dup) {
    uint32_t o[32];
    for (unsigned l = 0; l < 32; ++l) o[l] = 4 * (l + 6400);
    // bank 3 has four lanes, bank 9 has four lanes: bank 3's words are not all different (when dup), bank 9's are.
    unsigned q = 0;
    for (unsigned l = 0; l < 4; ++l) o[q++] = 4 * (3 + 32 * (l + 100 * 0 + (dup & 1 ? 0 : l)));
    for (unsigned l = 0; l < 4; ++l) o[q++] = 4 * (9 + 32 * (l + 7));
    if (dup & 2) o[1] = o[0];
    check(o, 0xFFFFFFFFu, 4);
    check(o, 0x0FFFFFFFu, 4);
  }
}

VTEST(the_edges_of_the_window_and_of_a_banks_four_words) {
  Rng r(2);
  // A spread of 127 words is inside the window and 128 is not, from every lowest word's place in its row, at every width.
  for (uint32_t lowest = 0; lowest < 70; ++lowest) {
    for (uint32_t spread : {0u, 1u, 30u, 31u, 32u, 63u, 64u, 65u, 95u, 96u, 126u, 127u, 128u, 129u, 191u, 192u, 200u}) {
      for (uint32_t bytes : {1u, 4u, 8u, 16u}) {
        for (int i = 0; i < 6; ++i) {
          uint32_t o[32];
          for (auto& x : o) x = 4096 + 4 * (lowest + static_cast<uint32_t>(r.below(spread + 1)));
          const unsigned hi = 1 + static_cast<unsigned>(r.below(31));
          o[0] = 4096 + 4 * lowest;   // (the lowest word's lane and the highest word's lane both run)
          o[hi] = 4096 + 4 * (lowest + spread);
          const uint32_t mask = random_mask(r) | 1u | 1u << hi;
          fill_idle(r, o, mask);
          check(o, mask, bytes);
        }
      }
    }
  }
  // Words of one bank: k distinct ones in rows 0..k-1 (k up to 6), each seen by one to three lanes, so the slots' fifth word and the window's four
  // rows are crossed; and the same with the rows' words one place apart (a bank each: one pass).
  for (unsigned k = 1; k <= 7; ++k) {
    for (unsigned bank = 0; bank < 32; bank += 7) {
      for (int i = 0; i < 20; ++i) {
        uint32_t o[32];
        for (unsigned l = 0; l < 32; ++l) o[l] = 4 * (bank + 32 * static_cast<uint32_t>(r.below(k)));
        // One lane per row at least, so that all k rows are present.
        for (unsigned q = 0; q < k; ++q) o[q] = 4 * (bank + 32 * q);
        check(o, 0xFFFFFFFFu, 4);
        check(o, static_cast<uint32_t>(r.next()) | ((1u << k) - 1), 4);
        // The same, one lane's bytes run into the next word (a second bank's word).
        o[k] += 2;
        check(o, 0xFFFFFFFFu, 4);
        check(o, 0xFFFFFFFFu, 8);
      }
    }
  }
  // Four rows share every bit, then three, then two; then one bank has a word in each of the rows 0, 1, 2 and 3 of the window and a second one in 0 and 3.
  for (unsigned rows = 1; rows <= 4; ++rows) {
    uint32_t o[32];
    for (unsigned l = 0; l < 32; ++l) o[l] = 4 * (l % 32) + 4 * 32 * (l % rows) + 8192;
    check(o, 0xFFFFFFFFu, 4);
    for (unsigned l = 0; l < 32; ++l) o[l] = 8192 + 4 * (3 + 32 * (l % rows));
    check(o, 0xFFFFFFFFu, 4);
    for (unsigned l = 0; l < 32; ++l) o[l] = 8192 + 4 * (3 + 32 * (l % rows)) + 4 * (l >= 16);
    check(o, 0xFFFFFFFFu, 4);
  }
}

VTEST(every_mask_of_the_shapes_kernels_make) {
  // The structured warps, each under every mask of its first sixteen lanes (the other sixteen full, empty, alternate, or the first lane's only).
  struct Shape {
    const char* name;
    uint32_t bytes;
    uint32_t (*off)(unsigned l);
  };
  static const Shape shapes[] = {
      {"unit4", 4, [](unsigned l) { return 4096u + 4 * l; }},
      {"stride2", 4, [](unsigned l) { return 4096u + 8 * l; }},
      {"stride33", 4, [](unsigned l) { return 4096u + 4 * 33 * l; }},
      {"bitonic4", 4, [](unsigned l) { return 4096u + 4 * (2 * l - (l & 3u)); }},
      {"bitonic16", 4, [](unsigned l) { return 4096u + 4 * (2 * l - (l & 15u)); }},
      {"pairs", 4, [](unsigned l) { return 4096u + 4 * (4 * (l / 2) + (l & 1)); }},
      {"two_rows16", 4, [](unsigned l) { return 4096u + 4 * ((l / 16) * 160 + (l & 15)); }},
      {"groups8_one_word", 4, [](unsigned l) { return 4096u + 4 * (l / 8); }},
      {"bytes_in_words", 1, [](unsigned l) { return 4096u + 64 * (l % 7) + (l / 7); }},
      {"eight_byte_pairs", 8, [](unsigned l) { return 4096u + 8 * (l / 4) + 8 * (l & 1) * 16; }},
      {"sixteen_byte", 16, [](unsigned l) { return 4096u + 16 * l; }},
      {"unaligned4", 4, [](unsigned l) { return 4096u + 4 * l + 2; }},
      {"unaligned8", 8, [](unsigned l) { return 4096u + 12 * l + 1; }},
      {"four_rows_one_bank", 4, [](unsigned l) { return 4096u + 4 * 32 * (l % 4) + 4 * (l / 16); }},
  };
  Rng r(3);
  for (const Shape& s : shapes) {
    uint32_t o[32];
    for (unsigned l = 0; l < 32; ++l) o[l] = s.off(l);
    for (uint32_t low = 0; low < 65536; ++low) {
      const uint32_t high = r.below(4) == 0 ? 0u : r.below(3) == 0 ? 0xAAAA0000u : r.below(2) ? 0xFFFF0000u : 0x00010000u;
      const uint32_t mask = low | high;
      if (!mask) continue;
      uint32_t p[32];
      for (unsigned l = 0; l < 32; ++l) p[l] = o[l];
      fill_idle(r, p, mask);
      check(p, mask, s.bytes);
    }
  }
}

VTEST(offsets_near_the_top_of_32_bits_are_counted_and_never_wrapped) {
  Rng r(4);
  for (int i = 0; i < 40000; ++i) {
    const uint32_t bytes = kWidths[r.below(sizeof kWidths / sizeof kWidths[0])];
    uint32_t o[32];
    const uint32_t top = 0xFFFFFFFFu - static_cast<uint32_t>(r.below(r.below(2) ? 40 : 4000));
    for (auto& x : o) x = top - static_cast<uint32_t>(r.below(40));
    const uint32_t mask = random_mask(r);
    fill_idle(r, o, mask);
    check(o, mask, bytes);
  }
  // One lane at the last offset the kernels take and one at the first they do not.
  for (uint32_t last : {0xFFFFFFF0u, 0xFFFFFFF1u, 0xFFFFFFF8u, 0xFFFFFFFFu}) {
    for (uint32_t bytes : {1u, 4u, 8u, 16u}) {
      uint32_t o[32];
      for (unsigned l = 0; l < 32; ++l) o[l] = 0xFFFFFF00u + 4 * l;
      o[31] = last;
      check(o, 0xFFFFFFFFu, bytes);
      check(o, 0x80000001u, bytes);
    }
  }
}

VTEST(widths_past_sixteen_bytes_and_wide_spreads_go_to_the_table_unchanged) {
  Rng r(5);
  for (int i = 0; i < 100000; ++i) {
    const uint32_t bytes = r.below(2) ? static_cast<uint32_t>(1 + r.below(64)) : static_cast<uint32_t>(17 + r.below(48));
    uint32_t o[32];
    const uint64_t win = r.below(2) ? 3000 : 90;
    for (auto& x : o) x = 8192 + 4 * static_cast<uint32_t>(r.below(win)) + static_cast<uint32_t>(r.below(4));
    const uint32_t mask = random_mask(r);
    fill_idle(r, o, mask);
    check(o, mask, bytes);
  }
}

VTEST(lanes_on_a_stride_keep_the_closed_forms) {
  // Every stride from -80 to 160 words' worth of bytes, widths 1 to 16, partial warps: the closed forms answer first; the answer is the definition's.
  Rng r(6);
  for (int stride = -330; stride <= 650; stride += (stride % 7 == 0 ? 1 : 3)) {
    for (uint32_t bytes : {1u, 2u, 4u, 8u, 16u}) {
      uint32_t o[32];
      const uint32_t base = 40000 + static_cast<uint32_t>(r.below(64));
      for (unsigned l = 0; l < 32; ++l) o[l] = base + static_cast<uint32_t>(stride * static_cast<int>(l) + (stride < 0 ? 12000 : 0));
      check(o, 0xFFFFFFFFu, bytes);
      check(o, 0x0000FFFFu, bytes);
      check(o, 0x5555AAAAu, bytes);
    }
  }
}

VTEST(the_avx2_form_of_the_offsets_function_is_the_scalar_one_state_for_state) {
  // Two tables fed the same calls: one through sass_bank_passes_offsets, the other through simd::bank_offsets_kernel (when the host has it; otherwise the
  // scalar function twice, which only checks the harness). After every call the answers are the definition's and the two tables' counters are equal; every
  // few hundred calls the whole tables (every entry, the last warp kept) are compared byte for byte. The calls: a handful of shapes asked again and again
  // (hits at the entry the instruction last used, hits elsewhere, evictions), random warps (misses, the table going quiet for a thousand calls and waking), partial
  // masks, widths 1 to 64, offsets with the top bit set, and a hint that is null, right, wrong, or past the table.
  Rng r(21);
  const auto kernel = vgpu::simd::bank_offsets_kernel();
  mc::BankMemo a, b;
  uint8_t hint_a[4] = {}, hint_b[4] = {};
  uint32_t shapes[24][32];
  uint32_t masks[24], widths[24];
  const auto make_shape = [&](unsigned q) {
    const unsigned kind = static_cast<unsigned>(r.below(6));
    for (unsigned l = 0; l < 32; ++l) {
      switch (kind) {
        case 0: shapes[q][l] = 4096 + 4 * l; break;
        case 1: shapes[q][l] = 4096 + 4 * (2 * l - (l & ((1u << (1 + q % 5)) - 1))); break;
        case 2: shapes[q][l] = 4096 + 4 * ((l / 16) * 160 + (l & 15)); break;
        case 3: shapes[q][l] = 4096 + 4 * static_cast<uint32_t>(r.below(70)); break;
        case 4: shapes[q][l] = 4096 + 4 * (l * 33) + static_cast<uint32_t>(r.below(4)); break;
        default: shapes[q][l] = 4096 + 128 * static_cast<uint32_t>(r.below(40)) + 4 * (l / 8); break;
      }
    }
    masks[q] = r.below(3) ? 0xFFFFFFFFu : random_mask(r);
    if (!masks[q]) masks[q] = 1;
    widths[q] = r.below(3) ? 4u : (r.below(2) ? static_cast<uint32_t>(1 + r.below(64)) : kWidths[r.below(sizeof kWidths / sizeof kWidths[0])]);
  };
  for (unsigned q = 0; q < 24; ++q) make_shape(q);
  for (int i = 0; i < 600000; ++i) {
    uint32_t o[32];
    uint32_t mask, bytes;
    const unsigned pick = static_cast<unsigned>(r.below(10));
    if (pick < 6) {
      const unsigned q = static_cast<unsigned>(r.below(24));
      const uint32_t shift = 4 * static_cast<uint32_t>(r.below(3) == 0 ? r.below(100) : 0) * 32 + (r.below(8) == 0 ? 4 * static_cast<uint32_t>(r.below(32)) : 0);
      for (unsigned l = 0; l < 32; ++l) o[l] = shapes[q][l] + shift;
      mask = masks[q];
      bytes = widths[q];
      if (r.below(30) == 0) make_shape(q);
    } else if (pick < 9) {
      bytes = r.below(2) ? 4u : static_cast<uint32_t>(1 + r.below(64));
      for (auto& x : o) x = 8192 + 4 * static_cast<uint32_t>(r.below(r.below(2) ? 1024 : 70)) + (r.below(3) == 0 ? static_cast<uint32_t>(r.below(4)) : 0u);
      mask = random_mask(r);
    } else {
      bytes = static_cast<uint32_t>(1 + r.below(70));
      // (offsets with bit 31 set go to the counting that widens them; those with only bit 30 set are in the table like any other)
      const uint32_t high_bits = r.below(4) == 0 ? 0u : r.below(2) ? 0x80000000u : r.below(2) ? 0x40000000u : 0xC0000000u;
      for (auto& x : o) x = (r.below(4) ? high_bits : 0u) + 4 * static_cast<uint32_t>(r.below(100));
      mask = random_mask(r);
    }
    if (!mask) mask = 1;
    fill_idle(r, o, mask);
    const uint32_t want = bytes >= 1 && bytes <= 64 ? definition(o, mask, bytes).passes : 0;
    uint8_t* ha = nullptr;
    uint8_t* hb = nullptr;
    switch (r.below(8)) {
      case 0: break;   // (no hint)
      case 1: hint_a[1] = hint_b[1] = static_cast<uint8_t>(r.below(256)); ha = hint_a + 1; hb = hint_b + 1; break;   // (anything, past the table)
      default: ha = hint_a + (i & 3); hb = hint_b + (i & 3); break;
    }
    const uint32_t pa = mc::sass_bank_passes_offsets(a, o, mask, bytes, ha);
    const uint32_t pb = kernel ? kernel(b, o, mask, bytes, hb) : mc::sass_bank_passes_offsets(b, o, mask, bytes, hb);
    if (bytes >= 1 && bytes <= 64) VCHECK_EQ(pa, want);
    VCHECK_EQ(pb, pa);
    VCHECK_EQ(b.skip32, a.skip32);
    VCHECK_EQ(b.last32, a.last32);
    VCHECK_EQ(b.lookups32, a.lookups32);
    VCHECK_EQ(b.hits32, a.hits32);
    VCHECK(std::memcmp(hint_a, hint_b, sizeof hint_a) == 0);
    if (i % 257 == 0) {
      for (unsigned e = 0; e < mc::BankMemo::kEntries32; ++e) {
        VCHECK_EQ(b.e32[e].bytes, a.e32[e].bytes);
        VCHECK_EQ(b.e32[e].passes, a.e32[e].passes);
        VCHECK(std::memcmp(b.e32[e].rel, a.e32[e].rel, sizeof a.e32[e].rel) == 0);
      }
      VCHECK(std::memcmp(b.last_offs, a.last_offs, sizeof a.last_offs) == 0);
      VCHECK_EQ(b.last_mask, a.last_mask);
      VCHECK_EQ(b.last_passes, a.last_passes);
    }
    if (i % 5000 == 4999) {   // (leave the table alone for a while, as a gather does)
      a.skip32 = b.skip32 = static_cast<uint32_t>(r.below(600));
    }
  }
  // The AVX2 form on its own with `fast` off is not reachable (the caller does not call it): VGPU_FASTCOUNT=0 is the caller's choice, checked by test_memfast.
}

VTEST(the_kernels_were_taken_and_declined_both_ways) {
  // (This test runs last of the random ones above: both outcomes must have happened many times, or the tests above did not reach the kernels.)
  VCHECK(g_max_taken > 100000);
  VCHECK(g_max_declined > 100000);
  VCHECK(g_slots_taken > 100000);
  VCHECK(g_slots_declined > 1000);
  if (have_window()) {
    VCHECK(g_window_taken > 100000);
    VCHECK(g_window_declined > 100000);
  } else {
    VCHECK_EQ(g_window_taken, 0L);
  }
}

VTEST_MAIN
