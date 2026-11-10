// The 32-lane ALU kernels of the SASS tier (simd::Kernels::iadd3, imad, lop3, ffma, fadd, fmul, icmp_mask, blend_m, iadd3x, imad_wide, mufu, dop_m) and the two
// kernels of a full warp's global addresses (warp_rows, warp_run) at the AVX-512 level: sixteen lanes to a step where kernels.inc takes eight
// (eight 64-bit lanes where it takes four), and mask registers where it builds compare vectors. simd.cpp includes this file
// inside namespace avx512_impl, under the same #pragma GCC target as kernels512.inc, and make_avx512 puts these entries in the Avx512 table;
// VGPU_SIMD=avx2 keeps the 256-bit ones (docs/perf-design/rounds/11-rounds-sp36-sp41.md, Round sp50, has what this measured).
//
// Every kernel gives the bits of the AVX2 one: tests/unit/test_lanes512.cpp runs each against the Avx2 table's entry on a host that has both
// (and reports itself skipped on a host that has not). The same operations at another width: wrapping integer sums and products, one IEEE
// operation a lane in the host's rounding mode (a fused multiply-add for ffma, never for the others: contraction is off for the whole file,
// simd.cpp's pragma), and the canonical NaN 0x7fffffff for any NaN result.
//
// Aliasing is as in kernels.inc: a block of lanes is read from every source before it is written, so the result row may be one of the source
// rows (the tier's tier_out); the blocks here are 16 lanes, which changes nothing for a row that is the same row or another one.

#define VGPU_L512_LOAD(p) _mm512_loadu_si512(reinterpret_cast<const void*>(p))
#define VGPU_L512_STORE(p, v) _mm512_storeu_si512(reinterpret_cast<void*>(p), (v))

// A float result as the register holds it: a NaN is the canonical 0x7fffffff, whatever its payload or sign.
[[gnu::always_inline]] inline __m512i canon_nan512(__m512 v) {
  return _mm512_mask_mov_epi32(_mm512_castps_si512(v), _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q), _mm512_set1_epi32(0x7fffffff));
}

void iadd3_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t add) {
  const __m512i k = _mm512_set1_epi32(static_cast<int>(add));
  const __m512i x0 = VGPU_L512_LOAD(a), y0 = VGPU_L512_LOAD(b), z0 = VGPU_L512_LOAD(c);
  const __m512i v0 = _mm512_add_epi32(_mm512_add_epi32(x0, y0), _mm512_add_epi32(z0, k));
  VGPU_L512_STORE(r, v0);
  const __m512i x1 = VGPU_L512_LOAD(a + 16), y1 = VGPU_L512_LOAD(b + 16), z1 = VGPU_L512_LOAD(c + 16);
  const __m512i v1 = _mm512_add_epi32(_mm512_add_epi32(x1, y1), _mm512_add_epi32(z1, k));
  VGPU_L512_STORE(r + 16, v1);
}

void imad_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
  for (unsigned i = 0; i < 32; i += 16) {
    const __m512i x = VGPU_L512_LOAD(a + i), y = VGPU_L512_LOAD(b + i), z = VGPU_L512_LOAD(c + i);
    VGPU_L512_STORE(r + i, _mm512_add_epi32(_mm512_mullo_epi32(x, y), z));
  }
}

#define VGPU_L512_LOP3(EXPR)                                                                        \
  do {                                                                                              \
    for (unsigned i = 0; i < 32; i += 16) {                                                         \
      const __m512i x = VGPU_L512_LOAD(a + i), y = VGPU_L512_LOAD(b + i), z = VGPU_L512_LOAD(c + i); \
      (void)x; (void)y; (void)z;                                                                    \
      VGPU_L512_STORE(r + i, (EXPR));                                                               \
    }                                                                                               \
    return;                                                                                         \
  } while (0)

// The three-input logic function whose truth table is `lut` (bit 4a + 2b + c): the instruction that takes the table itself (vpternlogd reads
// its table with the first operand as the high bit, as LOP3 does) for the tables compilers use most, since the table is an immediate of the
// instruction, and the sum of minterms for the rest, as kernels.inc.
void lop3_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t lut) {
  switch (lut) {
#define VGPU_L512_T(T) case T: VGPU_L512_LOP3(_mm512_ternarylogic_epi32(x, y, z, T));
    VGPU_L512_T(0xc0) VGPU_L512_T(0x3c) VGPU_L512_T(0xfc) VGPU_L512_T(0xf0) VGPU_L512_T(0xcc) VGPU_L512_T(0xaa) VGPU_L512_T(0x0f)
    VGPU_L512_T(0x33) VGPU_L512_T(0x55) VGPU_L512_T(0x96) VGPU_L512_T(0x80) VGPU_L512_T(0xfe) VGPU_L512_T(0xf8) VGPU_L512_T(0xe0)
    VGPU_L512_T(0xea) VGPU_L512_T(0xe8) VGPU_L512_T(0x30) VGPU_L512_T(0x0c) VGPU_L512_T(0xc3)
#undef VGPU_L512_T
    default: break;
  }
  // Each set bit t of the table is one minterm of (a, b, c): the lanes where a, b, c equal t's bits.
  __m512i ma[8], mb[8], mc[8];
  unsigned n = 0;
  for (unsigned t = 0; t < 8; ++t)
    if ((lut >> t) & 1) {
      ma[n] = _mm512_set1_epi32((t & 4) ? 0 : -1);
      mb[n] = _mm512_set1_epi32((t & 2) ? 0 : -1);
      mc[n] = _mm512_set1_epi32((t & 1) ? 0 : -1);
      ++n;
    }
  for (unsigned i = 0; i < 32; i += 16) {
    const __m512i x = VGPU_L512_LOAD(a + i), y = VGPU_L512_LOAD(b + i), z = VGPU_L512_LOAD(c + i);
    __m512i v = _mm512_setzero_si512();
    // (x ^ ma) & (y ^ mb) & (z ^ mc), ORed in.
    for (unsigned m = 0; m < n; ++m)
      v = _mm512_or_si512(v, _mm512_and_si512(_mm512_and_si512(_mm512_xor_si512(x, ma[m]), _mm512_xor_si512(y, mb[m])), _mm512_xor_si512(z, mc[m])));
    VGPU_L512_STORE(r + i, v);
  }
}

void ffma_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
  for (unsigned i = 0; i < 32; i += 16) {
    const __m512 x = _mm512_castsi512_ps(VGPU_L512_LOAD(a + i)), y = _mm512_castsi512_ps(VGPU_L512_LOAD(b + i)),
                 z = _mm512_castsi512_ps(VGPU_L512_LOAD(c + i));
    VGPU_L512_STORE(r + i, canon_nan512(_mm512_fmadd_ps(x, y, z)));
  }
}

void fadd_k(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  for (unsigned i = 0; i < 32; i += 16) {
    const __m512 x = _mm512_castsi512_ps(VGPU_L512_LOAD(a + i)), y = _mm512_castsi512_ps(VGPU_L512_LOAD(b + i));
    VGPU_L512_STORE(r + i, canon_nan512(_mm512_add_ps(x, y)));
  }
}

void fmul_k(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  for (unsigned i = 0; i < 32; i += 16) {
    const __m512 x = _mm512_castsi512_ps(VGPU_L512_LOAD(a + i)), y = _mm512_castsi512_ps(VGPU_L512_LOAD(b + i));
    VGPU_L512_STORE(r + i, canon_nan512(_mm512_mul_ps(x, y)));
  }
}

// Bit l of the result is `a[l] cmp b[l]` (cmp 0 Eq, 1 Ne, 2 Lt, 3 Le, 4 Gt, 5 Ge; signed or unsigned), n a multiple of 32, at most 64: the
// compare writes the lane bits to a mask register, sixteen at a time.
template <int Pred, bool Sgn>
[[gnu::always_inline]] inline uint64_t icmp512_t(const uint32_t* a, const uint32_t* b, unsigned n) {
  uint64_t out = 0;
  for (unsigned i = 0; i < n; i += 16) {
    const __m512i x = VGPU_L512_LOAD(a + i), y = VGPU_L512_LOAD(b + i);
    const __mmask16 m = Sgn ? _mm512_cmp_epi32_mask(x, y, Pred) : _mm512_cmp_epu32_mask(x, y, Pred);
    out |= uint64_t{m} << i;
  }
  return out;
}

uint64_t icmp_mask_k(unsigned cmp, bool sgn, const uint32_t* a, const uint32_t* b, unsigned n) {
  switch (cmp * 2 + (sgn ? 1 : 0)) {
    case 0: case 1: return icmp512_t<_MM_CMPINT_EQ, false>(a, b, n);
    case 2: case 3: return icmp512_t<_MM_CMPINT_NE, false>(a, b, n);
    case 4: return icmp512_t<_MM_CMPINT_LT, false>(a, b, n);
    case 5: return icmp512_t<_MM_CMPINT_LT, true>(a, b, n);
    case 6: return icmp512_t<_MM_CMPINT_LE, false>(a, b, n);
    case 7: return icmp512_t<_MM_CMPINT_LE, true>(a, b, n);
    case 8: return icmp512_t<_MM_CMPINT_NLE, false>(a, b, n);
    case 9: return icmp512_t<_MM_CMPINT_NLE, true>(a, b, n);
    case 10: return icmp512_t<_MM_CMPINT_NLT, false>(a, b, n);
    default: return icmp512_t<_MM_CMPINT_NLT, true>(a, b, n);
  }
}

// d[l] = r[l] for the lanes of `mask`; the others keep their value. A blend and a whole-vector store, not a masked store: the next kernel
// loads the row whole, and a load is served from a store only when the store wrote every byte it reads.
void blend_m_k(uint32_t* d, const uint32_t* r, uint32_t mask) {
  const __mmask16 m0 = static_cast<__mmask16>(mask), m1 = static_cast<__mmask16>(mask >> 16);
  VGPU_L512_STORE(d, _mm512_mask_loadu_epi32(VGPU_L512_LOAD(d), m0, r));
  VGPU_L512_STORE(d + 16, _mm512_mask_loadu_epi32(VGPU_L512_LOAD(d + 16), m1, r + 16));
}

// r[l] = a + b + c + add + bit l of cin_a + bit l of cin_b, wrapping; *ge1 / *ge2: the lanes whose sum, taken in 64 bits, reached 2^32 / 2 * 2^32.
// The addend and the two carries in are one 32-bit number a lane, as kernels.inc builds it (add - (-1) - (-1), wrapping), and the high word of
// the 64-bit sum of four 32-bit numbers is the number of the three 32-bit additions that wrapped: each one's carry is a lane bit in a mask
// register (the sum below the number added to), at least one of them for *ge1, at least two for *ge2.
void iadd3x_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t add, uint32_t cin_a, uint32_t cin_b, uint32_t* ge1,
              uint32_t* ge2) {
  uint32_t o1 = 0, o2 = 0;
  const __m512i vadd = _mm512_set1_epi32(static_cast<int>(add)), one = _mm512_set1_epi32(1);
  for (unsigned i = 0; i < 32; i += 16) {
    __m512i k = _mm512_mask_add_epi32(vadd, static_cast<__mmask16>(cin_a >> i), vadd, one);
    k = _mm512_mask_add_epi32(k, static_cast<__mmask16>(cin_b >> i), k, one);
    const __m512i x = VGPU_L512_LOAD(a + i), y = VGPU_L512_LOAD(b + i), z = VGPU_L512_LOAD(c + i);
    const __m512i s1 = _mm512_add_epi32(x, y);
    const __m512i s2 = _mm512_add_epi32(s1, z);
    const __m512i s3 = _mm512_add_epi32(s2, k);
    const uint32_t c1 = _mm512_cmp_epu32_mask(s1, x, _MM_CMPINT_LT), c2 = _mm512_cmp_epu32_mask(s2, z, _MM_CMPINT_LT),
                   c3 = _mm512_cmp_epu32_mask(s3, k, _MM_CMPINT_LT);
    VGPU_L512_STORE(r + i, s3);
    o1 |= (c1 | c2 | c3) << i;
    o2 |= ((c1 & c2) | (c1 & c3) | (c2 & c3)) << i;
  }
  *ge1 = o1;
  *ge2 = o2;
}

// ---- binary64 lanes held as register pairs (dop_m) ---------------------------------------------------------------------------------------

// Sixteen lanes at a time: the low words and the high words of the 16 lanes are interleaved into two vectors of eight doubles in lane order (one
// two-source permute each: lanes 0 to 7 and lanes 8 to 15), and the results' words are taken apart again by two more. The arithmetic and the NaN
// rule are those of kernels.inc's dop_avx2_t: one IEEE operation a lane in the host's rounding mode (a fused multiply-add for kind 2), the |x|
// and -x modifiers on the high words before it, and a lane whose result is a NaN and that had a NaN operand gets simd::dop_nan of the operands
// as the registers hold them. Only the lanes of `mask` are written (a masked store, as there), and a block of 16 lanes is read whole before it
// is written, so the result rows may be source rows.
template <unsigned Kind, bool Mods>
[[gnu::always_inline]] inline void dop512_t(const uint32_t* absm, const uint32_t* negm, uint32_t* dlo, uint32_t* dhi, const uint32_t* alo,
                                            const uint32_t* ahi, const uint32_t* blo, const uint32_t* bhi, const uint32_t* clo, const uint32_t* chi,
                                            uint32_t mask) {
  const __m512i first = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
  const __m512i second = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
  const __m512i evens = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
  const __m512i odds = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
  const auto load = [&](const uint32_t* lo, const uint32_t* hi, unsigned k, unsigned i, bool mods, __m512d& p, __m512d& q) {
    const __m512i l = VGPU_L512_LOAD(lo + i);
    __m512i h = VGPU_L512_LOAD(hi + i);
    if (mods) h = _mm512_xor_si512(_mm512_and_si512(h, _mm512_set1_epi32(static_cast<int>(absm[k]))), _mm512_set1_epi32(static_cast<int>(negm[k])));
    p = _mm512_castsi512_pd(_mm512_permutex2var_epi32(l, first, h));
    q = _mm512_castsi512_pd(_mm512_permutex2var_epi32(l, second, h));
  };
  const auto is_nan = [](uint64_t u) { return (u & 0x7fffffffffffffffull) > 0x7ff0000000000000ull; };
  for (unsigned i = 0; i < 32; i += 16) {
    const __mmask16 bits = static_cast<__mmask16>(mask >> i);
    __m512d xa, xb, ya, yb, za = _mm512_setzero_pd(), zb = _mm512_setzero_pd();
    load(alo, ahi, 0, i, Mods, xa, xb);
    load(blo, bhi, 1, i, Mods, ya, yb);
    __m512d va, vb;
    if constexpr (Kind == 0) {
      va = _mm512_add_pd(xa, ya);
      vb = _mm512_add_pd(xb, yb);
    } else if constexpr (Kind == 1) {
      va = _mm512_mul_pd(xa, ya);
      vb = _mm512_mul_pd(xb, yb);
    } else {
      load(clo, chi, 2, i, Mods, za, zb);
      va = _mm512_fmadd_pd(xa, ya, za);
      vb = _mm512_fmadd_pd(xb, yb, zb);
    }
    if (_mm512_cmp_pd_mask(va, va, _CMP_UNORD_Q) | _mm512_cmp_pd_mask(vb, vb, _CMP_UNORD_Q)) [[unlikely]] {
      // Rare: a NaN result. The operands as the registers hold them (reloaded without the modifiers) and the results, in lane order, and the
      // rule applied to each lane that had a NaN operand.
      alignas(64) uint64_t X[16], Y[16], Z[16], R[16];
      if constexpr (Mods) {
        load(alo, ahi, 0, i, false, xa, xb);
        load(blo, bhi, 1, i, false, ya, yb);
        if constexpr (Kind == 2) load(clo, chi, 2, i, false, za, zb);
      }
      _mm512_store_pd(X, xa), _mm512_store_pd(X + 8, xb);
      _mm512_store_pd(Y, ya), _mm512_store_pd(Y + 8, yb);
      _mm512_store_pd(Z, za), _mm512_store_pd(Z + 8, zb);
      _mm512_store_pd(R, va), _mm512_store_pd(R + 8, vb);
      for (unsigned k = 0; k < 16; ++k) {
        const uint64_t uc = Kind == 2 ? Z[k] : 0;
        if (is_nan(X[k]) || is_nan(Y[k]) || (Kind == 2 && is_nan(uc))) R[k] = dop_nan(Kind, X[k], Y[k], uc);
      }
      va = _mm512_load_pd(R);
      vb = _mm512_load_pd(R + 8);
    }
    const __m512i rlo = _mm512_permutex2var_epi32(_mm512_castpd_si512(va), evens, _mm512_castpd_si512(vb));
    const __m512i rhi = _mm512_permutex2var_epi32(_mm512_castpd_si512(va), odds, _mm512_castpd_si512(vb));
    if (bits == 0xffffu) {
      VGPU_L512_STORE(dlo + i, rlo);
      VGPU_L512_STORE(dhi + i, rhi);
    } else if (bits) {
      _mm512_mask_storeu_epi32(dlo + i, bits, rlo);
      _mm512_mask_storeu_epi32(dhi + i, bits, rhi);
    }
  }
}

void dop_m_k(unsigned kind, unsigned mods, uint32_t* dlo, uint32_t* dhi, const uint32_t* alo, const uint32_t* ahi, const uint32_t* blo,
             const uint32_t* bhi, const uint32_t* clo, const uint32_t* chi, uint32_t mask) {
  const uint32_t absm[3] = {(mods & 1) ? 0x7fffffffu : ~0u, (mods & 4) ? 0x7fffffffu : ~0u, (mods & 16) ? 0x7fffffffu : ~0u};
  const uint32_t negm[3] = {(mods & 2) ? 0x80000000u : 0u, (mods & 8) ? 0x80000000u : 0u, (mods & 32) ? 0x80000000u : 0u};
  if ((mods & 63u) == 0) {
    if (kind == 0) dop512_t<0, false>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
    else if (kind == 1) dop512_t<1, false>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
    else dop512_t<2, false>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
  } else {
    if (kind == 0) dop512_t<0, true>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
    else if (kind == 1) dop512_t<1, true>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
    else dop512_t<2, true>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
  }
}

// ---- the addresses of a full warp's global access (warp_rows, warp_run) ---------------------------------------------------------------------

// sassmem::full_warp_rows, eight lanes to a step (avx2_impl::warp_rows_body takes four): the same addresses stored, and the same four answers.
// in_window is "some lane's (address - window) has a zero high half" (the unsigned minimum of the differences' 32-bit halves over the steps, a
// zero high half in some lane of it); progression is "every step is the step before it plus 8 strides, and the first is the first lane's
// address plus 0 to 7 strides", which is every lane on the stride of the first two, modulo 2^64.
template <bool Lo, bool Hi>
[[gnu::always_inline]] inline void warp_rows512(const uint32_t* lo, const uint32_t* hi, uint64_t add, uint64_t window, uint64_t* addrs, WarpRowsInfo& info) {
  const auto lane = [&](unsigned l) { return Lo ? ((Hi ? static_cast<uint64_t>(hi[l]) << 32 : 0) | lo[l]) + add : add; };
  const uint64_t a0 = lane(0), stride = lane(1) - a0;
  const __m512i vadd = _mm512_set1_epi64(static_cast<long long>(add));
  const __m512i vwin = _mm512_set1_epi64(static_cast<long long>(window));
  const __m512i step = _mm512_set1_epi64(static_cast<long long>(8 * stride));
  __m512i prev = _mm512_sub_epi64(_mm512_add_epi64(_mm512_set1_epi64(static_cast<long long>(a0)),
                                                   _mm512_mullo_epi64(_mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7), _mm512_set1_epi64(static_cast<long long>(stride)))),
                                  step);
  __m512i differ = _mm512_setzero_si512();
  __m512i least = _mm512_set1_epi32(-1);
#pragma GCC unroll 4
  for (unsigned i = 0; i < 32; i += 8) {
    __m512i a = vadd;
    if constexpr (Lo) {
      a = _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(lo + i)));
      if constexpr (Hi) a = _mm512_or_si512(a, _mm512_slli_epi64(_mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(hi + i))), 32));
      a = _mm512_add_epi64(a, vadd);
    }
    _mm512_storeu_si512(reinterpret_cast<void*>(addrs + i), a);
    least = _mm512_min_epu32(least, _mm512_sub_epi64(a, vwin));
    differ = _mm512_or_si512(differ, _mm512_xor_si512(_mm512_sub_epi64(a, prev), step));
    prev = a;
  }
  info.a0 = a0;
  info.stride = stride;
  info.in_window = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(least, 32), _mm512_setzero_si512()) != 0;
  info.progression = _mm512_test_epi64_mask(differ, differ) == 0;
}

void warp_rows_k(const uint32_t* lo, const uint32_t* hi, uint64_t add, uint64_t window, uint64_t* addrs, WarpRowsInfo& info) {
  if (!lo) warp_rows512<false, false>(lo, hi, add, window, addrs, info);
  else if (!hi) warp_rows512<true, false>(lo, hi, add, window, addrs, info);
  else warp_rows512<true, true>(lo, hi, add, window, addrs, info);
}

// sassmem::full_warp_run, sixteen lanes to a step and nothing stored: lane l's (lo[l] | hi[l] << 32) is the first lane's plus l * bytes modulo
// 2^64. The low word is the first lane's plus l * bytes; the high word is the first lane's plus the carry out of that sum, and the carry is the
// sum being below the first lane's low word (a mask register); with no high row no lane may carry.
template <bool Hi>
[[gnu::always_inline]] inline bool warp_run512(const uint32_t* lo, const uint32_t* hi, unsigned bytes) {
  const __m512i vlo0 = _mm512_set1_epi32(static_cast<int>(lo[0]));
  const __m512i vhi0 = Hi ? _mm512_set1_epi32(static_cast<int>(hi[0])) : _mm512_setzero_si512();
  const __m512i one = _mm512_set1_epi32(1);
  const __m512i step = _mm512_set1_epi32(static_cast<int>(16 * bytes));
  __m512i e = _mm512_add_epi32(vlo0, _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(static_cast<int>(bytes))));
  __m512i differ = _mm512_setzero_si512();
  unsigned carried = 0;
  for (unsigned i = 0; i < 32; i += 16) {
    differ = _mm512_or_si512(differ, _mm512_xor_si512(VGPU_L512_LOAD(lo + i), e));
    const __mmask16 wrapped = _mm512_cmp_epu32_mask(e, vlo0, _MM_CMPINT_LT);   // the lanes whose low sum wrapped
    if constexpr (Hi) differ = _mm512_or_si512(differ, _mm512_xor_si512(VGPU_L512_LOAD(hi + i), _mm512_mask_add_epi32(vhi0, wrapped, vhi0, one)));
    else carried |= wrapped;
    e = _mm512_add_epi32(e, step);
  }
  return _mm512_test_epi32_mask(differ, differ) == 0 && carried == 0;
}

bool warp_run_k(const uint32_t* lo, const uint32_t* hi, unsigned bytes) {
  return hi ? warp_run512<true>(lo, hi, bytes) : warp_run512<false>(lo, hi, bytes);
}

// ---- IMAD.WIDE and MUFU -----------------------------------------------------------------------------------------------------------------------

// Bit k of a byte to bit 2k of a word: the carries of the even lanes of a block of sixteen (one bit a 64-bit lane) back to their lane bits.
struct Spread8 {
  uint16_t v[256];
  constexpr Spread8() : v{} {
    for (unsigned n = 0; n < 256; ++n)
      for (unsigned k = 0; k < 8; ++k) v[n] = static_cast<uint16_t>(v[n] | (((n >> k) & 1u) << (2 * k)));
  }
};
inline constexpr Spread8 kSpread8{};

// The 64-bit register pair (rlo, rhi) = a * b + (clo, chi) + bit l of cin, the product of the 32-bit operands signed or unsigned, the sum modulo
// 2^64, and bit l of *carry_out the carry out of that sum: kernels.inc's imad_wide_avx2, sixteen lanes to a step. The even lanes of the block
// are the low halves of the eight 64-bit lanes, which is where the widening multiply reads its operands, and the odd lanes are brought there by
// a shift; the addend's two words are put side by side with a blend, and the two results are taken apart the same way. No lane is moved to
// another, so there is nothing to put back in order. A sum below what was added to it carried (an unsigned compare into a mask register).
template <bool Sgn, bool Carry, bool Cin>
[[gnu::always_inline]] inline void imad_wide512(uint32_t* rlo, uint32_t* rhi, const uint32_t* a, const uint32_t* b, const uint32_t* clo, const uint32_t* chi,
                                                uint32_t cin, uint32_t* carry_out) {
  const __mmask16 odd = 0xaaaa;
  const __m512i one = _mm512_set1_epi32(1);
  uint32_t out = 0;
  for (unsigned i = 0; i < 32; i += 16) {
    const __m512i x = VGPU_L512_LOAD(a + i), y = VGPU_L512_LOAD(b + i), cl = VGPU_L512_LOAD(clo + i), ch = VGPU_L512_LOAD(chi + i);
    const __m512i xo = _mm512_srli_epi64(x, 32), yo = _mm512_srli_epi64(y, 32);
    const __m512i pe = Sgn ? _mm512_mul_epi32(x, y) : _mm512_mul_epu32(x, y);
    const __m512i po = Sgn ? _mm512_mul_epi32(xo, yo) : _mm512_mul_epu32(xo, yo);
    const __m512i ce = _mm512_mask_blend_epi32(odd, cl, _mm512_slli_epi64(ch, 32));
    const __m512i co = _mm512_mask_blend_epi32(odd, _mm512_srli_epi64(cl, 32), ch);
    const __m512i s1e = _mm512_add_epi64(pe, ce), s1o = _mm512_add_epi64(po, co);
    __m512i s2e = s1e, s2o = s1o;
    if constexpr (Cin) {
      const __m512i cm = _mm512_maskz_mov_epi32(static_cast<__mmask16>(cin >> i), one);   // 1 in the lanes that carry in
      s2e = _mm512_add_epi64(s1e, _mm512_maskz_mov_epi32(0x5555, cm));
      s2o = _mm512_add_epi64(s1o, _mm512_srli_epi64(cm, 32));
    }
    VGPU_L512_STORE(rlo + i, _mm512_mask_blend_epi32(odd, s2e, _mm512_slli_epi64(s2o, 32)));
    VGPU_L512_STORE(rhi + i, _mm512_mask_blend_epi32(odd, _mm512_srli_epi64(s2e, 32), s2o));
    if constexpr (Carry) {
      const unsigned even_c = _mm512_cmp_epu64_mask(s1e, pe, _MM_CMPINT_LT) | _mm512_cmp_epu64_mask(s2e, s1e, _MM_CMPINT_LT);
      const unsigned odd_c = _mm512_cmp_epu64_mask(s1o, po, _MM_CMPINT_LT) | _mm512_cmp_epu64_mask(s2o, s1o, _MM_CMPINT_LT);
      out |= (uint32_t{kSpread8.v[even_c]} | (uint32_t{kSpread8.v[odd_c]} << 1)) << i;
    }
  }
  if constexpr (Carry) *carry_out = out;
}

void imad_wide_k(uint32_t* rlo, uint32_t* rhi, const uint32_t* a, const uint32_t* b, const uint32_t* clo, const uint32_t* chi, bool sgn, uint32_t cin,
                 uint32_t* carry_out) {
  // (`carry_out` null: the carry is not wanted and is not found.)
  if (sgn) {
    if (carry_out) {
      if (cin) imad_wide512<true, true, true>(rlo, rhi, a, b, clo, chi, cin, carry_out);
      else imad_wide512<true, true, false>(rlo, rhi, a, b, clo, chi, cin, carry_out);
    } else {
      if (cin) imad_wide512<true, false, true>(rlo, rhi, a, b, clo, chi, cin, carry_out);
      else imad_wide512<true, false, false>(rlo, rhi, a, b, clo, chi, cin, carry_out);
    }
  } else {
    if (carry_out) {
      if (cin) imad_wide512<false, true, true>(rlo, rhi, a, b, clo, chi, cin, carry_out);
      else imad_wide512<false, true, false>(rlo, rhi, a, b, clo, chi, cin, carry_out);
    } else {
      if (cin) imad_wide512<false, false, true>(rlo, rhi, a, b, clo, chi, cin, carry_out);
      else imad_wide512<false, false, false>(rlo, rhi, a, b, clo, chi, cin, carry_out);
    }
  }
}

// MUFU.RCP (kind 4), .RSQ (5), .SQRT (8): in double, rounded once to float in the host's rounding mode; a subnormal input is the zero of its
// sign first, a subnormal result after, and a NaN result is the canonical one (kernels.inc's mufu_k: the same divisions and roots, eight
// doubles to a vector where it has four).
void mufu_k(unsigned kind, uint32_t* r, const uint32_t* a) {
  const __m512d one = _mm512_set1_pd(1.0);
  const __m256i expo = _mm256_set1_epi32(0x7f800000), sign = _mm256_set1_epi32(static_cast<int>(0x80000000u));
  for (unsigned i = 0; i < 32; i += 8) {
    __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    x = _mm256_mask_and_epi32(x, _mm256_testn_epi32_mask(x, expo), x, sign);
    const __m512d d = _mm512_cvtps_pd(_mm256_castsi256_ps(x));
    const __m512d v = kind == 4 ? _mm512_div_pd(one, d) : kind == 5 ? _mm512_div_pd(one, _mm512_sqrt_pd(d)) : _mm512_sqrt_pd(d);
    const __m256 f = _mm512_cvtpd_ps(v);
    __m256i u = _mm256_castps_si256(f);
    u = _mm256_mask_and_epi32(u, _mm256_testn_epi32_mask(u, expo), u, sign);
    u = _mm256_mask_mov_epi32(u, _mm256_cmp_ps_mask(f, f, _CMP_UNORD_Q), _mm256_set1_epi32(0x7fffffff));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), u);
  }
}

#undef VGPU_L512_LOP3
#undef VGPU_L512_LOAD
#undef VGPU_L512_STORE
