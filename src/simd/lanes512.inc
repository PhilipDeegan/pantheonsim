// The binary64 kernel of the SASS tier (simd::Kernels::dop_m: DADD, DMUL, DFMA on register pairs) at the AVX-512 level: sixteen lanes to a step
// (two vectors of eight doubles) where kernels.inc's dop_avx2_t takes four. simd.cpp includes this file inside namespace avx512_impl, under the
// same #pragma GCC target as kernels512.inc, and make_avx512 puts the entry in the Avx512 table; VGPU_SIMD=avx2 keeps the 256-bit one. It is the
// one lane kernel with a 512-bit form: round sp50 built sixteen-lane forms of thirteen others, and none of them paid; round sp54 measured this
// one alone (docs/perf-design/rounds/11-rounds-sp36-sp41.md, Round sp50 and Round sp54).
//
// The kernel gives the bits of the AVX2 one: tests/unit/test_lanes512.cpp runs it against the Avx2 table's entry, and against a definition
// written lane by lane, on a host that has both (and reports itself skipped on a host that has not). The same operations at another width: one
// IEEE operation a lane in the host's rounding mode (a fused multiply-add for kind 2, never for the others: contraction is off for the whole
// file, simd.cpp's pragma).
//
// Aliasing is as in kernels.inc: a block of lanes is read from every source before it is written, so the result rows may be source rows (the
// tier's tier_out); the blocks here are 16 lanes, which changes nothing for a row that is the same row or another one.

#define VGPU_L512_LOAD(p) _mm512_loadu_si512(reinterpret_cast<const void*>(p))
#define VGPU_L512_STORE(p, v) _mm512_storeu_si512(reinterpret_cast<void*>(p), (v))

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

#undef VGPU_L512_LOAD
#undef VGPU_L512_STORE
