#!/usr/bin/env python3
"""mut.py: each mutant is one textual change of src/simd/lanes512.inc, src/simd/lanes.cpp or src/simd/simd.cpp in /tmp/sp50-avx512/mut;
test_lanes512 is rebuilt (nice, -j3) and run; a mutant is killed when the test fails. Output: mut.txt. Run the whole script under the shared
builder lock (flock /tmp/pantheonsim-builder.lock python3 mut.py [budget seconds] [mutant numbers...]): one acquisition for the batch."""
import subprocess, sys, os, time
R = "/tmp/sp50-avx512/mut"
L, C, S = "src/simd/lanes512.inc", "src/simd/lanes.cpp", "src/simd/simd.cpp"
M = [
 ("iadd3: the addend left out of lanes 16 to 31", L, "const __m512i v1 = _mm512_add_epi32(_mm512_add_epi32(x1, y1), _mm512_add_epi32(z1, k));", "const __m512i v1 = _mm512_add_epi32(_mm512_add_epi32(x1, y1), z1);"),
 ("imad: a * a + c", L, "_mm512_add_epi32(_mm512_mullo_epi32(x, y), z)", "_mm512_add_epi32(_mm512_mullo_epi32(x, x), z)"),
 ("lop3: table 0xe8 run as 0xe0", L, "VGPU_L512_T(0xe8)", "case 0xe8: VGPU_L512_LOP3(_mm512_ternarylogic_epi32(x, y, z, 0xe0));"),
 ("lop3: a minterm's c input not complemented in the general form", L, "_mm512_xor_si512(z, mc[m])", "z"),
 ("ffma: multiply, round, then add (not fused)", L, "canon_nan512(_mm512_fmadd_ps(x, y, z))", "canon_nan512(_mm512_add_ps(_mm512_mul_ps(x, y), z))"),
 ("float kernels: the NaN is 0x7fc00000, not 0x7fffffff", L, "_mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q), _mm512_set1_epi32(0x7fffffff)", "_mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q), _mm512_set1_epi32(0x7fc00000)"),
 ("fadd: always to nearest, whatever the rounding mode", L, "canon_nan512(_mm512_add_ps(x, y))", "canon_nan512(_mm512_add_round_ps(x, y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))"),
 ("fmul: lanes 16 to 31 not written", L, "void fmul_k(uint32_t* r, const uint32_t* a, const uint32_t* b) {\n  for (unsigned i = 0; i < 32; i += 16) {", "void fmul_k(uint32_t* r, const uint32_t* a, const uint32_t* b) {\n  for (unsigned i = 0; i < 16; i += 16) {"),
 ("icmp_mask: signed less-than compared unsigned", L, "case 5: return icmp512_t<_MM_CMPINT_LT, true>(a, b, n);", "case 5: return icmp512_t<_MM_CMPINT_LT, false>(a, b, n);"),
 ("icmp_mask: lanes 32 to 63 never compared", L, "  for (unsigned i = 0; i < n; i += 16) {\n    const __m512i x = VGPU_L512_LOAD(a + i), y = VGPU_L512_LOAD(b + i);\n    const __mmask16 m", "  for (unsigned i = 0; i < 32; i += 16) {\n    const __m512i x = VGPU_L512_LOAD(a + i), y = VGPU_L512_LOAD(b + i);\n    const __mmask16 m"),
 ("blend_m: the upper sixteen lanes take the lower lanes' mask", L, "m1 = static_cast<__mmask16>(mask >> 16)", "m1 = static_cast<__mmask16>(mask)"),
 ("iadd3x: two carries counted only when the first addition is one of them", L, "o2 |= ((c1 & c2) | (c1 & c3) | (c2 & c3)) << i;", "o2 |= ((c1 & c2) | (c1 & c3)) << i;"),
 ("iadd3x: the second carry in ignored", L, "k = _mm512_mask_add_epi32(k, static_cast<__mmask16>(cin_b >> i), k, one);", ""),
 ("imad_wide: the odd lanes take the even lanes' carry in", L, "s2o = _mm512_add_epi64(s1o, _mm512_srli_epi64(cm, 32));", "s2o = _mm512_add_epi64(s1o, _mm512_maskz_mov_epi32(0x5555, cm));"),
 ("imad_wide: the signed product of the odd lanes made unsigned", L, "const __m512i po = Sgn ? _mm512_mul_epi32(xo, yo) : _mm512_mul_epu32(xo, yo);", "const __m512i po = _mm512_mul_epu32(xo, yo);"),
 ("imad_wide: the carry of the carry-in addition not looked at (even lanes)", L, "const unsigned even_c = _mm512_cmp_epu64_mask(s1e, pe, _MM_CMPINT_LT) | _mm512_cmp_epu64_mask(s2e, s1e, _MM_CMPINT_LT);", "const unsigned even_c = _mm512_cmp_epu64_mask(s1e, pe, _MM_CMPINT_LT);"),
 ("mufu: a subnormal result not flushed", L, "    u = _mm256_mask_and_epi32(u, _mm256_testn_epi32_mask(u, expo), u, sign);\n", ""),
 ("mufu: the double rounded to float to nearest, whatever the rounding mode", L, "const __m256 f = _mm512_cvtpd_ps(v);", "const __m256 f = _mm512_cvt_roundpd_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);"),
 ("dop_m: the NaN rule applied to the operands after |x| and -x", L, "      if constexpr (Mods) {\n        load(alo, ahi, 0, i, false, xa, xb);", "      if constexpr (false) {\n        load(alo, ahi, 0, i, false, xa, xb);"),
 ("dop_m: the fma's NaN rule leaves out the addend", L, "if (is_nan(X[k]) || is_nan(Y[k]) || (Kind == 2 && is_nan(uc))) R[k] = dop_nan(Kind, X[k], Y[k], uc);", "if (is_nan(X[k]) || is_nan(Y[k])) R[k] = dop_nan(Kind, X[k], Y[k], uc);"),
 ("dop_m: every lane written, whatever the mask", L, "    } else if (bits) {\n      _mm512_mask_storeu_epi32(dlo + i, bits, rlo);\n      _mm512_mask_storeu_epi32(dhi + i, bits, rhi);", "    } else {\n      VGPU_L512_STORE(dlo + i, rlo);\n      VGPU_L512_STORE(dhi + i, rhi);"),
 ("dop_m: the product and the sum not fused", L, "va = _mm512_fmadd_pd(xa, ya, za);", "va = _mm512_add_pd(_mm512_mul_pd(xa, ya), za);"),
 ("dop_m: -x applied before |x|", L, "h = _mm512_xor_si512(_mm512_and_si512(h, _mm512_set1_epi32(static_cast<int>(absm[k]))), _mm512_set1_epi32(static_cast<int>(negm[k])));", "h = _mm512_and_si512(_mm512_xor_si512(h, _mm512_set1_epi32(static_cast<int>(negm[k]))), _mm512_set1_epi32(static_cast<int>(absm[k])));"),
 ("warp_rows: the stride check steps by four strides", L, "const __m512i step = _mm512_set1_epi64(static_cast<long long>(8 * stride));", "const __m512i step = _mm512_set1_epi64(static_cast<long long>(4 * stride));"),
 ("warp_rows: the window test on the whole difference, not its high half", L, "_mm512_cmpeq_epi64_mask(_mm512_srli_epi64(least, 32), _mm512_setzero_si512()) != 0;", "_mm512_cmpeq_epi64_mask(least, _mm512_setzero_si512()) != 0;"),
 ("warp_rows: the high words not shifted into place", L, "a = _mm512_or_si512(a, _mm512_slli_epi64(_mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(hi + i))), 32));", "a = _mm512_or_si512(a, _mm512_slli_epi64(_mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(hi + i))), 31));"),
 ("warp_run: a carry with no high row is let through", L, "return _mm512_test_epi32_mask(differ, differ) == 0 && carried == 0;", "return _mm512_test_epi32_mask(differ, differ) == 0;"),
 ("warp_run: the high word's carry not added", L, "_mm512_mask_add_epi32(vhi0, wrapped, vhi0, one)", "vhi0"),
 ("warp_run: lanes 16 to 31 stepped by 8 elements", L, "const __m512i step = _mm512_set1_epi32(static_cast<int>(16 * bytes));", "const __m512i step = _mm512_set1_epi32(static_cast<int>(8 * bytes));"),
 ("pick_group: lanes that do not run take part in the minimum", C, "best = _mm512_mask_min_epu64(best, static_cast<__mmask8>(run >> (8 * i)), best, v[i]);", "best = _mm512_min_epu64(best, v[i]);"),
 ("pick_group: a signed minimum", C, "best = _mm512_mask_min_epu64(best, static_cast<__mmask8>(run >> (8 * i)), best, v[i]);", "best = _mm512_mask_min_epi64(best, static_cast<__mmask8>(run >> (8 * i)), best, v[i]);"),
 ("pick_group: the group not limited to the lanes that run", C, "  lo = low;\n  return group & run;", "  lo = low;\n  return group;"),
 ("set_pcs: each step takes the mask bits of a 4-lane step", C, "_mm512_storeu_si512(at, _mm512_mask_mov_epi64(_mm512_loadu_si512(at), static_cast<__mmask8>(group >> (8 * i)), vv));", "_mm512_storeu_si512(at, _mm512_mask_mov_epi64(_mm512_loadu_si512(at), static_cast<__mmask8>(group >> (4 * i)), vv));"),
 ("select_pcs: target and next exchanged", C, "const __m512i to = _mm512_mask_mov_epi64(n, static_cast<__mmask8>(take >> (8 * i)), t);", "const __m512i to = _mm512_mask_mov_epi64(t, static_cast<__mmask8>(take >> (8 * i)), n);"),
 ("select_pcs: lanes outside the group written", C, "_mm512_storeu_si512(at, _mm512_mask_mov_epi64(_mm512_loadu_si512(at), static_cast<__mmask8>(group >> (8 * i)), to));", "_mm512_storeu_si512(at, to);"),
 ("dispatch: the Avx512 table keeps the AVX2 ffma", S, "  k.ffma = avx512_impl::ffma_k;\n", ""),
 ("dispatch: warp_rows_at gives the AVX2 kernel for Avx512", S, "if (l == Level::Avx512 && cpu_has_avx512()) return avx512_impl::warp_rows_k;", "if (l == Level::Avx512 && cpu_has_avx512()) return avx2_impl::warp_rows_k;"),
]
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
only = sys.argv[2:]
t_start = time.time()
# one mutant of each kernel first, so that a batch cut short by its budget has covered every kernel
ORDER = [1, 2, 3, 5, 6, 9, 11, 12, 14, 17, 19, 24, 27, 30, 33, 34, 36, 7, 18, 21, 22, 16, 25, 28, 31, 35, 4, 8, 10, 13, 15, 20, 23, 26, 29, 32, 37]
assert sorted(ORDER) == list(range(1, len(M) + 1))
out = open("/tmp/sp50-avx512/mut.txt", "a")
def sh(cmd, **kw): return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
for n in ORDER:
    name, f, a, b = M[n - 1]
    if only and str(n) not in only: continue
    if time.time() - t_start > budget:
        print("M%02d NOT RUN (batch budget used): %s" % (n, name), file=out, flush=True); continue
    p = os.path.join(R, f); src = open(p).read()
    if src.count(a) != 1:
        print("M%02d NOT APPLIED (%d matches): %s" % (n, src.count(a), name), file=out, flush=True); continue
    open(p, "w").write(src.replace(a, b))
    t0 = time.time()
    try:
        r = sh("nice -n 19 cmake --build %s/build -j2 --target test_lanes512 2>&1 | tail -4" % R)
        if "error" in r.stdout or "FAILED" in r.stdout:
            print("M%02d DOES NOT COMPILE: %s :: %s" % (n, name, r.stdout[-300:].replace("\n", " ")), file=out, flush=True); continue
        t = sh("nice -n 19 %s/build/test_lanes512 2>&1" % R)
        fails = [l[9:] for l in t.stdout.splitlines() if l.startswith("[ FAIL ]")]
        print("M%02d %s (%.0f s): %s :: %s" % (n, "killed" if t.returncode != 0 and fails else "SURVIVED", time.time() - t0, name, ", ".join(fails) or t.stdout[-120:].replace("\n", " ")), file=out, flush=True)
    finally:
        open(p, "w").write(src)
print("MUTANTS DONE", file=out, flush=True)
