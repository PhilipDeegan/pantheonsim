#!/usr/bin/env python3
"""mutants.py [budget seconds]: each mutant is one textual change of src/simd/lanes512.inc or src/simd/simd.cpp in ~/.cache/sp54/src; simd.cpp is
compiled with the build's own command for that object (ninja -t commands: -O3, the hot-unit flags) into ~/.cache/sp54/mut/simd.o, test_lanes512
is linked from the build's test object, that simd.o and the build's libvgpu.a (the object comes first on the line, so the archive's own simd.cpp.o
is not taken), and run; a mutant is killed when the test fails. The source is put back after each one, and at the end the unmutated source goes
through the same route and must pass. Not through `cmake --build`: on the measuring host re-creating libvgpu.a (ar and ranlib of the archive with
LTO bytecode) took five minutes a mutant (the first two were run that way: mutants-archive-route.txt). Results are appended to
~/.cache/sp54/log/mutants.txt; mutants already there are skipped. Run it under the shared builder lock:
flock /tmp/pantheonsim-builder.lock python3 mutants.py 600"""
import os, subprocess, sys, time
R = os.path.expanduser("~/.cache/sp54")
L, S = "src/simd/lanes512.inc", "src/simd/simd.cpp"
M = [
 ("the NaN rule applied to the operands after |x| and -x", L, "      if constexpr (Mods) {\n        load(alo, ahi, 0, i, false, xa, xb);", "      if constexpr (false) {\n        load(alo, ahi, 0, i, false, xa, xb);"),
 ("the fma's NaN rule leaves out the addend", L, "if (is_nan(X[k]) || is_nan(Y[k]) || (Kind == 2 && is_nan(uc))) R[k] = dop_nan(Kind, X[k], Y[k], uc);", "if (is_nan(X[k]) || is_nan(Y[k])) R[k] = dop_nan(Kind, X[k], Y[k], uc);"),
 ("every lane written, whatever the mask", L, "    } else if (bits) {\n      _mm512_mask_storeu_epi32(dlo + i, bits, rlo);\n      _mm512_mask_storeu_epi32(dhi + i, bits, rhi);", "    } else {\n      VGPU_L512_STORE(dlo + i, rlo);\n      VGPU_L512_STORE(dhi + i, rhi);"),
 ("the product and the sum not fused (lanes 0 to 7 of a block)", L, "va = _mm512_fmadd_pd(xa, ya, za);", "va = _mm512_add_pd(_mm512_mul_pd(xa, ya), za);"),
 ("-x applied before |x|", L, "h = _mm512_xor_si512(_mm512_and_si512(h, _mm512_set1_epi32(static_cast<int>(absm[k]))), _mm512_set1_epi32(static_cast<int>(negm[k])));", "h = _mm512_and_si512(_mm512_xor_si512(h, _mm512_set1_epi32(static_cast<int>(negm[k]))), _mm512_set1_epi32(static_cast<int>(absm[k])));"),
 ("the sum rounded to nearest, whatever the rounding mode (lanes 8 to 15 of a block)", L, "vb = _mm512_add_pd(xb, yb);", "vb = _mm512_add_round_pd(xb, yb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);"),
 ("the product rounded toward zero, whatever the rounding mode", L, "va = _mm512_mul_pd(xa, ya);", "va = _mm512_mul_round_pd(xa, ya, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);"),
 ("lanes 16 to 31 not computed", L, "  for (unsigned i = 0; i < 32; i += 16) {\n    const __mmask16 bits", "  for (unsigned i = 0; i < 16; i += 16) {\n    const __mmask16 bits"),
 ("lanes 16 to 31 take the mask bits of lanes 0 to 15", L, "const __mmask16 bits = static_cast<__mmask16>(mask >> i);", "const __mmask16 bits = static_cast<__mmask16>(mask);"),
 ("the high words of the results taken from the low words' positions in lanes 8 to 15", L, "const __m512i odds = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);", "const __m512i odds = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28, 30);"),
 ("lanes 8 to 15 of a block read lanes 0 to 7's operands", L, "q = _mm512_castsi512_pd(_mm512_permutex2var_epi32(l, second, h));", "q = _mm512_castsi512_pd(_mm512_permutex2var_epi32(l, first, h));"),
 ("the NaN rule's results not taken for lanes 8 to 15", L, "      vb = _mm512_load_pd(R + 8);\n", ""),
 ("a NaN result in lanes 8 to 15 alone does not reach the NaN rule", L, "if (_mm512_cmp_pd_mask(va, va, _CMP_UNORD_Q) | _mm512_cmp_pd_mask(vb, vb, _CMP_UNORD_Q)) [[unlikely]] {", "if (_mm512_cmp_pd_mask(va, va, _CMP_UNORD_Q)) [[unlikely]] {"),
 ("the modifiers of source b applied to source c", L, "load(clo, chi, 2, i, Mods, za, zb);", "load(clo, chi, 1, i, Mods, za, zb);"),
 ("DMUL without modifiers runs DADD", L, "else if (kind == 1) dop512_t<1, false>(", "else if (kind == 1) dop512_t<0, false>("),
 ("|x| of source c never applied", L, "(mods & 16) ? 0x7fffffffu : ~0u};", "~0u};"),
 ("dispatch: the Avx512 table keeps the AVX2 dop_m", S, "  k.dop_m = avx512_impl::dop_m_k;\n", ""),
]
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 600
outp = R + "/log/mutants.txt"
done = {l.split()[0] for l in open(outp)} if os.path.exists(outp) else set()
out = open(outp, "a"); t_start = time.time()
def sh(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True)
B = R + "/build"; os.makedirs(R + "/mut", exist_ok=True)
cc = [l for l in sh("ninja -C %s -t commands test_lanes512" % B).stdout.splitlines() if "-o CMakeFiles/vgpu.dir/src/simd/simd.cpp.o " in l][0]
cc = cc.replace("-o CMakeFiles/vgpu.dir/src/simd/simd.cpp.o", "-o %s/mut/simd.o" % R).replace("-MD -MT CMakeFiles/vgpu.dir/src/simd/simd.cpp.o -MF CMakeFiles/vgpu.dir/src/simd/simd.cpp.o.d", "")
BUILD = ("cd %s && nice -n 19 %s 2>&1 | grep -E 'error' ; nice -n 19 /usr/bin/c++ -g1 -O3 -DNDEBUG -fno-lto CMakeFiles/test_lanes512.dir/tests/unit/test_lanes512.cpp.o "
         "%s/mut/simd.o libvgpu.a -ldl -o %s/mut/test_lanes512 2>&1 | tail -3") % (B, cc, R, R)
def build_and_run():
    if os.path.exists(R + "/mut/test_lanes512"): os.remove(R + "/mut/test_lanes512")
    r = sh(BUILD)
    if not os.path.exists(R + "/mut/test_lanes512"): return None, r.stdout[-300:].replace("\n", " ")
    t = sh("nice -n 19 %s/mut/test_lanes512 2>&1" % R)
    return t, ""
for n, (name, f, a, b) in enumerate(M, 1):
    tag = "M%02d" % n
    if tag in done: continue
    if time.time() - t_start > budget: print("budget used; rerun for the rest"); break
    p = os.path.join(R, "src", f); src = open(p).read()   # (the source tree is ~/.cache/sp54/src)
    if src.count(a) != 1:
        print("%s NOT APPLIED (%d matches): %s" % (tag, src.count(a), name), file=out, flush=True); continue
    open(p, "w").write(src.replace(a, b)); t0 = time.time()
    try:
        t, err = build_and_run()
        if t is None:
            print("%s DOES NOT COMPILE: %s :: %s" % (tag, name, err), file=out, flush=True); continue
        fails = [l[9:] for l in t.stdout.splitlines() if l.startswith("[ FAIL ]")]
        print("%s %s (%.0f s): %s :: %s" % (tag, "killed" if t.returncode != 0 and fails else "SURVIVED", time.time() - t0, name, ", ".join(fails) or t.stdout[-120:].replace("\n", " ")), file=out, flush=True)
    finally:
        open(p, "w").write(src)
else:
    t, err = build_and_run()
    print("UNMUTATED AGAIN: " + (err if t is None else "rc %d :: %s" % (t.returncode, t.stdout.strip().splitlines()[-1])), file=out, flush=True)
