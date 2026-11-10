#!/usr/bin/env python3
"""classify.py <prefix>.ctlN.script [--comm NAME] [--top N] [--insn]
Splits the IBS-op samples of one profiled run (prof.sh) of the simulator into what a block compiler could and could not remove.

Every sample has an exact instruction address. A sample is classified twice:
  * by the function it is in (the table FUNC below: run loop, ALU handler, memory handler, memory system, generated code, lane kernel, ...);
  * inside ALU handlers, lane kernels and generated code, by the host instruction at that address (objdump of the DSO):
      vector arithmetic / logic / compare / blend on ymm, zmm, xmm or mask registers  -> "lanes"   (the guest instruction's own work)
      a vector move with a memory operand (the register file, a local row, a splat)   -> "rows"    (register-file and row traffic)
      everything else (scalar moves, tests, jumps, calls, the counters)               -> "scalar"  (decode, guard, dispatch, counters)
Prints percentages of the process's samples. --insn prints the instruction-level split per function class; --top N the hottest symbols."""
import collections, re, subprocess, sys, os

FUNC = [  # first match wins, on the demangled name
    ("gen",      r"^vgpu_jit_r\d+"),
    ("loop",     r"step_warp|tier_run|tier_issue|jit_run_region|pick_group|block_round|run_block|run_grid|select_pcs|break_spin_wait|jit_sreg|jit_note|tier_sreg|tier_program|Runner::run\b|Runner::launch|worker"),
    ("memh",     r"h_global|h_shared|h_local|h_atom|h_ld|h_st|h_red|h_tex|exec_mem|exec_atom|fast_mem|tier_mem"),
    ("memsys",   r"MemoryManager|memcount|sassatom|count_run|count_|bank_|sector|thread_bank_memo|resolve|Allocation|warp_run_k|warp_rows|run_allocation|load_run|store_run|RaceDetector|\brun_in\b|\brun_out\b|sassmem|global_run"),
    ("aluh",     r"\bh_[a-z0-9_]+|tier_dyn|fast_alu|exec_alu|Runner::exec\b|exec_one"),
    ("kernel",   r"simd::"),
]
VEC = re.compile(r"%[xyz]mm|%k[0-7]")
MOVE = re.compile(r"^v?(mov(dq[au]|[au]p[sd]|[dq]|ntdq|ss|sd|dqa64|dqu64|dqu32|dqa32|dqu8|dqu16)|pbroadcast[bwdq]|broadcast(ss|sd|f128|i128|f32x4|i32x4)?|pmovzx..|pmovsx..|extract[fi]128|insert[fi]128|extract[fi]32x4|extract[fi]64x4|insert[fi]64x4)$")

def demangle(names):
    p = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    return dict(zip(names, p.stdout.split("\n")))

_dis = {}
def disasm(dso):
    if dso in _dis: return _dis[dso]
    m = {}
    if os.path.exists(dso):
        p = subprocess.run(["objdump", "-d", "--no-show-raw-insn", dso], capture_output=True, text=True)
        for l in p.stdout.split("\n"):
            mm = re.match(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$", l)
            if mm: m[int(mm.group(1), 16)] = (mm.group(2), mm.group(3))
    _dis[dso] = m
    return m

def insn_class(mn, ops):
    if not VEC.search(ops): return "scalar"
    if MOVE.match(mn) and "(" in ops: return "rows"
    return "lanes"

def main():
    a = sys.argv[1:]
    path = a[0]
    comm = a[a.index("--comm") + 1] if "--comm" in a else None
    top = int(a[a.index("--top") + 1]) if "--top" in a else 0
    rx = re.compile(r"^\s*(\S+)\s+([0-9a-f]+)\s+(\S+?)\+0x([0-9a-f]+)\s+\((.*?)\+0x([0-9a-f]+)\)\s*$")
    rx2 = re.compile(r"^\s*(\S+)\s+([0-9a-f]+)\s+(\S+)\s+\((.*?)(\+0x([0-9a-f]+))?\)\s*$")
    S = collections.Counter(); total = 0; other = 0
    for l in open(path, errors="replace"):
        m = rx.match(l)
        if m: c, ip, sym, so, dso, do = m.groups(); do = int(do, 16)
        else:
            m = rx2.match(l)
            if not m: continue
            c, ip, sym, dso, _, do = m.groups(); do = int(do, 16) if do else -1
        if comm and not c.startswith(comm[:15]): other += 1; continue
        total += 1
        S[(dso, sym, do)] += 1
    names = sorted(set(k[1] for k in S))
    dm = demangle(names)
    byfn = collections.Counter(); bycls = collections.Counter(); ins = collections.defaultdict(collections.Counter)
    mn_hist = collections.defaultdict(collections.Counter)
    for (dso, sym, do), n in S.items():
        name = dm.get(sym, sym)
        base = os.path.basename(dso)
        if "[kernel" in dso or dso.startswith("[k") or sym.startswith("[k"): cls = "os"
        elif re.search(r"vgpu-jit|/jit/|\.cache/.*\.so$", dso) and "libcud" not in base and "libvgpu" not in base: cls = "gen"
        elif "libc.so" in base or "libm.so" in base or "ld-linux" in base: cls = "libc"
        elif "libcud" not in base and "libvgpu" not in base: cls = "program"
        else:
            cls = "sim-other"
            for cname, r in FUNC[1:]:
                if re.search(r, name): cls = cname; break
        byfn[(cls, name if cls != "gen" else base + ":" + name)] += n
        bycls[cls] += n
        if cls in ("aluh", "kernel", "gen", "loop", "memh", "memsys"):
            d = disasm(dso).get(do)
            ic = insn_class(*d) if d else "unknown"
            ins[cls][ic] += n
            if d: mn_hist[(cls, ic)][d[0]] += n
    print("samples %d (other processes on the cpu: %d)" % (total, other))
    order = ["loop", "aluh", "kernel", "gen", "memh", "memsys", "sim-other", "libc", "program", "os"]
    for c in order:
        print("  %-10s %6.2f %%" % (c, 100.0 * bycls[c] / max(total, 1)))
    if "--insn" in a:
        print("instruction-level split (% of all samples):")
        for c in ("aluh", "kernel", "gen", "loop", "memh", "memsys"):
            t = ins[c]
            print("  %-8s lanes %5.2f  rows %5.2f  scalar %5.2f  unknown %5.2f" % (c, *(100.0 * t[k] / max(total, 1) for k in ("lanes", "rows", "scalar", "unknown"))))
        if "--mn" in a:
            for key, h in sorted(mn_hist.items()):
                print("   ", key, ", ".join("%s %.2f" % (m, 100.0 * n / total) for m, n in h.most_common(12)))
    if top:
        print("top symbols:")
        for (c, name), n in byfn.most_common(top):
            print("  %6.2f %%  %-9s %s" % (100.0 * n / total, c, name[:150]))

main()
