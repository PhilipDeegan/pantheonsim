#!/usr/bin/env python3
"""ceiling.py prof.log -> the markdown tables of round sp48's stage 1 (one row per program and mode, micro-ops and cycles).
Columns, as % of the process's IBS samples:
  loop      the run loop and the scheduler around the handlers (dispatch, the per-instruction counters, the chain)
  decode    scalar instructions of the ALU handlers and lane kernels (operand kinds, guards, calls) and of generated code (counters, guards, exits)
  rows      vector moves with a memory operand in ALU handlers, kernels and generated code (register-file rows, local rows, splats)
  lanes     vector arithmetic, logic, compares and blends in ALU handlers, kernels and generated code: the guest instructions' own work
  mem       the memory handlers and the memory system (lookup, copies, access counting)
  other     launch and block setup, libc, the program's host code, the kernel
`removable` = loop + decode + rows: what a compiler of whole blocks could take out at most if every ALU op were in a block and no row ever went to memory."""
import re, sys
blocks = {}; cur = None
for l in open(sys.argv[1]):
    m = re.match(r"== (\S+) (\S+) ibs_op cnt_ctl=(\d)", l)
    if m: cur = blocks.setdefault((m.group(1), m.group(2), m.group(3)), {"cls": {}, "ins": {}}); continue
    if cur is None: continue
    m = re.match(r"\s+(\S+)\s+([\d.]+) %$", l)
    if m: cur["cls"][m.group(1)] = float(m.group(2)); continue
    m = re.match(r"\s+(\S+)\s+lanes\s+([\d.]+)\s+rows\s+([\d.]+)\s+scalar\s+([\d.]+)\s+unknown\s+([\d.]+)", l)
    if m: cur["ins"][m.group(1)] = [float(x) for x in m.groups()[1:]]; continue
    m = re.match(r"samples (\d+)", l)
    if m: cur["n"] = int(m.group(1))
for ctl, what in (("1", "micro-ops (`ibs_op/cnt_ctl=1/`)"), ("0", "cycles (`ibs_op/cnt_ctl=0/`)")):
    print("\nShares of %s, %% of the process's samples:\n" % what)
    print("| program | mode | samples | loop | decode | rows | lanes | of which generated code (decode / rows / lanes) | mem handlers | mem system | other | **removable (loop + decode + rows)** |")
    print("|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for (prog, mode, c), b in blocks.items():
        if c != ctl or not b.get("n"): continue
        cl, ins = b["cls"], b["ins"]
        z = [0, 0, 0, 0]
        a, k, g = ins.get("aluh", z), ins.get("kernel", z), ins.get("gen", z)
        lanes = a[0] + k[0] + g[0]; rows = a[1] + k[1] + g[1]; dec = a[2] + k[2] + g[2] + a[3] + k[3] + g[3]
        loop = cl.get("loop", 0)
        other = cl.get("sim-other", 0) + cl.get("libc", 0) + cl.get("program", 0) + cl.get("os", 0)
        print("| %s | %s | %d | %.1f | %.1f | %.1f | %.1f | %.1f / %.1f / %.1f | %.1f | %.1f | %.1f | **%.1f** |" % (
            prog, mode, b["n"], loop, dec, rows, lanes, g[2] + g[3], g[1], g[0], cl.get("memh", 0), cl.get("memsys", 0), other, loop + dec + rows))
