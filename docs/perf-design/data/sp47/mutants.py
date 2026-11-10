#!/usr/bin/env python3
"""Round sp47 mutation checks: each mutant is one wrong line in the exact-threads code; test_exact_threads must fail on it.

  mutants.py <src> <build> [name ...]
For each mutant: patch the source in place, rebuild test_exact_threads, run it (and its VGPU_TIER=0 / VGPU_FASTPATH=0 forms),
put the source back. Prints "killed" (some form failed) or "SURVIVED" per mutant, then rebuilds the unmutated test and requires it to pass."""
import os, subprocess, sys

M = "src/core/memory_spec.cpp"
C = "src/core/memory.cpp"
X = "src/sass/exec.cpp"
MUTANTS = [
    ("write_conflict_ignored", M,
     '      spec_refuse(v != Spec::kMany && (v & Spec::kWrite) ? "two blocks wrote the same bytes" : "a block wrote what another block read");',
     '      break;'),
    ("read_of_written_ignored", M,
     '        spec_refuse("a block read what another block wrote");',
     '        break;'),
    ("readers_not_merged", M,
     '      } else if (word.compare_exchange_weak(v, Spec::kMany, std::memory_order_acq_rel)) {',
     '      } else if (word.compare_exchange_weak(v, tag, std::memory_order_acq_rel)) {'),
    ("no_write_back", M,
     '        write(addr, w.kept.data() + at, len);',
     '        (void)addr;'),
    ("nothing_kept", M,
     '    read_chunks(it->second, addr - it->first, w.kept.data() + at, n);',
     '    std::memset(w.kept.data() + at, 0, n);'),
    ("kept_first_word_of_a_run_only", M,
     '    spec_keep(s.base + run * Spec::kGranule, (upto - run) * Spec::kGranule);',
     '    spec_keep(s.base + run * Spec::kGranule, Spec::kGranule);'),
    ("shadow_not_cleared", M,
     '      Spec::Word* page = s.top[slot].exchange(nullptr, std::memory_order_acq_rel);',
     '      Spec::Word* page = s.top[slot].load(std::memory_order_acquire);\n      continue;'),
    ("host_store_allowed", M,
     '  if (off >= s.span || len > s.span - off) spec_refuse("a block wrote outside the device\'s own memory");',
     '  if (off >= s.span || len > s.span - off) return;'),
    ("hook_store_batch", C,
     '  if (spec_) [[unlikely]] spec_write_each(addrs, count, size);\n  const Allocation* ca = nullptr;\n  uint64_t base = 0;\n  if (!batch_ok(addrs, count, size, /*store=*/true, &ca, &base, hint)) return false;\n  store_batch_in(',
     '  const Allocation* ca = nullptr;\n  uint64_t base = 0;\n  if (!batch_ok(addrs, count, size, /*store=*/true, &ca, &base, hint)) return false;\n  store_batch_in('),
    ("hook_load_run", C,
     '  if (spec_) [[unlikely]] spec_read(addr, len);\n  if (len == 0) return true;\n  uint64_t base = 0;\n  const Allocation* a = run_allocation(',
     '  if (len == 0) return true;\n  uint64_t base = 0;\n  const Allocation* a = run_allocation('),
    ("hook_store_scalar", C,
     '  if (spec_) [[unlikely]] spec_write(addr, size);\n  if (size != 1 && size != 2 && size != 4 && size != 8)\n    throw Error::make(Err::Internal, "store_scalar: bad size ", size);',
     '  if (size != 1 && size != 2 && size != 4 && size != 8)\n    throw Error::make(Err::Internal, "store_scalar: bad size ", size);'),
    ("hook_load_scalar", C,
     '  if (spec_) [[unlikely]] spec_read(addr, size);\n  if (size != 1 && size != 2 && size != 4 && size != 8)\n    throw Error::make(Err::Internal, "load_scalar: bad size ", size);',
     '  if (size != 1 && size != 2 && size != 4 && size != 8)\n    throw Error::make(Err::Internal, "load_scalar: bad size ", size);'),
    ("window_not_declined", C,
     '  if (spec_) [[unlikely]] return false;   // (a trial takes the plain path, which is noted)\n  if (count == 0 || (size != 1 && size != 2 && size != 4 && size != 8)) return false;\n  if (!batch_faults_clear(false)',
     '  if (count == 0 || (size != 1 && size != 2 && size != 4 && size != 8)) return false;\n  if (!batch_faults_clear(false)'),
    ("stopped_trial_kept", X,
     '  if (!out.committed) {\n    next.store(0);   // every block again, in order\n    return false;\n  }',
     '  if (false) {\n    next.store(0);   // every block again, in order\n    return false;\n  }'),
    ("blocks_not_run_again_from_zero", X,
     '    next.store(0);   // every block again, in order',
     '    // next.store(0);'),
    ("counters_not_summed", X,
     '  for (const exec::LaunchStats& s : sums) stats_.add(s);   // (integer sums: the order of adding changes nothing)',
     '  if (!sums.empty()) stats_.add(sums[0]);'),
    ("smid_not_stopped", X,
     '      if (exact_trial_) {   // (the number below is the host thread\'s: in order it is thread 0\'s for every block)',
     '      if (false) {'),
    ("failed_block_not_a_stop", X,
     '        mem_.spec_stop("a block failed");',
     '        (void)0;'),
    ("no_hold_after_a_stop", X,
     '      hold.first = hold.second;',
     '      hold.first = 0;'),
    ("tried_without_blocks_in_order", X,
     'x && *x && env_.threads && *env_.threads && std::atoi(env_.threads) <= 1) {',
     'x && *x) {'),
]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def build(bld):
    r = sh("nice -n 19 cmake --build %s -j3 --target test_exact_threads" % bld)
    return r.returncode == 0, r.stdout[-2000:]


def run(bld):
    """-> list of the forms that failed"""
    failed = []
    for form, env in (("default", {}), ("tier0", {"VGPU_TIER": "0"}), ("fastpath0", {"VGPU_FASTPATH": "0"})):
        e = dict(os.environ)
        e.update(env)
        try:
            r = sh("nice -n 19 %s/test_exact_threads" % bld, env=e, timeout=600)
            if r.returncode != 0:
                failed.append(form)
        except subprocess.TimeoutExpired:
            failed.append(form + "(timeout)")
    return failed


def main():
    src, bld = sys.argv[1], sys.argv[2]
    want = sys.argv[3:]
    for name, path, old, new in MUTANTS:
        if want and name not in want:
            continue
        p = os.path.join(src, path)
        text = open(p).read()
        if text.count(old) != 1:
            print("%-32s NOT APPLIED (%d matches)" % (name, text.count(old)), flush=True)
            continue
        open(p, "w").write(text.replace(old, new))
        try:
            ok, log = build(bld)
            if not ok:
                print("%-32s DOES NOT BUILD\n%s" % (name, log), flush=True)
                continue
            failed = run(bld)
            print("%-32s %s" % (name, "killed by " + ", ".join(failed) if failed else "SURVIVED"), flush=True)
        finally:
            open(p, "w").write(text)
    ok, log = build(bld)
    failed = run(bld) if ok else ["build"]
    print("%-32s %s" % ("(no mutant)", "passes" if not failed else "FAILS: " + ", ".join(failed)), flush=True)


if __name__ == "__main__":
    main()
