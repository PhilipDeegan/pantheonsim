#!/usr/bin/env python3
"""Round sp55 mutation checks: each mutant is one wrong line in the exact-threads code; test_exact_threads must fail on it.

  mutants.py <src> <build> [name ...]
For each mutant: patch the source in place, rebuild test_exact_threads, run it in its four forms (default, VGPU_TIER=0,
VGPU_FASTPATH=0, VGPU_EXACT_PAGES=0), put the source back. Prints "killed by <forms>" or "SURVIVED" per mutant, then rebuilds
the unmutated test and requires it to pass. The first group is this round's mechanisms (the note by page, the worker's memory of
pages, a warp's lanes as one range or one page, the second trial); the second is round sp47's mutants on the lines this round moved
or rewrote, and a sample of those it did not touch."""
import os, subprocess, sys

M = "src/core/memory_spec.cpp"
C = "src/core/memory.cpp"
X = "src/sass/exec.cpp"
SEEN = "    if (seen.slot == slot && (seen.tag == w.tag || seen.tag == Spec::kMany)) [[likely]] continue;"
MUTANTS = [
    # ---- this round ----
    ("seen_for_any_block", M, SEEN, "    if (seen.slot == slot) [[likely]] continue;"),
    ("seen_for_any_page", M, SEEN, "    if (seen.tag == w.tag || seen.tag == Spec::kMany) [[likely]] continue;"),
    ("write_in_a_page_others_read", M,
     '      refuse_coarse(addr, "a block wrote in a page other blocks read");',
     '      break;'),
    ("own_reads_forgotten_on_write", M,
     '        for (uint64_t i = 0; i < kPageWords; ++i) mine[i].store(tag, std::memory_order_relaxed);',
     '        for (uint64_t i = 0; i < kPageWords; ++i) mine[i].store(0, std::memory_order_relaxed);'),
    ("second_reader_not_noted", M,
     '        if (ps.compare_exchange_weak(v, kMany, std::memory_order_acq_rel)) {\n          ++w.st.pages_shared;\n          v = kMany;\n        }',
     '        {\n          ++w.st.pages_shared;\n          v = kMany;\n        }'),
    ("writer_takes_a_page_of_several_readers", M,
     '      if (v == tag) {\n        // The page\'s only reader so far.',
     '      if (v == tag || v == kMany) {\n        // The page\'s only reader so far.'),
    ("warp_page_check_without_last_bytes", M,
     '    if (((apart | ends | ((a0 + (size - 1)) ^ a0)) >> 12) == 0',
     '    if (((apart | ((a0 + (size - 1)) ^ a0)) >> 12) == 0'),
    ("warp_page_check_without_first_bytes", M,
     '    if (((apart | ends | ((a0 + (size - 1)) ^ a0)) >> 12) == 0',
     '    if (((ends | ((a0 + (size - 1)) ^ a0)) >> 12) == 0'),
    ("warp_page_check_without_lane_0", M,
     '    if (((apart | ends | ((a0 + (size - 1)) ^ a0)) >> 12) == 0',
     '    if (((apart | ends) >> 12) == 0'),
    ("read_run_is_its_first_lane", M,
     '      spec_read_in(s, worker, a0, uint64_t{count} * size);',
     '      spec_read_in(s, worker, a0, size);'),
    ("write_run_is_its_first_lane", M,
     '      ++static_cast<Spec::Worker*>(worker)->st.runs_as_one;\n      spec_write_in(s, worker, a0, uint64_t{count} * size);\n      return;\n    }\n  }\n  for (size_t i = 0; i < count; ++i) spec_write_in(s, worker, addrs[i], size);',
     '      ++static_cast<Spec::Worker*>(worker)->st.runs_as_one;\n      spec_write_in(s, worker, a0, size);\n      return;\n    }\n  }\n  for (size_t i = 0; i < count; ++i) spec_write_in(s, worker, addrs[i], size);'),
    ("gapped_lanes_taken_as_a_run", M,
     '    for (size_t i = 1; i < count; ++i) gaps |= addrs[i] - (a0 + i * size);\n    if (gaps == 0 &&',
     '    for (size_t i = 1; i < count; ++i) gaps |= addrs[i] - (a0 + i * size);\n    if ('),
    ("any_stride_taken_as_a_run", M,
     '  if (count > 1 && stride == static_cast<int64_t>(size) && a0 + uint64_t{count} * size > a0) {',
     '  if (count > 1 && a0 + uint64_t{count} * size > a0) {'),
    ("coarse_stop_not_said", M,
     '    out.coarse = true;',
     '    out.coarse = false;'),
    ("coarse_word_stop_not_said", M,
     '      if (s.pstate[slot].load(std::memory_order_acquire) == Spec::kWordsCoarse)\n        s.refuse_coarse(',
     '      if (false)\n        s.refuse_coarse('),
    ("page_states_not_cleared", M,
     '    for (const uint64_t slot : w.noted) s.pstate[slot].store(0, std::memory_order_relaxed);',
     '    (void)0;'),
    ("plan_by_word_ignored", M,
     '      if (lo < to && from < hi) return true;',
     '      if (lo < to && from < hi) return false;'),
    ("plan_pages_off_ignored", M,
     '    if (!plan_pages) return true;',
     '    if (!plan_pages) return false;'),
    ("no_second_trial", X,
     '    if (out.committed || !out.coarse || !plan.pages) break;',
     '    break;'),
    ("second_trial_with_the_same_plan", X,
     '    mem_.spec_begin(threads, &plan);',
     '    mem_.spec_begin(threads, nullptr);'),
    ("second_trial_keeps_the_first_one_s_counters", X,
     '    sums.assign(threads, exec::LaunchStats{});',
     '    if (sums.empty()) sums.assign(threads, exec::LaunchStats{});'),
    # ---- round sp47's, on this round's lines ----
    ("write_conflict_ignored", M,
     '      if (v != Spec::kMany && (v & Spec::kWrite)) spec_refuse("two blocks wrote the same bytes");',
     '      break;'),
    ("read_of_written_ignored", M,
     '          refuse("a block read what another block wrote");   // (a writer\'s mark is always exact: no coarse stop here)',
     '          break;'),
    ("readers_not_merged", M,
     '        } else if (word.compare_exchange_weak(v, kMany, std::memory_order_acq_rel)) {',
     '        } else if (word.compare_exchange_weak(v, tag, std::memory_order_acq_rel)) {'),
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
    ("stopped_trial_kept", X,
     '  if (!out.committed) {\n    next.store(0);   // every block again, in order\n    return false;\n  }',
     '  if (false) {\n    next.store(0);   // every block again, in order\n    return false;\n  }'),
]
FORMS = (("default", {}), ("tier0", {"VGPU_TIER": "0"}), ("fastpath0", {"VGPU_FASTPATH": "0"}), ("pages0", {"VGPU_EXACT_PAGES": "0"}))


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def build(bld):
    r = sh("nice -n 19 cmake --build %s -j3 --target test_exact_threads" % bld)
    return r.returncode == 0, r.stdout[-2000:]


def run(bld):
    """-> list of the forms that failed"""
    failed = []
    for form, env in FORMS:
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
            print("%-44s NOT APPLIED (%d matches)" % (name, text.count(old)), flush=True)
            continue
        open(p, "w").write(text.replace(old, new))
        try:
            ok, log = build(bld)
            if not ok:
                print("%-44s DOES NOT BUILD\n%s" % (name, log), flush=True)
                continue
            failed = run(bld)
            print("%-44s %s" % (name, "killed by " + ", ".join(failed) if failed else "SURVIVED"), flush=True)
        finally:
            open(p, "w").write(text)
    if not want:
        ok, log = build(bld)
        failed = run(bld) if ok else ["build"]
        print("%-44s %s" % ("(no mutant)", "passes" if not failed else "FAILS: " + ", ".join(failed)), flush=True)


if __name__ == "__main__":
    main()
