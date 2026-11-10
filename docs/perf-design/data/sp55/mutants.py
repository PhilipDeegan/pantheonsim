#!/usr/bin/env python3
"""Round sp55 mutation checks: each mutant is one wrong line in the exact-threads code; test_exact_threads must fail on it.

  mutants.py <src> <build> [name ...]
For each mutant: patch the source in place, rebuild test_exact_threads, run it in its four forms (default, VGPU_TIER=0,
VGPU_FASTPATH=0, VGPU_EXACT_PAGES=0), put the source back. Prints "killed by <forms>" or "SURVIVED" per mutant, then rebuilds
the unmutated test and requires it to pass. The first groups are this round's mechanisms (the note by page and the second trial; what a
worker remembers of pages, runs of words and pages of words; two words taken at once); the last is round sp47's mutants on the lines
this round moved or rewrote, and a sample of those it did not touch."""
import fcntl, os, subprocess, sys

LOCK = "/tmp/pantheonsim-builder.lock"   # the machine lock: held for one mutant's build and runs at a time

M = "src/core/memory_spec.cpp"
C = "src/core/memory.cpp"
X = "src/sass/exec.cpp"
SEEN = "    if (seen.slot == slot && (seen.tag == w.tag || seen.tag == Spec::kMany)) [[likely]] continue;"
KNOWN = "      if (runs[i].tag == tag && g0 >= runs[i].g0 && g1 <= runs[i].g1) return true;"
JOIN = "      if (runs[i].tag == tag && g0 <= runs[i].g1 + 1 && g1 + 1 >= runs[i].g0) {"
WP = "        if (!(v == Spec::kMany || v == tag || v == (tag | Spec::kWrite))) break;"
MUTANTS = [
    # ---- this round: the note by page ----
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
    # ---- this round: what a worker remembers of words ----
    ("run_for_any_block", M, KNOWN, "      if (g0 >= runs[i].g0 && g1 <= runs[i].g1) return true;"),
    ("run_known_by_its_first_word", M, KNOWN, "      if (runs[i].tag == tag && g0 >= runs[i].g0 && g0 <= runs[i].g1) return true;"),
    ("runs_joined_across_a_gap", M, JOIN, "      if (runs[i].tag == tag) {"),
    ("read_run_remembered_before_it_is_noted", M,
     '    const uint64_t g0 = first % kPageWords, g1 = last % kPageWords;\n    w.st.words_read += g1 - g0 + 1;',
     '    const uint64_t g0 = first % kPageWords, g1 = last % kPageWords;\n    learn(w.read, w.read_next, tag, first, last);\n    w.st.words_read += g1 - g0 + 1;'),
    ("write_run_remembered_before_it_is_taken", M,
     '  const uint64_t mine2 = uint64_t{mine} | uint64_t{mine} << 32;',
     '  const uint64_t mine2 = uint64_t{mine} | uint64_t{mine} << 32;\n  Spec::learn(w.wrote, w.wrote_next, w.tag, g0, g1);'),
    ("readable_words_of_another_block", M, WP, "        if (v != Spec::kMany && (v & Spec::kWrite) && v != (tag | Spec::kWrite)) break;"),
    ("readable_words_not_looked_at", M, WP, "        if (false) break;"),
    ("word_page_of_another_page", M,
     '    if (wp.slot == slot) {',
     '    if (wp.page) {'),
    # ---- this round: two words at once ----
    ("pair_written_when_the_first_is_free", M,
     '      if ((a == 0 || a == w.tag) && (b == 0 || b == w.tag) &&',
     '      if ((a == 0 || a == w.tag) &&'),
    ("pair_written_when_the_second_is_free", M,
     '      if ((a == 0 || a == w.tag) && (b == 0 || b == w.tag) &&',
     '      if ((b == 0 || b == w.tag) &&'),
    ("pair_written_over_any_reader", M,
     '      const uint32_t a = static_cast<uint32_t>(v2), b = static_cast<uint32_t>(v2 >> 32);',
     '      const uint32_t a = static_cast<uint32_t>(v2) & Spec::kWrite ? 1u << 30 : 0, b = static_cast<uint32_t>(v2 >> 32) & Spec::kWrite ? 1u << 30 : 0;'),
    ("pair_written_not_kept", M,
     '        w.st.words_written_first += 2;\n        if (run == ~uint64_t{0}) run = g;',
     '        w.st.words_written_first += 2;'),
    ("pair_read_by_its_first_word", M,
     '        if (mine(static_cast<uint32_t>(v2)) && mine(static_cast<uint32_t>(v2 >> 32))) {',
     '        if (mine(static_cast<uint32_t>(v2))) {'),
    ("pair_read_by_its_second_word", M,
     '        if (mine(static_cast<uint32_t>(v2)) && mine(static_cast<uint32_t>(v2 >> 32))) {',
     '        if (mine(static_cast<uint32_t>(v2 >> 32))) {'),
    ("pair_already_written_by_its_first_word", M,
     '      if (v2 == mine2) {',
     '      if (static_cast<uint32_t>(v2) == mine) {'),
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
    ("undo_log_not_emptied", M,
     '    w.kept_at.clear();\n    w.kept.clear();\n    // (the log\'s memory stays for the next trial, unless it grew large)',
     '    // (the log\'s memory stays for the next trial, unless it grew large)'),
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
        lock = open(LOCK, "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
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
            lock.close()
    if not want:
        lock = open(LOCK, "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        ok, log = build(bld)
        failed = run(bld) if ok else ["build"]
        print("%-44s %s" % ("(no mutant)", "passes" if not failed else "FAILS: " + ", ".join(failed)), flush=True)


if __name__ == "__main__":
    main()
