#!/usr/bin/env python3
"""Round sp55 mutation checks: each mutant is one wrong line in the exact-threads code; test_exact_threads must fail on it.

  mutants.py <src> <build> <work> [name ...]
For each mutant: a copy of the source file with the one line changed is compiled with the build's own command into <work>, and
test_exact_threads is linked with that object ahead of libvgpu.a (the linker takes the object's definitions and leaves the
library's member out), so neither the source tree nor the build tree is touched. The test is then run in its four forms (default,
VGPU_TIER=0, VGPU_FASTPATH=0, VGPU_EXACT_PAGES=0). Prints "killed by <the first form that failed>" or "SURVIVED" (all four
passed) per mutant, and first of all "(no mutant: <file>)" for each of the three files: the unchanged file through the same road,
which must pass in the four forms (`mutants.py ... controls` runs these alone).
The first groups are this round's mechanisms (the note by page and the second trial; what a worker remembers of pages, runs of
words and pages of words; two words taken at once); the last is round sp47's mutants on the lines this round moved or rewrote,
and a sample of those it did not touch. The machine lock is held for one mutant at a time."""
import fcntl, os, re, subprocess, sys, time

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
    ("few_blocks_noted_by_page", X,
     '    if (pages_off || units < kExactPageBlocks) plan.pages = false;',
     '    if (pages_off) plan.pages = false;'),
    ("second_trial_with_the_same_plan", X,
     '    mem_.spec_begin(threads, &plan);',
     '    mem_.spec_begin(threads, nullptr);'),
    # (survives, and is equivalent: every worker of a trial writes its sum into its slot before the trial ends, so no slot is read with the first trial's value)
    ("second_trial_keeps_the_first_one_s_counters", X,
     '    sums.assign(threads, exec::LaunchStats{});',
     '    if (sums.empty()) sums.assign(threads, exec::LaunchStats{});'),
    # ---- this round: what a worker remembers of words ----
    ("run_for_any_block", M, KNOWN, "      if (g0 >= runs[i].g0 && g1 <= runs[i].g1) return true;"),
    ("run_known_by_its_first_word", M, KNOWN, "      if (runs[i].tag == tag && g0 >= runs[i].g0 && g0 <= runs[i].g1) return true;"),
    ("runs_joined_across_a_gap", M, JOIN, "      if (runs[i].tag == tag) {"),
    # (survives, and is equivalent: a note that fails stops the trial, and every later hook of a stopped trial throws before it asks the memory)
    ("read_run_remembered_before_it_is_noted", M,
     '    const uint64_t g0 = first % kPageWords, g1 = last % kPageWords;\n    w.st.words_read += g1 - g0 + 1;',
     '    const uint64_t g0 = first % kPageWords, g1 = last % kPageWords;\n    learn(w.read, w.read_next, tag, first, last);\n    w.st.words_read += g1 - g0 + 1;'),
    # (survives, and is equivalent: as the one above)
    ("write_run_remembered_before_it_is_taken", M,
     '  const uint64_t mine2 = uint64_t{mine} | uint64_t{mine} << 32;',
     '  const uint64_t mine2 = uint64_t{mine} | uint64_t{mine} << 32;\n  Spec::learn(w.wrote, w.wrote_next, w.tag, g0, g1);'),
    ("readable_words_of_another_block", M, WP, "        if (v != Spec::kMany && (v & Spec::kWrite) && v != (tag | Spec::kWrite)) break;"),
    ("readable_words_not_looked_at", M, WP, "        if (false) break;"),
    ("lone_word_taken_whatever_it_holds", M,
     '        if (wp.page[g].compare_exchange_strong(none, tag, std::memory_order_acq_rel)) {',
     '        if (wp.page[g].compare_exchange_strong(none, tag, std::memory_order_acq_rel) || true) {'),
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
    # (survives, and is equivalent: the log is emptied again when the next trial begins)
    # (kept in the list: the two places must not both go)
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


def compile_command(src, bld, path):
    """The build's command for `path` (the rule of CMakeFiles/vgpu.dir/build.make with the variables of flags.make), with {src} and
    {obj} for the file and the object."""
    obj = "CMakeFiles/vgpu.dir/%s.o" % path
    full = os.path.join(src, path)
    flags = {}
    for l in open(os.path.join(bld, "CMakeFiles/vgpu.dir/flags.make")):
        if " = " in l and not l.startswith("#"):
            k, v = l.rstrip("\n").split(" = ", 1)
            flags[k] = v
    cmd = [l.strip() for l in open(os.path.join(bld, "CMakeFiles/vgpu.dir/build.make")) if l.startswith("\t") and (" -o " + obj + " ") in l and (" -c " + full) in l][0]
    for k, v in flags.items():
        cmd = cmd.replace("$(%s)" % k, v)
    assert "$(" not in cmd, cmd
    cmd = re.sub(r" -MD -MT \S+ -MF \S+", "", cmd)
    return cmd.replace(" -o " + obj + " ", " -o {obj} ").replace(" -c " + full, " -iquote %s -c {src}" % os.path.dirname(full))


def build(src, bld, work, path, text):
    """Compiles `text` as `path` and links the test with it. -> (ok, log)"""
    msrc = os.path.join(work, "mut_" + os.path.basename(path))
    open(msrc, "w").write(text)
    obj = os.path.join(work, "mut.o")
    r = sh("nice -n 19 " + compile_command(src, bld, path).format(src=msrc, obj=obj), cwd=bld)
    if r.returncode != 0:
        return False, r.stdout[-2000:]
    link = open(os.path.join(bld, "CMakeFiles/test_exact_threads.dir/link.txt")).read().strip()
    assert " -o test_exact_threads " in link and " libvgpu.a" in link, link
    link = link.replace(" -o test_exact_threads ", " %s -o %s " % (obj, os.path.join(work, "test_exact_threads")))
    r = sh("nice -n 19 " + link, cwd=bld)
    return r.returncode == 0, r.stdout[-2000:]


def run(work, every=False):
    """-> the forms that failed: the first one (a mutant is killed by one), or all of them with `every`"""
    failed = []
    for form, env in FORMS:
        e = dict(os.environ)
        e.update(env)
        try:
            r = sh("nice -n 19 %s/test_exact_threads" % work, env=e, timeout=600)
            if r.returncode != 0:
                failed.append(form)
        except subprocess.TimeoutExpired:
            failed.append(form + "(timeout)")
        if failed and not every:
            break
    return failed


def main():
    src, bld, work = sys.argv[1], sys.argv[2], sys.argv[3]
    want = sys.argv[4:]
    os.makedirs(work, exist_ok=True)
    controls = [("(no mutant: %s)" % p, p, None, None) for p in (M, C, X)]
    todo = [c for c in controls if not want or "controls" in want] + [m for m in MUTANTS if not want or m[0] in want]
    for name, path, old, new in todo:
        text = open(os.path.join(src, path)).read()
        if old is not None:
            if text.count(old) != 1:
                print("%-44s NOT APPLIED (%d matches)" % (name, text.count(old)), flush=True)
                continue
            text = text.replace(old, new)
        with open(LOCK, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ok, log = build(src, bld, work, path, text)
            if not ok:
                print("%-44s DOES NOT BUILD\n%s" % (name, log), flush=True)
                continue
            failed = run(work, every=old is None)
        if old is None:
            print("%-44s %s" % (name, "passes in the four forms" if not failed else "FAILS: " + ", ".join(failed)), flush=True)
        else:
            print("%-44s %s" % (name, "killed by " + failed[0] if failed else "SURVIVED"), flush=True)
        time.sleep(2)   # (another holder of the lock gets in)


if __name__ == "__main__":
    main()
