#!/usr/bin/env python3
"""Round sp58 mutation checks: each mutant is one wrong change in this round's code; its test must fail on it.

  mutants.py <src> <build> <work> [name ... | controls]
The road of round sp55's mutants.py for a Ninja build tree: a copy of the source file with the change is compiled with the build's
own command (ninja -t commands, with -O1 for -O3: a third of the time) into <work>, and the test (test_exact_threads, or test_worker_pool for the width rule) is linked
with that object ahead of libvgpu.a, so neither tree is touched. test_exact_threads is run in its four forms (default,
VGPU_TIER=0, VGPU_FASTPATH=0, VGPU_EXACT_PAGES=0); a mutant is killed by the first form that fails. Each run is repeated
(RUNS times) while it passes: three of the mutants are of a race between two threads, which one run may not meet. First of all
the unchanged files go through the same road and must pass ("controls"). The machine lock is held for one mutant at a time."""
import fcntl, os, re, subprocess, sys, time

LOCK = "/tmp/pantheonsim-builder.lock"
RUNS = 3
M = "src/core/memory_spec.cpp"
X = "src/sass/exec.cpp"
P = "src/exec/worker_pool.cpp"
T, TP = "test_exact_threads", "test_worker_pool"
FORMS = {T: (("default", {}), ("tier0", {"VGPU_TIER": "0"}), ("fastpath0", {"VGPU_FASTPATH": "0"}), ("pages0", {"VGPU_EXACT_PAGES": "0"})),
         TP: (("default", {}),)}
MET_R = '      if (m & written_bits(v)) refuse("a block read what another block wrote");\n      to_words(w, page, l, v);'
OUT = "      words[i].store((wr >> i & 1) ? (b | kWrite) : (r >> i & 1) ? b : 0, std::memory_order_relaxed);"
MUTANTS = [
    # ---- lines: a second block at another block's record ----
    ("line_read_of_written_ignored", M, [(MET_R, "      to_words(w, page, l, v);")], T),
    ("line_write_of_written_ignored", M, [('    if (m & written_bits(v)) refuse("two blocks wrote the same bytes");\n    if (const uint32_t r = m & read_bits(v)) {',
                                           '    if (const uint32_t r = m & read_bits(v)) {')], T),
    ("line_write_of_read_ignored", M, [("    if (const uint32_t r = m & read_bits(v)) {", "    if (const uint32_t r = m & read_bits(v) & 0) {")], T),
    ("line_coarse_stop_not_said", M, [("      if (pstate[slot].load(std::memory_order_acquire) == kWordsCoarse)\n        refuse_coarse(base + (slot * kPageWords + l * kLineWords + static_cast",
                                       "      if (false)\n        refuse_coarse(base + (slot * kPageWords + l * kLineWords + static_cast")], T),
    ("line_read_by_any_block", M, [("      if (owner(v) == tag) [[likely]] {\n        const uint32_t fresh = m & ~(read_bits(v) | written_bits(v));",
                                    "      if (v != 0 && owner(v) < kLineBusy) [[likely]] {\n        const uint32_t fresh = m & ~(read_bits(v) | written_bits(v));")], T),
    ("line_written_by_any_block", M, [("      if (o == w.tag && Spec::written_bits(v)) [[likely]] {", "      if (o < Spec::kLineBusy && Spec::written_bits(v)) [[likely]] {")], T),
    ("untouched_line_s_first_writer_any_block", M, [("      if (v == 0 || o == w.tag) {\n        // The block's first write", "      if (o < Spec::kLineBusy) {\n        // The block's first write")], T),
    # ---- lines: the record written out to words ----
    ("written_out_as_read", M, [(OUT, "      words[i].store((wr >> i & 1) ? b : (r >> i & 1) ? b : 0, std::memory_order_relaxed);")], T),
    ("read_marks_not_written_out", M, [(OUT, "      words[i].store((wr >> i & 1) ? (b | kWrite) : 0, std::memory_order_relaxed);")], T),
    ("written_out_without_taking_the_record", M, [("    if (!line.take(v, kLineBusy)) return;   // (the record moved",
                                                   "    line.store(kLineBusy);   // (the record moved")], T),
    ("busy_line_read_as_by_word", M, [("      if (o == kLineByWord) return read_words(w, page->words + l * kLineWords, m);\n      if (o == kLineBusy) {",
                                       "      if (o == kLineByWord || o == kLineBusy) return read_words(w, page->words + l * kLineWords, m);\n      if (o == kLineBusy) {")], T),
    ("busy_line_written_as_by_word", M, [("      if (o == Spec::kLineByWord) {\n        taken = s.write_words(w, page, slot, l, m);", "      if (o == Spec::kLineByWord || o == Spec::kLineBusy) {\n        taken = s.write_words(w, page, slot, l, m);")], T),
    # ---- lines: the owner's own moves, and what is kept ----
    ("first_write_forgets_the_block_s_reads", M, [("        line.store((v ? v : uint64_t{w.tag}) | Spec::as_written(m));", "        line.store(uint64_t{w.tag} | Spec::as_written(m));")], T),
    ("line_not_kept", M, [("        spec_keep(s.base + line0 * Spec::kGranule, Spec::kLineWords * Spec::kGranule);", "        (void)0;")], T),
    ("half_a_line_kept", M, [("        spec_keep(s.base + line0 * Spec::kGranule, Spec::kLineWords * Spec::kGranule);", "        spec_keep(s.base + line0 * Spec::kGranule, Spec::kLineWords * Spec::kGranule / 2);")], T),
    ("line_kept_again", M, [("      if (o == w.tag && Spec::written_bits(v)) [[likely]] {", "      if (false) {")], T),
    ("later_words_of_a_line_not_marked", M, [("        if (fresh && !line.take(v, v | Spec::as_written(fresh))) continue;", "        if (false) continue;")], T),
    ("first_write_without_taking_the_record", M, [("        if (!line.take(v, Spec::kLineBusy)) continue;\n        keep();", "        keep();")], T),
    ("words_of_a_shared_line_not_kept", M, [("          taken |= 1u << i;", "          (void)0;")], T),
    ("pieces_joined_across_the_scout_s", M, [("    if (w.kept_at.size() > w.sealed && w.kept_at.back()", "    if (!w.kept_at.empty() && w.kept_at.back()")], T),
    ("pieces_joined_that_do_not_touch", M, [("w.kept_at.back().first + w.kept_at.back().second == addr) w.kept_at.back().second += len;", "true) w.kept_at.back().second += len;")], T),
    ("kept_runs_joined_across_a_gap", M, [("      if (run != ~uint64_t{0} && run_end == line0 + lo) {", "      if (run != ~uint64_t{0}) {")], T),
    ("bits_of_a_line_not_placed", M, [("return (0xFFFFu >> (15 - (last - first))) << first;", "return (0xFFFFu >> (15 - (last - first)));")], T),
    ("own_reads_forgotten_on_write", M, [("mine->lines[i].store(uint64_t{tag} | as_read(0xFFFFu));", "mine->lines[i].store(0);")], T),
    ("records_not_cleared", M, [("      std::memset(static_cast<void*>(page->lines), 0, sizeof(page->lines));", "      (void)0;")], T),
    ("written_out_words_not_cleared", M, [("        if (Spec::owner(v) == Spec::kLineByWord) std::memset(", "        if (Spec::owner(v) == Spec::kLineByWord && false) std::memset(")], T),
    # ---- round sp47's and sp55's, on lines this round rewrote ----
    ("word_read_of_written_ignored", M, [('        } else if (v & kWrite) {\n          refuse("a block read what another block wrote");', '        } else if (false) {\n          refuse("a block read what another block wrote");')], T),
    ("word_readers_not_merged", M, [("        } else if (word.compare_exchange_weak(v, kMany, std::memory_order_acq_rel)) {\n          ++w.st.words_read_shared;\n          break;",
                                     "        } else {\n          ++w.st.words_read_shared;\n          break;")], T),
    ("word_write_conflict_ignored", M, [('        if (v != kMany && (v & kWrite)) refuse("two blocks wrote the same bytes");', '        if (v != kMany && (v & kWrite)) break;')], T),
    ("word_write_of_read_ignored", M, [('          refuse_coarse(base + (slot * kPageWords + l * kLineWords + i) * kGranule, "a block wrote what another block read");\n        refuse("a block wrote what another block read");',
                                        '          refuse_coarse(base + (slot * kPageWords + l * kLineWords + i) * kGranule, "a block wrote what another block read");\n        break;')], T),
    ("no_write_back", M, [("        write(w.kept_at[i].first, w.kept.data() + at, w.kept_at[i].second);\n        at += w.kept_at[i].second;\n      }\n    }\n  }",
                           "        at += w.kept_at[i].second;\n      }\n    }\n  }")], T),
    # ---- the scout ----
    ("scout_page_states_not_forgotten", M, [("  for (const uint64_t slot : w.noted) s.pstate[slot].store(0, std::memory_order_relaxed);\n  for (const uint64_t slot : w.installed) {",
                                             "  for (const uint64_t slot : w.installed) {")], T),
    ("scout_lines_not_forgotten", M, [("    for (uint64_t l = 0; l < Spec::kPageLines; ++l) page->lines[l].store(0);\n    w.fresh.push_back(page);",
                                       "    s.top[slot].store(page, std::memory_order_release);")], T),
    ("scout_memory_not_forgotten", M, [("  w.forget();\n  return learned;", "  return learned;")], T),
    ("scout_bytes_not_put_back", M, [("  if (!out.committed && s.scout_pieces) {", "  if (false) {")], T),
    ("scout_bytes_put_back_first", M, [("      uint64_t at = first ? s.scout_bytes : 0;\n      for (size_t i = first ? s.scout_pieces : 0; i < w.kept_at.size(); ++i) {",
                                        "      uint64_t at = 0;\n      for (size_t i = 0; i < w.kept_at.size(); ++i) {"),
                                       ("  if (!out.committed && s.scout_pieces) {", "  if (false) {")], T),
    ("scout_learns_nothing", M, [("        learned.push_back(range);", "        (void)0;")], T),
    ("scout_learns_what_it_only_wrote", M, [("      if (std::find(read_in.begin(), read_in.end(), range) != read_in.end() &&", "      if (")], T),
    ("scout_learned_not_noted_by_word", M, [("  s.plan_words.insert(s.plan_words.end(), learned.begin(), learned.end());", "  (void)0;")], T),
    ("stopped_scout_learns", M, [("  if (s.stopped.load(std::memory_order_acquire)) return learned;", "  (void)0;")], T),
    ("scout_counters_not_added", X, [("  stats_.add(scout_sum);\n", "")], T),
    ("scout_block_run_again", X, [("        const uint64_t i = next.fetch_add(1);\n        try {\n          Block& blk = *held.block;\n          mem_.spec_block(i);\n          init_block(blk, i);\n          blk.worker = 0;",
                                   "        const uint64_t i = next.load();\n        try {\n          Block& blk = *held.block;\n          mem_.spec_block(i);\n          init_block(blk, i);\n          blk.worker = 0;")], T),
    ("scout_on_any_grid", X, [("!scout_off && units >= kExactScoutBlocksPerThread * threads;", "!scout_off;")], T),
    ("scout_every_launch", X, [("      kept.scouted = true;", "      kept.scouted = false;")], T),
    ("scout_switch_ignored", X, [("    const bool scout = plan.pages && !plan.scouted && !scout_off &&", "    const bool scout = plan.pages && !plan.scouted &&")], T),
    ("scout_learned_not_kept", X, [("        if (std::find(kept.by_word.begin(), kept.by_word.end(), range) == kept.by_word.end()) kept.by_word.push_back(range);\n      g_exact_scouted",
                                    "        (void)range;\n      g_exact_scouted")], T),
    # ---- the width rule of proved launches ----
    ("proved_verdict_by_the_best_launch", P, [("  const bool pays = proved ? t.wins * 2 > kTrialLaunches - 1 && t.sum <= (kTrialLaunches - 1) * 1000 : t.best < kTrialPaysBelowPermille;",
                                               "  const bool pays = t.best < kTrialPaysBelowPermille;")], TP),
    ("proved_sum_ignored", P, [("t.wins * 2 > kTrialLaunches - 1 && t.sum <= (kTrialLaunches - 1) * 1000 :", "t.wins * 2 > kTrialLaunches - 1 :")], TP),
    ("proved_wins_ignored", P, [("t.wins * 2 > kTrialLaunches - 1 && t.sum <= (kTrialLaunches - 1) * 1000 :", "t.sum <= (kTrialLaunches - 1) * 1000 :")], TP),
    ("proved_win_at_the_threshold", P, [("      t.wins += exact < kProvedPaysBelowPermille;", "      t.wins += exact <= kProvedPaysBelowPermille;")], TP),
    ("proved_worst_launch_not_capped", P, [("wall_ns / solo_ns >= kProvedWorstPermille / 1000 ? kProvedWorstPermille : wall_ns * 1000 / solo_ns;", "wall_ns * 1000 / solo_ns;")], TP),
    ("proved_first_launch_counted", P, [("    if (proved) {\n      const uint64_t exact", "  }\n  {\n    if (proved) {\n      const uint64_t exact")], TP),
    ("proved_counts_not_reset", P, [("  t.wins = t.sum = 0;", "  (void)0;")], TP),
    ("pool_figure_not_halved", P, [("known / 2 ? known / 2 : 1", "known")], TP),
    ("pool_figure_replaces_one_thread_s", P, [("  const bool keep = proved_ && used_ > 1 && mine && old && s.solo.load(std::memory_order_relaxed) && cost >= old - old / 4;",
                                               "  const bool keep = false;")], TP),
    ("too_high_a_figure_kept", P, [(" && s.solo.load(std::memory_order_relaxed) && cost >= old - old / 4;", " && s.solo.load(std::memory_order_relaxed);")], TP),
    ("one_thread_s_figure_not_marked", P, [("    s.solo.store(used_ <= 1, std::memory_order_relaxed);", "    s.solo.store(false, std::memory_order_relaxed);")], TP),
    ("pool_figure_marked_as_one_thread_s", P, [("    s.solo.store(used_ <= 1, std::memory_order_relaxed);", "    s.solo.store(true, std::memory_order_relaxed);")], TP),
    ("unmeasured_goes_wide_on_trial", P, [("  if (!by_hand && mine && known && measured) {", "  if (!by_hand && mine && known) {")], TP),
    ("proved_judged_as_unproved", P, [("    pool_trial_note(t, end_ns - begin_ns_, solo_ns_, proved_);", "    pool_trial_note(t, end_ns - begin_ns_, solo_ns_, false);")], TP),
]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def commands(bld, target):
    return sh("ninja -C %s -t commands %s" % (bld, target)).stdout.strip().splitlines()


def build(src, bld, work, path, text, test):
    """Compiles `text` as `path` and links `test` with it. -> (ok, log)"""
    msrc = os.path.join(work, "mut_" + os.path.basename(path))
    open(msrc, "w").write(text)
    obj, full, o = os.path.join(work, "mut.o"), os.path.join(src, path), "CMakeFiles/vgpu.dir/%s.o" % path
    cmd = [c for c in commands(bld, o) if (" -o " + o + " ") in c][-1]
    cmd = re.sub(r" -MD -MT \S+ -MF \S+", "", cmd)
    assert (" -c " + full) in cmd, cmd
    cmd = cmd.replace(" -o " + o + " ", " -o %s " % obj).replace(" -c " + full, " -iquote %s -c %s" % (os.path.dirname(full), msrc))
    cmd = cmd.replace(" -O3 ", " -O1 ")   # (the mutant's file only, to shorten the hold of the machine lock: the same code, less optimised)
    r = sh("nice -n 19 " + cmd, cwd=bld)
    if r.returncode != 0:
        return False, r.stdout[-2000:]
    link = [c for c in commands(bld, test) if (" -o " + test + " ") in c][-1]
    assert " libvgpu.a" in link, link
    link = link.replace(" -o " + test + " ", " %s -o %s " % (obj, os.path.join(work, test)))
    r = sh("nice -n 19 " + link, cwd=bld)
    return r.returncode == 0, r.stdout[-2000:]


def run(work, test, every=False):
    """-> the forms that failed: the first one (a mutant is killed by one), or all of them with `every`"""
    failed = []
    for form, env in FORMS[test]:
        e = dict(os.environ)
        e.update(env)
        for i in range(1 if every else RUNS):
            try:
                r = sh("nice -n 19 %s/%s" % (work, test), env=e, timeout=900)
                if r.returncode != 0:
                    failed.append(form if i == 0 else "%s (run %d)" % (form, i + 1))
                    break
            except subprocess.TimeoutExpired:
                failed.append(form + " (timeout)")
                break
        if failed and not every:
            break
    return failed


def main():
    src, bld, work = sys.argv[1], sys.argv[2], sys.argv[3]
    want = sys.argv[4:]
    os.makedirs(work, exist_ok=True)
    controls = [("(no mutant: %s)" % p, p, None, t) for p, t in ((M, T), (X, T), (P, TP))]
    todo = [c for c in controls if not want or "controls" in want] + [m for m in MUTANTS if not want or m[0] in want]
    for name, path, edits, test in todo:
        text = open(os.path.join(src, path)).read()
        bad = [old for old, _ in edits or [] if text.count(old) != 1]
        if bad:
            print("%-40s NOT APPLIED (%d matches of %r)" % (name, text.count(bad[0]), bad[0][:50]), flush=True)
            continue
        for old, new in edits or []:
            text = text.replace(old, new)
        with open(LOCK, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ok, log = build(src, bld, work, path, text, test)
            if not ok:
                print("%-40s DOES NOT BUILD\n%s" % (name, log), flush=True)
                continue
            failed = run(work, test, every=edits is None)
        if edits is None:
            print("%-40s %s" % (name, "passes in every form" if not failed else "FAILS: " + ", ".join(failed)), flush=True)
        else:
            print("%-40s %s" % (name, "killed by " + failed[0] if failed else "SURVIVED"), flush=True)
        time.sleep(2)   # (another holder of the lock gets in)


if __name__ == "__main__":
    main()
