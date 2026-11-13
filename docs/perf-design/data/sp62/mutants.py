#!/usr/bin/env python3
"""Round sp62 mutation checks of the line record (round sp58's road: mutants.py of data/sp58).

  nice -n 19 mutants.py <src> <build> <work> [name ... | controls]
Each mutant is one wrong change in src/core/memory_spec.cpp. A copy of the file with the change is compiled twice with the build's
own commands (-O1 for -O3), outside both trees: as the library's object, linked ahead of libvgpu.a into test_exact_threads; and as
test_exact_lines_race's own object (with its test hooks), linked into that test. A mutant is killed by the first test that fails:
test_exact_lines_race first (three runs), then test_exact_threads in its four forms (default, VGPU_TIER=0, VGPU_FASTPATH=0,
VGPU_EXACT_PAGES=0; three runs each while it passes). The unchanged file goes through the same road first and must pass
("controls"). The machine lock is held over several mutants at a time, let go after about 150 s."""
import fcntl, os, re, subprocess, sys, time

LOCK = "/tmp/pantheonsim-builder.lock"
RUNS = 3
M = "src/core/memory_spec.cpp"
TR, TX = "test_exact_lines_race", "test_exact_threads"
FORMS = {TR: (("default", {}),),
         TX: (("default", {}), ("tier0", {"VGPU_TIER": "0"}), ("fastpath0", {"VGPU_FASTPATH": "0"}), ("pages0", {"VGPU_EXACT_PAGES": "0"}))}
OUT = "      words[i].store((wr >> i & 1) ? (b | kWrite) : (r >> i & 1) ? b : 0, std::memory_order_relaxed);"
MUTANTS = [
    # ---- the two that survived round sp58 ----
    ("written_out_without_taking_the_record", [("    if (!line.take(v, kLineBusy)) return;   // (the record moved",
                                                "    line.store(kLineBusy);   // (the record moved")]),
    ("busy_line_read_as_by_word", [("      if (o == kLineByWord) return read_words(w, page->words + l * kLineWords, m);\n      if (o == kLineBusy) {",
                                    "      if (o == kLineByWord || o == kLineBusy) return read_words(w, page->words + l * kLineWords, m);\n      if (o == kLineBusy) {")]),
    # ---- the other windows of the record ----
    ("busy_line_written_as_by_word", [("      if (o == Spec::kLineByWord) {\n        taken = s.write_words(w, page, slot, l, m);",
                                       "      if (o == Spec::kLineByWord || o == Spec::kLineBusy) {\n        taken = s.write_words(w, page, slot, l, m);")]),
    ("first_write_without_taking_the_record", [("        if (!line.take(v, Spec::kLineBusy)) continue;\n        VGPU_SPEC_HOOK(kKeepBusy);", "        VGPU_SPEC_HOOK(kKeepBusy);")]),
    ("busy_line_read_without_waiting", [("        wait_busy(w);\n        v = line.load();\n        continue;", "        v = line.load();\n        continue;")]),
    ("owner_s_read_note_stored_not_taken", [("        if (!line.take(v, v | as_read(fresh))) continue;", "        line.store(v | as_read(fresh));")]),
    ("owner_s_write_note_stored_not_taken", [("        if (fresh && !line.take(v, v | Spec::as_written(fresh))) continue;", "        if (fresh) line.store(v | Spec::as_written(fresh));")]),
    ("first_reader_stores_the_record", [("        if (!line.take(v, uint64_t{tag} | as_read(m))) continue;", "        line.store(uint64_t{tag} | as_read(m));")]),
    ("record_said_by_word_before_the_words", [("    for (uint64_t i = 0; i < kLineWords; ++i) {\n      words[i].store((wr", "    line.store(kLineByWord);\n    for (uint64_t i = 0; i < kLineWords; ++i) {\n      words[i].store((wr")]),
    ("written_out_as_read", [(OUT, "      words[i].store((wr >> i & 1) ? b : (r >> i & 1) ? b : 0, std::memory_order_relaxed);")]),
    ("read_marks_not_written_out", [(OUT, "      words[i].store((wr >> i & 1) ? (b | kWrite) : 0, std::memory_order_relaxed);")]),
    ("line_read_of_written_ignored", [('      if (m & written_bits(v)) refuse("a block read what another block wrote");\n      to_words(w, page, l, v);', "      to_words(w, page, l, v);")]),
    # ---- round sp62: a page whose lines keep being written out, and two words at once ----
    ("straight_over_another_block_s_record", [("      if (v == 0 && s.lines_to_words(page)) [[unlikely]] {", "      if (o != Spec::kLineByWord && o != Spec::kLineBusy && o != w.tag && s.lines_to_words(page)) [[unlikely]] {")]),
    ("outs_not_cleared", [("      page->outs = 0;\n", "")]),
    ("outs_not_counted", [("    __atomic_fetch_add(&page->outs, 1, __ATOMIC_RELAXED);\n", "")]),
    ("pair_read_taken_when_only_the_first_is_free", [("        if (v2 == 0 && __atomic_compare_exchange_n(two, &v2, uint64_t{tag} | uint64_t{tag} << 32,", "        if (static_cast<uint32_t>(v2) == 0 && __atomic_compare_exchange_n(two, &v2, uint64_t{tag} | uint64_t{tag} << 32,")]),
    ("pair_readable_by_its_first_word", [("        if (readable(static_cast<uint32_t>(v2)) && readable(static_cast<uint32_t>(v2 >> 32))) {", "        if (readable(static_cast<uint32_t>(v2))) {")]),
    ("pair_written_when_only_the_first_is_free", [("        if ((a == 0 || a == tag) && (b == 0 || b == tag) && __atomic", "        if ((a == 0 || a == tag) && __atomic")]),
    ("pair_written_when_only_the_second_is_free", [("        if ((a == 0 || a == tag) && (b == 0 || b == tag) && __atomic", "        if ((b == 0 || b == tag) && __atomic")]),
    ("pair_taken_for_written_by_its_first_word", [("        if (v2 == mine2) {", "        if (static_cast<uint32_t>(v2) == mine) {")]),
    ("written_pair_not_kept", [("          taken |= 3u << i;", "          taken |= 1u << i;")]),
]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def commands(bld, target):
    return sh("ninja -C %s -t commands %s" % (bld, target)).stdout.strip().splitlines()


def compile_as(src, bld, work, msrc, o, obj):
    full = os.path.join(src, M)
    cmd = [c for c in commands(bld, o) if (" -o " + o + " ") in c][-1]
    cmd = re.sub(r" -MD -MT \S+ -MF \S+", "", cmd)
    assert (" -c " + full) in cmd, cmd
    cmd = cmd.replace(" -o " + o + " ", " -o %s " % obj).replace(" -c " + full, " -iquote %s -c %s" % (os.path.dirname(full), msrc))
    cmd = cmd.replace(" -O3 ", " -O1 ")
    r = sh(cmd, cwd=bld)
    return r.returncode == 0, r.stdout[-2000:]


def build(src, bld, work, text):
    """-> (ok, log): both tests linked with the mutated file, in `work`."""
    msrc = os.path.join(work, "mut_memory_spec.cpp")
    open(msrc, "w").write(text)
    # the library's object, ahead of libvgpu.a in test_exact_threads
    ok, log = compile_as(src, bld, work, msrc, "CMakeFiles/vgpu.dir/%s.o" % M, os.path.join(work, "mut_lib.o"))
    if not ok:
        return False, log
    link = [c for c in commands(bld, TX) if (" -o " + TX + " ") in c][-1]
    assert " libvgpu.a" in link, link
    r = sh(link.replace(" -o " + TX + " ", " %s -o %s " % (os.path.join(work, "mut_lib.o"), os.path.join(work, TX))), cwd=bld)
    if r.returncode:
        return False, r.stdout[-2000:]
    # the race test's own object (its hooks), in its place
    o = "CMakeFiles/%s.dir/%s.o" % (TR, M)
    ok, log = compile_as(src, bld, work, msrc, o, os.path.join(work, "mut_race.o"))
    if not ok:
        return False, log
    link = [c for c in commands(bld, TR) if (" -o " + TR + " ") in c][-1]
    assert (" " + o + " ") in link, link
    r = sh(link.replace(" " + o + " ", " %s " % os.path.join(work, "mut_race.o")).replace(" -o " + TR + " ", " -o %s " % os.path.join(work, TR)), cwd=bld)
    return r.returncode == 0, r.stdout[-2000:]


def run(work, every=False):
    """-> the first (test, form) that failed, or all of them with `every`"""
    failed = []
    for test in (TR, TX):
        for form, env in FORMS[test]:
            e = dict(os.environ)
            e.update(env)
            for i in range(1 if every else RUNS):
                try:
                    r = sh("%s/%s" % (work, test), env=e, timeout=600)
                    if r.returncode != 0:
                        failed.append("%s %s%s" % (test, form, "" if i == 0 else " (run %d)" % (i + 1)))
                        break
                except subprocess.TimeoutExpired:
                    failed.append("%s %s (timeout)" % (test, form))
                    break
            if failed and not every:
                return failed
    return failed


def main():
    src, bld, work = sys.argv[1], sys.argv[2], sys.argv[3]
    want = sys.argv[4:]
    os.makedirs(work, exist_ok=True)
    todo = ([("(no mutant)", None)] if not want or "controls" in want else []) + [m for m in MUTANTS if not want or m[0] in want]
    lock, since = open(LOCK, "w"), None
    for name, edits in todo:
        if since is not None and time.monotonic() - since > 150:
            fcntl.flock(lock, fcntl.LOCK_UN)
            since = None
            time.sleep(2)
        if since is None:
            fcntl.flock(lock, fcntl.LOCK_EX)
            since = time.monotonic()
        text = open(os.path.join(src, M)).read()
        bad = [old for old, _ in edits or [] if text.count(old) != 1]
        if bad:
            print("%-42s NOT APPLIED (%d matches of %r)" % (name, text.count(bad[0]), bad[0][:50]), flush=True)
            continue
        for old, new in edits or []:
            text = text.replace(old, new)
        ok, log = build(src, bld, work, text)
        if not ok:
            print("%-42s DOES NOT BUILD\n%s" % (name, log), flush=True)
            continue
        failed = run(work, every=edits is None)
        if edits is None:
            print("%-42s %s" % (name, "passes everywhere" if not failed else "FAILS: " + ", ".join(failed)), flush=True)
        else:
            print("%-42s %s" % (name, "killed by " + failed[0] if failed else "SURVIVED"), flush=True)
    if since is not None:
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
