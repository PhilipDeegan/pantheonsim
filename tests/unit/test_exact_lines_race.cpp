// The windows of a line's record (docs/exact-threads.md, "By line"), each held open. A line's record is moved by one thread while
// another comes to the line; the two orders that matter last a few instructions, which real threads meet too rarely for a test
// (round sp58: 9000 rounds of the race tests of test_exact_threads met neither). This test compiles src/core/memory_spec.cpp
// with VGPU_SPEC_TEST_HOOKS, which calls `spec_test::hook` at each window, and parks one thread there while others act:
//   * a block that is about to write a line's record out to its words (it has read the record, and not yet taken it), while the
//     line's own block notes more in it, or a third block writes the record out first;
//   * a block that has taken the record to write it out (the line is busy), while a third block reads or writes in the line;
//   * a line's block that has taken its record to keep the line's bytes before its first write in it, while another block
//     reads or writes in the line.
// In every case the mark of every word must survive (a later access is stopped exactly where the words stop it), a block that
// finds the line busy must wait (and must not stop where the words would not stop it), and a stopped trial is undone to the byte.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/memory.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace vgpu::spec_test {
extern void (*hook)(int at);
}  // namespace vgpu::spec_test

namespace {

constexpr int kOutBefore = 1, kOutBusy = 2, kKeepBusy = 3, kWaitBusy = 4, kOwnBefore = 5;
constexpr uint64_t kPage = 4096;

// The hook: the one thread that is armed parks at its window, once, until the test lets it go; a thread that waits at a busy line
// is counted.
std::atomic<int> g_park_at{0}, g_parked{0}, g_release{0}, g_waits{0};
thread_local bool t_armed = false;

void hook(int at) {
  if (at == kWaitBusy) {
    g_waits.fetch_add(1, std::memory_order_acq_rel);
    return;
  }
  if (!t_armed || at != g_park_at.load(std::memory_order_acquire)) return;
  t_armed = false;
  g_parked.store(1, std::memory_order_release);
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while (!g_release.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < until) std::this_thread::yield();
}

// Waits for `f` to hold, at most `ms` milliseconds.
bool within(int ms, const std::function<bool()>& f) {
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
  while (!f()) {
    if (std::chrono::steady_clock::now() >= until) return false;
    std::this_thread::yield();
  }
  return true;
}

struct Lines {
  MemoryManager mem{1 << 24};
  uint64_t a = 0, line = 0, size = 0;
  std::vector<uint8_t> init;
  MemoryManager::SpecPlan by_word;
  Lines() {
    size = 3 * kPage;
    a = mem.alloc(size);
    line = (a + kPage - 1) / kPage * kPage + 1024;   // (a line: a multiple of 64 from the device's base)
    init.resize(size);
    for (size_t i = 0; i < init.size(); ++i) init[i] = static_cast<uint8_t>(i * 29 + 3);
    mem.write(a, init.data(), size);
    by_word.pages = false;
  }
  std::vector<uint8_t> all() {
    std::vector<uint8_t> now(size);
    mem.read(a, now.data(), size);
    return now;
  }
};

// One access of block `block` on worker `worker`, on a thread of its own (the shadow's worker is the thread's); `armed` parks
// it at the window g_park_at. `stopped` says whether the trial stopped it; `done` is set when it returns.
struct Access {
  std::thread t;
  std::atomic<bool> stopped{false}, done{false};
  Access(Lines& d, unsigned worker, uint64_t block, uint64_t word, bool writes, bool armed = false) {
    t = std::thread([&d, this, worker, block, word, writes, armed] {
      t_armed = armed;
      d.mem.spec_enter(worker);
      d.mem.spec_block(block);
      uint32_t v = 0xC0DE0000u + static_cast<uint32_t>(block);
      try {
        if (writes) d.mem.write(d.line + 4 * word, &v, 4);
        else d.mem.read(d.line + 4 * word, &v, 4);
      } catch (const MemoryManager::SpecStop&) {
        stopped.store(true);
      }
      d.mem.spec_leave();
      t_armed = false;
      done.store(true, std::memory_order_release);
    });
  }
  ~Access() {
    if (t.joinable()) t.join();
  }
  bool end() {
    t.join();
    return stopped.load();
  }
};
bool access(Lines& d, unsigned worker, uint64_t block, uint64_t word, bool writes) { return Access(d, worker, block, word, writes).end(); }

void arm(int at) {
  g_park_at.store(at);
  g_parked.store(0);
  g_release.store(0);
  g_waits.store(0);
}

struct Hooked {
  Hooked() { spec_test::hook = &hook; }
  ~Hooked() {
    spec_test::hook = nullptr;
    g_release.store(1);
  }
};

void undone(Lines& d, const MemoryManager::SpecOutcome& out, const std::string& what) {
  VCHECK(!out.committed);
  if (d.all() != d.init) throw vtest::Failure(what + ": a stopped trial left bytes of the blocks behind");
}

}  // namespace

// Block 1 is about to write block 0's record out (it has read the record, which says block 0 read word 0) and is held there.
// Meanwhile block 0 notes more in its line: it writes word `w0` (a word it had not touched, or the word it read), or reads word
// `w0`. Block 1 is let go and finishes its read of word 1. What block 0 did meanwhile must be in the words: a read of a word
// block 0 wrote stops, and a write of a word block 0 read stops.
VTEST(a_record_its_block_moves_while_another_block_is_about_to_write_it_out_loses_nothing) {
  Hooked hooked;
  struct Case {
    uint64_t w0;
    bool w0_writes;   // block 0's access meanwhile
    bool probe_writes;   // the probe's access of w0, by block 1 again
  };
  const Case cases[] = {{3, true, false}, {0, true, false}, {15, true, true}, {7, false, true}, {2, false, true}};
  for (const Case& c : cases) {
    for (int round = 0; round < 4; ++round) {
      Lines d;
      d.mem.spec_begin(3, &d.by_word);
      VCHECK(!access(d, 0, 0, 0, false));   // block 0 reads word 0: the line is its record
      arm(kOutBefore);
      Access b(d, 1, 1, 1, false, true);   // block 1 reads word 1: it comes to block 0's record ...
      VCHECK(within(10000, [] { return g_parked.load() == 1; }));   // ... and is held before it takes it
      VCHECK(!access(d, 0, 0, c.w0, c.w0_writes));   // block 0, meanwhile (the line is not busy: it does not wait)
      g_release.store(1);
      VCHECK(!b.end());
      // Block 1 at w0 (round 0, 1) or block 2 on worker 2 (round 2, 3): stopped, by the word block 0 marked meanwhile.
      const unsigned worker = round < 2 ? 1 : 2;
      const uint64_t block = round < 2 ? 1 : 2;
      if (!access(d, worker, block, c.w0, c.probe_writes))
        throw vtest::Failure("word " + std::to_string(c.w0) + ": what block 0 did while block 1 was about to write its record out was lost");
      undone(d, d.mem.spec_end(), "a record moved before it was written out");
    }
  }
}

// The same window, and a third block writes the record out first (it writes word 5, which nobody had touched): the words then
// hold block 0's and block 2's marks, and block 1, let go, must not write the old record over them.
VTEST(a_record_another_block_wrote_out_first_is_not_written_out_again) {
  Hooked hooked;
  for (const bool probe_writes : {false, true}) {
    Lines d;
    d.mem.spec_begin(3, &d.by_word);
    VCHECK(!access(d, 0, 0, 0, false));
    arm(kOutBefore);
    Access b(d, 1, 1, 1, false, true);
    VCHECK(within(10000, [] { return g_parked.load() == 1; }));
    VCHECK(!access(d, 2, 2, 5, true));   // block 2 writes word 5: it writes block 0's record out, and takes word 5 as its writer
    g_release.store(1);
    VCHECK(!b.end());
    // Block 1 again, at word 5 (read: stopped, block 2 wrote it), or at word 0 (write: stopped, block 0 read it).
    if (!access(d, 1, 1, probe_writes ? 0 : 5, probe_writes))
      throw vtest::Failure(std::string("the mark of word ") + (probe_writes ? "0 (block 0's read)" : "5 (block 2's write)") + " was written over");
    undone(d, d.mem.spec_end(), "a record written out twice");
  }
}

// Block 1 has taken block 0's record to write it out (the line is busy) and is held there. A third block reads or writes in the
// line: it must wait for the words, then be stopped where they stop it (block 0 wrote word 3 and read word 9), and not where they
// do not (word 7 and word 9 for a read).
VTEST(a_block_that_finds_a_line_being_written_out_waits_for_the_words) {
  Hooked hooked;
  struct Case {
    uint64_t word;
    bool writes, stops;
  };
  const Case cases[] = {{3, false, true}, {3, true, true}, {9, true, true}, {7, false, false}, {7, true, false}, {9, false, false}};
  for (const Case& c : cases) {
    const std::string what = std::string(c.writes ? "write" : "read") + " of word " + std::to_string(c.word);
    Lines d;
    d.mem.spec_begin(3, &d.by_word);
    VCHECK(!access(d, 0, 0, 9, false));
    VCHECK(!access(d, 0, 0, 3, true));   // block 0's record: word 9 read, word 3 written
    arm(kOutBusy);
    Access b(d, 1, 1, 1, false, true);   // block 1 reads word 1, takes the record, and is held with the line busy
    VCHECK(within(10000, [] { return g_parked.load() == 1; }));
    Access c2(d, 2, 2, c.word, c.writes);
    // Block 2 must wait at the busy line (it cannot finish while block 1 holds it).
    const bool waited = within(10000, [&] { return g_waits.load() > 0 || c2.done.load(); }) && !c2.done.load();
    g_release.store(1);
    VCHECK(!b.end());
    const bool got = c2.end();
    if (!waited) throw vtest::Failure(what + ": block 2 did not wait for a line that was being written out");
    if (got != c.stops) throw vtest::Failure(what + (c.stops ? ": went on where a word stops it" : ": stopped where no word stops it"));
    const MemoryManager::SpecOutcome out = d.mem.spec_end();
    VCHECK_EQ(out.committed, !c.stops);
    VCHECK(!out.coarse);
    if (!c.stops) {
      VCHECK_EQ(out.stats.lines_by_word, uint64_t{1});
    } else if (d.all() != d.init) {
      throw vtest::Failure(what + ": a stopped trial left bytes behind");
    }
  }
}

// Block 0 has taken its own record to keep the line's bytes before its first write in it, and is held there (between taking the
// record and keeping the bytes). Another block that comes to the line must wait, and then find the write: a read of the word
// block 0 is writing stops, a read of another word does not.
VTEST(a_block_that_finds_a_line_whose_bytes_are_being_kept_waits_for_the_write) {
  Hooked hooked;
  struct Case {
    uint64_t word;
    bool writes, stops;
  };
  const Case cases[] = {{4, false, true}, {4, true, true}, {0, true, true}, {0, false, false}, {11, false, false}, {11, true, false}};
  for (const Case& c : cases) {
    const std::string what = std::string(c.writes ? "write" : "read") + " of word " + std::to_string(c.word);
    Lines d;
    d.mem.spec_begin(3, &d.by_word);
    VCHECK(!access(d, 0, 0, 0, false));   // block 0 read word 0
    arm(kKeepBusy);
    Access a(d, 0, 0, 4, true, true);    // block 0 writes word 4: takes the record, held before the line is kept
    VCHECK(within(10000, [] { return g_parked.load() == 1; }));
    Access c2(d, 2, 2, c.word, c.writes);
    const bool waited = within(10000, [&] { return g_waits.load() > 0 || c2.done.load(); }) && !c2.done.load();
    g_release.store(1);
    VCHECK(!a.end());
    const bool got = c2.end();
    if (!waited) throw vtest::Failure(what + ": block 2 did not wait for a line whose bytes were being kept");
    if (got != c.stops) throw vtest::Failure(what + (c.stops ? ": went on where a word stops it" : ": stopped where no word stops it"));
    const MemoryManager::SpecOutcome out = d.mem.spec_end();
    VCHECK_EQ(out.committed, !c.stops);
    if (c.stops && d.all() != d.init) throw vtest::Failure(what + ": a stopped trial left bytes behind");
    if (!c.stops) {
      // Kept: block 0's store is there, and the rest of the line as it was.
      std::vector<uint8_t> expect = d.init;
      const uint32_t v = 0xC0DE0000u;
      std::memcpy(expect.data() + (d.line + 16 - d.a), &v, 4);
      if (c.writes) {
        const uint32_t v2 = 0xC0DE0002u;
        std::memcpy(expect.data() + (d.line + 4 * c.word - d.a), &v2, 4);
      }
      if (d.all() != expect) throw vtest::Failure(what + ": the kept trial's bytes are not the stores");
    }
  }
}

// Block 0 is about to note more in its own record (it has read the record and not yet moved it) and is held there; meanwhile
// block 1 comes to the line and writes the record out. Block 0, let go, must not put its record back over the words: block 1's
// mark (it read word 1, or wrote word 1) and block 0's new one (word 2, read or written) must both be in the words afterwards.
VTEST(a_block_about_to_note_more_in_its_record_while_another_writes_it_out_loses_nothing) {
  Hooked hooked;
  for (const bool own_writes : {false, true}) {
    for (const bool other_writes : {false, true}) {
      for (const bool probe_other : {false, true}) {
        const std::string what = std::string("block 0 ") + (own_writes ? "writes" : "reads") + " word 2, block 1 " +
                                 (other_writes ? "writes" : "reads") + " word 1, block 2 at word " + (probe_other ? "1" : "2");
        Lines d;
        d.mem.spec_begin(3, &d.by_word);
        // Block 0's record: word 0 read, or written (the owner's later notes then go by the written path).
        VCHECK(!access(d, 0, 0, 0, own_writes));
        arm(kOwnBefore);
        Access a(d, 0, 0, 2, own_writes, true);
        VCHECK(within(10000, [] { return g_parked.load() == 1; }));
        VCHECK(!access(d, 1, 1, 1, other_writes));   // block 1 writes block 0's record out (the line is not busy)
        g_release.store(1);
        VCHECK(!a.end());
        // Block 2 writes the word (stopped by any mark), or reads it where it was written (stopped by a writer's mark).
        const bool written = probe_other ? other_writes : own_writes;
        const bool got = written && !probe_other ? access(d, 2, 2, 2, false) : access(d, 2, 2, probe_other ? 1 : 2, true);
        if (!got) throw vtest::Failure(what + ": a mark was lost");
        undone(d, d.mem.spec_end(), what);
      }
    }
  }
}

VTEST_MAIN
